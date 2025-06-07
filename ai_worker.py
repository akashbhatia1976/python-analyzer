#!/usr/bin/env python3
# ----------------------------------------------------------------
# ai_worker.py  –  Annotate radiology previews with OpenAI GPT-4o(Vision)
# ----------------------------------------------------------------

import os
import io
import json
import time
import textwrap
from datetime import datetime
from typing import List, Dict, Any
from urllib.parse import urlparse

import openai
from PIL import Image, ImageDraw, ImageFont
import boto3
from pymongo import MongoClient
from bson import ObjectId

# --------------------  CONFIG  --------------------------------
openai.api_key   = os.getenv("OPENAI_API_KEY")

MONGO_URI        = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME          = os.getenv("MONGO_DB", "medicalReportsTestDB")
COLL             = "imagingStudies"

AWS_REGION       = os.getenv("S3_REGION")
S3_BUCKET        = os.getenv("S3_BUCKET_NAME")
s3               = boto3.client(
    "s3",
    region_name           = AWS_REGION,
    aws_access_key_id     = os.getenv("S3_ACCESS_KEY_ID"),
    aws_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY"),
)

MODEL            = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
FONT_PATH        = os.getenv("CAPTION_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_SIZE        = int(os.getenv("CAPTION_FONT_SIZE", "20"))
POLL_INTERVAL    = int(os.getenv("WORKER_POLL_SECONDS", "10"))

mongo  = MongoClient(MONGO_URI)
coll   = mongo[DB_NAME][COLL]

# ------------------------------------------------------------
# Helper: fetch JPEG bytes from S3 using SDK rather than HTTP
# ------------------------------------------------------------
def fetch_jpeg_from_s3(presigned_url: str) -> bytes:
    parsed = urlparse(presigned_url)
    path_parts = parsed.path.lstrip("/").split("/", 1)
    # If bucket name is virtual-hosted (in host), path_parts[0] is key prefix
    if path_parts[0] == S3_BUCKET and len(path_parts) > 1:
        key = path_parts[1]
    else:
        key = parsed.path.lstrip("/")
    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return resp["Body"].read()

# ------------------------------------------------------------
# 1. OpenAI – per-image analysis
# ------------------------------------------------------------
def analyse_image(url: str) -> Dict[str, Any]:
    resp = openai.chat.completions.create(
        model       = MODEL,
        max_tokens  = 400,
        temperature = 0.2,
        messages    = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
                    {"type": "text", "text": (
                        "You are an experienced radiologist explaining a single image "
                            "to a non-specialist.\n\n"
                            "1. Describe main anatomical structures you recognise.\n"
                            "2. State whether the image is normal or abnormal.\n"
                            "3. If abnormal, list the key finding(s) and for each give up "
                            "to two POSSIBLE conditions (differential diagnosis) in plain English. "
                            "Use phrases like “could indicate …” or “may represent …”, "
                            "never a definitive diagnosis.\n"
                            "4. Keep the entire reply ≤80 words.\n\n"
                            "Return EXACTLY this JSON schema only:\n"
                            '{ "caption": "...", "findings": [ {"observation":"...",'
                            ' "possibleConditions":["...","..."]} ] }
                    )},
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

# ------------------------------------------------------------
# 2. Draw caption onto the JPEG
# ------------------------------------------------------------
def draw_caption(jpeg_bytes: bytes, caption: str) -> bytes:
    img  = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    wrapped = textwrap.fill(caption, width=50)
    # measure text
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = 10, img.height - h - 10

    # draw outline
    for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
        draw.multiline_text((x+dx, y+dy), wrapped, font=font, fill="black")
    # draw text
    draw.multiline_text((x, y), wrapped, font=font, fill="white")

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()

# ------------------------------------------------------------
# 3. Aggregate study summary
# ------------------------------------------------------------
def summarise_study(findings_arrays: List[List[Dict[str, Any]]]) -> str:
    prompt_json = json.dumps(findings_arrays, indent=2)
    resp = openai.chat.completions.create(
        model       = MODEL,
        max_tokens  = 250,
        temperature = 0.3,
        messages    = [{"role":"user","content":(
            f"Here are the findings arrays... {prompt_json}"
        )}],
    )
    return resp.choices[0].message.content.strip()

# ------------------------------------------------------------
# 4. Main processing per study
# ------------------------------------------------------------
def process_study(study: dict) -> None:
    study_id = str(study["_id"])
    user_id  = study["userId"]
    print(f"[{datetime.utcnow().isoformat()}] ➜  Processing study {study_id}")

    done = {e["url"] for e in study.get("aiInterpretation", {}).get("enhancedCaptions", [])}
    new_caps = []
    findings_lists = []

    for url in study["imageUrls"]:
        if url in done: continue
        # fetch
        try:
            jpeg_bytes = fetch_jpeg_from_s3(url)
        except Exception as e:
            print(f"❌ Failed to fetch {url}: {e}")
            continue
        # analyse
        try:
            analysis = analyse_image(url)
            caption  = analysis["caption"]
            findings = analysis["findings"]
            findings_lists.append(findings)
        except Exception as e:
            print(f"❌ OpenAI analysis failed for {url}: {e}")
            continue
        # draw
        annotated = draw_caption(jpeg_bytes, caption)
        # upload
        ts  = int(time.time()*1000)
        key = f"annotated/{user_id}/{ts}.jpg"
        try:
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=annotated, ContentType="image/jpeg", ACL="private")
            annotated_url = s3.generate_presigned_url("get_object", Params={"Bucket":S3_BUCKET,"Key":key}, ExpiresIn=3600)
        except Exception as e:
            print(f"❌ S3 upload failed for {key}: {e}")
            continue
        new_caps.append({"url":url, "caption":caption, "raw":analysis, "annotatedUrl":annotated_url, "timestamp":datetime.utcnow()})

    if not new_caps:
        print("No new images to annotate.")
        return

    # summary
    try:
        agg = summarise_study(findings_lists)
    except Exception:
        agg = ""

    coll.update_one({"_id":ObjectId(study_id)}, {
        "$push": {"aiInterpretation.enhancedCaptions":{"$each":new_caps},
                   "imageUrls":{"$each":[c['annotatedUrl'] for c in new_caps]}},
        "$set": {"aiInterpretation.aggregateSummary":agg, "aiInterpretation.updatedAt":datetime.utcnow()}
    })
    print(f"✅ Annotated {len(new_caps)} image(s) for study {study_id}")

# ------------------------------------------------------------
# 5. Polling loop
# ------------------------------------------------------------
def poll_forever() -> None:
    while True:
        try:
            cursor = coll.find({"analysisRequested":True,"aiInterpretation.enhancedCaptions":{"$exists":False}}, sort=[("uploadedAt",1)]).limit(5)
            studies = list(cursor)
            if not studies:
                time.sleep(POLL_INTERVAL)
                continue
            for s in studies: process_study(s)
        except Exception as exc:
            print(f"⚠️ Worker error: {exc}")
            time.sleep(5)

if __name__ == "__main__":
    print(f"[{datetime.utcnow().isoformat()}] 🟢 ai_worker started, polling every {POLL_INTERVAL}s.")
    poll_forever()

