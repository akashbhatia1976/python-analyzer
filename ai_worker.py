#!/usr/bin/env python3
# ------------------------------------------------------------
# ai_worker.py  –  Annotate radiology previews with OpenAI GPT-4o(Vision)
# ------------------------------------------------------------
import os
import io
import json
import time
import textwrap
import tempfile
from datetime import datetime
from typing import List, Dict, Any

import openai
import requests
from PIL import Image, ImageDraw, ImageFont
import boto3
from pymongo import MongoClient
from bson import ObjectId

# --------------------  CONFIG  --------------------------------
openai.api_key   = os.getenv("OPENAI_API_KEY")

MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME          = os.getenv("MONGO_DB", "aether")
COLL             = "imagingStudies"

AWS_REGION       = os.getenv("S3_REGION")
S3_BUCKET        = os.getenv("S3_BUCKET_NAME")
s3               = boto3.client(
    "s3",
    region_name      = AWS_REGION,
    aws_access_key_id     = os.getenv("S3_ACCESS_KEY_ID"),
    aws_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY"),
)

MODEL            = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")  # or gpt-4o
FONT_PATH        = os.getenv("CAPTION_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_SIZE        = 20
POLL_INTERVAL    = int(os.getenv("WORKER_POLL_SECONDS", "10"))

# --------------------  DB  ------------------------------------
mongo  = MongoClient(MONGO_URI)
coll   = mongo[DB_NAME][COLL]

# ------------------------------------------------------------
# 1.  OpenAI - per-image analysis
# ------------------------------------------------------------
def analyse_image(url: str) -> Dict[str, Any]:
    """Returns JSON: { caption:str, findings:[{observation, possibleConditions:[..]}] }"""
    resp = openai.chat.completions.create(
        model       = MODEL,
        max_tokens  = 400,
        temperature = 0.2,
        messages    = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": url, "detail": "high"}
                    },
                    {
                        "type": "text",
                        "text": (
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
                            ' "possibleConditions":["...","..."]} ] }'
                        )
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

# ------------------------------------------------------------
# 2.  Draw caption on JPEG
# ------------------------------------------------------------
def draw_caption(jpeg_bytes: bytes, caption: str) -> bytes:
    img   = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    draw  = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    wrapped = textwrap.fill(caption, width=50)
    w, h    = draw.multiline_textsize(wrapped, font=font)
    x, y    = 10, img.height - h - 10

    # black outline
    draw.multiline_text((x-1, y-1), wrapped, font=font, fill="black")
    draw.multiline_text((x+1, y-1), wrapped, font=font, fill="black")
    draw.multiline_text((x-1, y+1), wrapped, font=font, fill="black")
    draw.multiline_text((x+1, y+1), wrapped, font=font, fill="black")
    # white text
    draw.multiline_text((x, y), wrapped, font=font, fill="white")

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()

# ------------------------------------------------------------
# 3.  Aggregate study-level summary
# ------------------------------------------------------------
def summarise_study(findings_arrays: List[List[Dict[str, Any]]]) -> str:
    prompt_json = json.dumps(findings_arrays, indent=2)
    resp = openai.chat.completions.create(
        model       = MODEL,
        max_tokens  = 250,
        temperature = 0.3,
        messages    = [
            {
                "role": "user",
                "content": (
                    "You are reviewing multiple annotated radiology images.\n"
                    f"Here are the findings arrays for each image:\n{prompt_json}\n\n"
                    "Write a concise layperson-friendly report (≤150 words):\n"
                    "• Summarise overall state (normal or abnormal).\n"
                    "• Group related findings.\n"
                    "• Mention the MOST LIKELY conditions using cautious language "
                    "(e.g., “could indicate…”).\n"
                    "• Finish with: “These observations are informational and "
                    "not a substitute for professional diagnosis.”"
                )
            }
        ],
    )
    return resp.choices[0].message.content.strip()

# ------------------------------------------------------------
# 4.  Main study processor
# ------------------------------------------------------------
def process_study(study: dict) -> None:
    study_id = str(study["_id"])
    user_id  = study["userId"]
    print(f"[{datetime.utcnow().isoformat()}] ➜  Processing study {study_id}")

    # ---- collect URLs already analysed so we skip duplicates
    already_done = { au["url"] for au in study.get("aiInterpretation", {}).get("enhancedCaptions", []) }

    new_captions   = []        # [{url, caption, raw, annotatedUrl}]
    findings_lists = []        # list of findings[] arrays

    for url in study["imageUrls"]:
        if url in already_done:
            continue

        # 1) Download JPEG
        img_resp = requests.get(url, timeout=30)
        img_resp.raise_for_status()
        jpeg_bytes = img_resp.content

        # 2) Analyse with OpenAI
        analysis   = analyse_image(url)
        caption    = analysis["caption"]
        findings_lists.append(analysis["findings"])

        # 3) Draw caption → annotated JPEG
        annotated  = draw_caption(jpeg_bytes, caption)

        # 4) Upload annotated
        ts         = int(time.time() * 1000)
        key        = f"annotated/{user_id}/{ts}.jpg"
        s3.put_object(
            Bucket      = S3_BUCKET,
            Key         = key,
            Body        = annotated,
            ContentType = "image/jpeg",
            ACL         = "private",
        )
        annotated_url = s3.generate_presigned_url(
            "get_object",
            Params  = {"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn = 3600,
        )

        new_captions.append({
            "url": url,
            "caption": caption,
            "raw": analysis,
            "annotatedUrl": annotated_url,
            "timestamp": datetime.utcnow()
        })

    if not new_captions:
        print("No new images to annotate.")
        return

    # ---- Build / update interpretation blob
    agg_summary = summarise_study(findings_lists)

    coll.update_one(
        {"_id": ObjectId(study_id)},
        {
            "$push": {
                "aiInterpretation.enhancedCaptions": {"$each": new_captions},
                "imageUrls": {"$each": [c["annotatedUrl"] for c in new_captions]}
            },
            "$set": {
                "aiInterpretation.aggregateSummary": agg_summary,
                "aiInterpretation.updatedAt": datetime.utcnow()
            }
        }
    )
    print(f"✅  Annotated {len(new_captions)} image(s) for study {study_id}")

# ------------------------------------------------------------
# 5.  Polling loop
# ------------------------------------------------------------
def poll_forever() -> None:
    while True:
        try:
            # grab studies that have DICOMKeys but no enhancedCaptions yet
            cursor = coll.find(
                { "aiInterpretation.enhancedCaptions": { "$exists": False } },
                sort=[("uploadedAt", 1)]
            ).limit(5)

            to_process = list(cursor)
            if not to_process:
                time.sleep(POLL_INTERVAL)
                continue

            for s in to_process:
                process_study(s)

        except Exception as exc:
            print("⚠️  Worker error:", exc)
            time.sleep(5)

# ------------------------------------------------------------
if __name__ == "__main__":
    poll_forever()
