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
# Helper: fetch JPEG bytes from S3
# ------------------------------------------------------------
def fetch_jpeg_from_s3(presigned_url: str) -> bytes:
    parsed = urlparse(presigned_url)
    path_parts = parsed.path.lstrip("/").split("/", 1)
    key = path_parts[1] if path_parts[0] == S3_BUCKET and len(path_parts) > 1 else parsed.path.lstrip("/")
    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return resp["Body"].read()

# ------------------------------------------------------------
# 1. OpenAI – per-image analysis with bounding boxes
# ------------------------------------------------------------
def analyse_image(url: str) -> Dict[str, Any]:
    prompt = f"""
You are an experienced radiologist explaining a single image to a non-specialist.
Here is the image URL: {url}

1. Describe the main anatomical structures you recognize.
2. State whether the image is normal or abnormal.
3. If abnormal, list key finding(s) and for each give up to two POSSIBLE conditions.
   Use phrases like "could indicate …" or "may represent …", never definitive diagnosis.
4. For each finding, provide approximate bbox [x, y, width, height] normalized to image dimensions.
5. Keep reply ≤80 words.

Return EXACTLY this JSON schema only:
{{
  "caption": "...",
  "findings": [
    {{
      "observation": "...",
      "possibleConditions": ["...","..."],
      "bbox": [x, y, width, height]
    }}
  ]
}}
""".strip()

    resp = openai.chat.completions.create(
        model=MODEL,
        max_tokens=400,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
                    {"type": "text",      "text": prompt}
                ]
            }
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

# ------------------------------------------------------------
# 2. Draw caption & highlight boxes onto the JPEG
# ------------------------------------------------------------
def draw_annotations(jpeg_bytes: bytes, caption: str, findings: List[Dict[str, Any]]):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # draw semi-transparent red box for each finding
    for f in findings:
        bbox = f.get("bbox")
        if bbox:
            x, y, bw, bh = bbox
            left = int(x * w)
            top = int(y * h)
            right = int((x + bw) * w)
            bottom = int((y + bh) * h)
            draw.rectangle([left, top, right, bottom], outline=(255,0,0,255), width=3)

    # overlay caption text
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()
    text = textwrap.fill(caption, width=50)
    tb = draw.multiline_textbbox((0,0), text, font=font)
    tx, ty = 10, h - (tb[3] - tb[1]) - 10
    # text shadow
    for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
        draw.multiline_text((tx+dx, ty+dy), text, font=font, fill=(0,0,0,255))
    draw.multiline_text((tx, ty), text, font=font, fill=(255,255,255,255))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

# ------------------------------------------------------------
# 3. Study-level summary
# ------------------------------------------------------------
def summarise_study(findings_arrays: List[List[Dict[str, Any]]]) -> str:
    data = json.dumps(findings_arrays, indent=2)
    resp = openai.chat.completions.create(
        model=MODEL,
        max_tokens=250,
        temperature=0.3,
        messages=[{"role":"user","content":f"Here are the findings... {data}"}],
    )
    return resp.choices[0].message.content.strip()

# ------------------------------------------------------------
# 4. Main processing per study
# ------------------------------------------------------------
def process_study(study: dict):
    sid = str(study["_id"])
    uid = study["userId"]
    print(f"[{datetime.utcnow().isoformat()}] ➜ Processing {sid}")

    done = {c["url"] for c in study.get("aiInterpretation",{}).get("enhancedCaptions",[])}
    new_caps, findings_list = [], []

    for url in study.get("imageUrls",[]):
        if url in done: continue
        try:
            img_bytes = fetch_jpeg_from_s3(url)
        except Exception as e:
            print(f"❌ fetch {url}: {e}"); continue
        try:
            result = analyse_image(url)
            caption = result.get("caption","")
            finds   = result.get("findings",[])
            findings_list.append(finds)
        except Exception as e:
            print(f"❌ analyse {url}: {e}"); continue
        ann = draw_annotations(img_bytes, caption, finds)
        ts = int(time.time()*1000)
        key = f"annotated/{uid}/{ts}.jpg"
        try:
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=ann, ContentType="image/jpeg")
            ann_url = s3.generate_presigned_url("get_object", Params={"Bucket":S3_BUCKET,"Key":key}, ExpiresIn=3600)
        except Exception as e:
            print(f"❌ upload {key}: {e}"); continue
        new_caps.append({"url":url,"caption":caption,"raw":result,"annotatedUrl":ann_url,"timestamp":datetime.utcnow()})

    if not new_caps:
        print("No new images to annotate."); return

    try:
        summary = summarise_study(findings_list)
    except:
        summary = ""

    coll.update_one({"_id":ObjectId(sid)}, {
        "$push":{"aiInterpretation.enhancedCaptions":{"$each":new_caps}},
        "$set":{"aiInterpretation.aggregateSummary":summary,"aiInterpretation.updatedAt":datetime.utcnow()}
    })
    print(f"✅ Annotated {len(new_caps)} image(s) for study {sid}")

# ------------------------------------------------------------
# Poll loop
# ------------------------------------------------------------
def poll_forever():
    while True:
        try:
            cur = coll.find({"analysisRequested":True, "aiInterpretation.enhancedCaptions":{"$exists":False}}).limit(5)
            for s in list(cur): process_study(s)
        except Exception as e:
            print(f"⚠️ error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    print(f"[{datetime.utcnow().isoformat()}] 🟢 Worker started, polling every {POLL_INTERVAL}s.")
    poll_forever()

