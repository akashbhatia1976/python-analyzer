#!/usr/bin/env python3
# ----------------------------------------------------------------
# ai_worker.py  –  Annotate radiology previews via aiRequests queue in MongoDB
# ----------------------------------------------------------------

import os
import io
import json
import time
import textwrap
import traceback
import base64
from datetime import datetime
from typing import List, Dict, Any
from urllib.parse import urlparse

import openai
from PIL import Image, ImageDraw, ImageFont
import boto3
from pymongo import MongoClient
from bson import ObjectId

# -------------------- CONFIG --------------------------------
openai.api_key   = os.getenv("OPENAI_API_KEY")
MONGO_URI        = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME          = os.getenv("MONGO_DB", "medicalReportsTestDB")
COLL             = "imagingStudies"

AWS_REGION       = os.getenv("S3_REGION")
S3_BUCKET        = os.getenv("S3_BUCKET_NAME")
s3 = boto3.client(
    "s3",
    region_name           = AWS_REGION,
    aws_access_key_id     = os.getenv("S3_ACCESS_KEY_ID"),
    aws_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY"),
)

MODEL         = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
FONT_PATH     = os.getenv("CAPTION_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_SIZE     = int(os.getenv("CAPTION_FONT_SIZE", "20"))
POLL_INTERVAL = int(os.getenv("WORKER_POLL_SECONDS", "10"))

mongo = MongoClient(MONGO_URI)
coll  = mongo[DB_NAME][COLL]

# ------------------------------------------------------------
# 1. Fetch JPEG bytes from S3 by key
# ------------------------------------------------------------
def fetch_jpeg_by_key(key: str) -> bytes:
    return s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()

# ------------------------------------------------------------
# 2. Analyse image bytes via OpenAI
# ------------------------------------------------------------
def analyse_image_bytes(img_bytes: bytes, source_desc: str = None) -> Dict[str, Any]:
    data_uri = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode('utf-8')
    prompt = textwrap.dedent(f"""
        You are an experienced radiologist explaining an image to a non-specialist.
        URL: {source_desc or data_uri}
        1. Describe main anatomical structures.
        2. State whether normal or abnormal.
        3. If abnormal, list findings with up to two possible conditions.
        4. For each, provide bbox [x,y,width,height] normalized.
        5. Reply ≤80 words.
        Return ONLY this JSON:
        {{
          "caption":"...",
          "findings":[{{"observation":"...","possibleConditions":["...","..."],"bbox":[x,y,w,h]}}]
        }}
    """
    ).strip()
    try:
        resp = openai.chat.completions.create(
            model=MODEL,
            max_tokens=400,
            temperature=0.2,
            messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                    {"type": "text",      "text": prompt}
                ]}
            ],
            response_format={"type": "json_object"}
        )
        raw = resp.choices[0].message.content
        if isinstance(raw, str):
            return json.loads(raw)
        return raw
    except Exception as e:
        print(f"❌ analyse_image_bytes failed: {e}")
        traceback.print_exc()
        raise

# ------------------------------------------------------------
# 3. Draw caption & highlight bounding boxes
# ------------------------------------------------------------
def draw_annotations(jpeg: bytes, caption: str, findings: List[Dict[str, Any]]) -> bytes:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    for f in findings:
        bbox = f.get("bbox", [0, 0, 0, 0])
        x, y, bw, bh = bbox
        draw.rectangle([
            int(x * w), int(y * h),
            int((x + bw) * w), int((y + bh) * h)
        ], outline=(255, 0, 0, 255), width=3)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()
    draw.multiline_text((10, h - FONT_SIZE - 10), caption, font=font, fill=(255, 255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

# ------------------------------------------------------------
# 4. Summarize study findings
# ------------------------------------------------------------
def summarise_study(arrays: List[List[Dict[str, Any]]]) -> str:
    js = json.dumps(arrays, indent=2)
    resp = openai.chat.completions.create(
        model=MODEL,
        max_tokens=250,
        temperature=0.3,
        messages=[{"role": "user", "content": f"Provide a brief summary in JSON: {js}"}],
    )
    return resp.choices[0].message.content.strip()

# ------------------------------------------------------------
# 5. Process a single pending aiRequests entry
# ------------------------------------------------------------
def process_study_request(study: dict) -> None:
    sid = study["_id"]
    pending = next((r for r in study.get("aiRequests", []) if r.get("status") == "pending"), None)
    if not pending:
        return
    req_ts = pending.get("requestedAt")

    new_caps, all_findings = [], []
    for key in study.get("previewKeys", []):
        try:
            img_bytes = fetch_jpeg_by_key(key)
            res = analyse_image_bytes(img_bytes, source_desc=key)
            cap = res.get("caption", "")
            finds = res.get("findings", [])
            ann_bytes = draw_annotations(img_bytes, cap, finds)
            ann_key = f"annotated/{study['userId']}/{int(time.time() * 1000)}.jpg"
            s3.put_object(Bucket=S3_BUCKET, Key=ann_key, Body=ann_bytes, ContentType="image/jpeg")
            ann_url = s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": ann_key}, ExpiresIn=3600)
            new_caps.append({"url": ann_url, "caption": cap, "annotatedUrl": ann_url, "timestamp": datetime.utcnow()})
            all_findings.append(finds)
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            traceback.print_exc()
            continue

    if not new_caps:
        return

    summary = summarise_study(all_findings)
    coll.update_one(
        {"_id": ObjectId(sid), "aiRequests.requestedAt": req_ts},
        {"$set": {
            "aiRequests.$.interpretation": {"enhancedCaptions": new_caps, "aggregateSummary": summary},
            "aiRequests.$.status": "completed",
            "aiRequests.$.completedAt": datetime.utcnow()
        }}
    )

    remaining = coll.count_documents({
        "_id": ObjectId(sid),
        "aiRequests": {"$elemMatch": {"status": "pending"}}
    })
    if remaining == 0:
        coll.update_one({"_id": ObjectId(sid)}, {"$set": {"analysisRequested": False}})

    print(f"✅ Processed request {req_ts}; {remaining} pending left.")

# ------------------------------------------------------------
# 6. Poll loop
# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"[{datetime.utcnow().isoformat()}] 🟢 Worker started, polling every {POLL_INTERVAL}s.")
    while True:
        try:
            studies = list(coll.find({
                "analysisRequested": True,
                "aiRequests": {"$elemMatch": {"status": "pending"}}
            }).limit(5))
            print(f"[Worker] ⏱ polled, found {len(studies)} pending requests.")
            for s in studies:
                process_study_request(s)
        except Exception as e:
            print(f"⚠️ Worker error: {e}")
            traceback.print_exc()
        time.sleep(POLL_INTERVAL)

