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
    
    # Only draw bounding boxes if there are findings with bboxes
    for f in findings:
        bbox = f.get("bbox", [])
        if len(bbox) == 4:  # Ensure valid bbox
            x, y, bw, bh = bbox
            # Convert normalized coordinates to pixel coordinates
            x1, y1 = int(x * w), int(y * h)
            x2, y2 = int((x + bw) * w), int((y + bh) * h)
            
            # Draw red rectangle for abnormal findings
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)
    
    # Add caption at bottom of image
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    # Wrap text if too long
    wrapped_text = textwrap.fill(caption, width=80)
    draw.multiline_text((10, h - 60), wrapped_text, font=font, fill=(255, 255, 255, 255))
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

# ------------------------------------------------------------
# 4. Summarize study findings - FIXED to return plain text
# ------------------------------------------------------------
def summarise_study(arrays: List[List[Dict[str, Any]]]) -> str:
    # Flatten all findings into a single list
    all_findings = []
    for finding_array in arrays:
        all_findings.extend(finding_array)
    
    if not all_findings:
        return "No significant abnormalities detected in this study."
    
    # Create a summary prompt that returns plain text
    findings_text = []
    for i, finding in enumerate(all_findings, 1):
        obs = finding.get("observation", "")
        conditions = finding.get("possibleConditions", [])
        if obs:
            findings_text.append(f"Finding {i}: {obs}")
            if conditions:
                findings_text.append(f"Possible conditions: {', '.join(conditions)}")
    
    findings_summary = "\n".join(findings_text)
    
    prompt = f"""
    Based on these radiological findings, provide a concise medical summary in plain text (not JSON):
    
    {findings_summary}
    
    Provide a brief, professional summary suitable for a medical report. Use clear, medical terminology but keep it accessible. Limit to 3-4 sentences.
    """
    
    try:
        resp = openai.chat.completions.create(
            model=MODEL,
            max_tokens=200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        # Fallback to simple text summary
        return f"Analysis completed for {len(all_findings)} findings. " + findings_summary

# ------------------------------------------------------------
# 5. Process a single pending aiRequests entry - FIXED URLs
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
            # Fetch original image
            img_bytes = fetch_jpeg_by_key(key)
            
            # Analyze with AI
            res = analyse_image_bytes(img_bytes, source_desc=key)
            cap = res.get("caption", "")
            finds = res.get("findings", [])
            
            # Create annotated version
            ann_bytes = draw_annotations(img_bytes, cap, finds)
            ann_key = f"annotated/{study['userId']}/{int(time.time() * 1000)}.jpg"
            s3.put_object(Bucket=S3_BUCKET, Key=ann_key, Body=ann_bytes, ContentType="image/jpeg")
            
            # Generate URLs for both original and annotated
            orig_url = s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=3600)
            ann_url = s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": ann_key}, ExpiresIn=3600)
            
            # FIXED: Correct URLs and include findings
            new_caps.append({
                "url": orig_url,           # ← Original image URL (no annotations)
                "annotatedUrl": ann_url,   # ← Annotated image URL (with boxes)
                "caption": cap,
                "raw": {"findings": finds}, # ← Include findings for frontend
                "timestamp": datetime.utcnow()
            })
            all_findings.append(finds)
            
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            traceback.print_exc()
            continue

    if not new_caps:
        return

    # Generate plain text summary
    summary = summarise_study(all_findings)
    
    # Update database
    coll.update_one(
        {"_id": ObjectId(sid), "aiRequests.requestedAt": req_ts},
        {"$set": {
            "aiRequests.$.interpretation": {
                "enhancedCaptions": new_caps,
                "aggregateSummary": summary  # Now plain text, not JSON
            },
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
