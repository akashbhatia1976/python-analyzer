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





# Force output flushing
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

print("🚀 ============================================")
print("🚀 AI WORKER STARTING...")
print(f"🚀 Python version: {sys.version}")
print(f"🚀 Working directory: {os.getcwd()}")
print(f"🚀 Environment variables:")
print(f"  - OPENAI_API_KEY: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ MISSING'}")
print(f"  - MONGODB_URI: {'✅ SET' if os.getenv('MONGODB_URI') else '❌ MISSING'}")
print(f"  - MONGO_DB: {os.getenv('MONGO_DB', '❌ NOT SET')}")
print(f"  - S3_BUCKET_NAME: {'✅ SET' if os.getenv('S3_BUCKET_NAME') else '❌ MISSING'}")
print(f"  - S3_ACCESS_KEY_ID: {'✅ SET' if os.getenv('S3_ACCESS_KEY_ID') else '❌ MISSING'}")
print(f"  - S3_SECRET_ACCESS_KEY: {'✅ SET' if os.getenv('S3_SECRET_ACCESS_KEY') else '❌ MISSING'}")
print(f"  - S3_REGION: {os.getenv('S3_REGION', '❌ NOT SET')}")
print("🚀 ============================================")

print("📦 Importing dependencies...")
try:
    import io
    print("  ✅ io")
    import json
    print("  ✅ json")
    import time
    print("  ✅ time")
    import textwrap
    print("  ✅ textwrap")
    import traceback
    print("  ✅ traceback")
    import base64
    print("  ✅ base64")
    from datetime import datetime
    print("  ✅ datetime")
    from typing import List, Dict, Any
    print("  ✅ typing")
    from urllib.parse import urlparse
    print("  ✅ urllib.parse")

    import openai
    print("  ✅ openai")
    from PIL import Image, ImageDraw, ImageFont
    print("  ✅ PIL")
    import boto3
    print("  ✅ boto3")
    from pymongo import MongoClient
    print("  ✅ pymongo")
    from bson import ObjectId
    print("  ✅ bson")
    
    print("🎯 All dependencies imported successfully!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

# Rest of your existing code starting from CONFIG section...




# -------------------- CONFIG --------------------------------
openai.api_key   = os.getenv("OPENAI_API_KEY")
MONGO_URI        = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME          = os.getenv("MONGODB_DB", "medicalReportsTestDB")
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




#debug lines - to remove later

# Add this after: coll = mongo[DB_NAME][COLL]
try:
    print(f"🔍 MongoDB Debug Info:")
    print(f"  - Connection URI: {MONGO_URI}")
    print(f"  - Database name: {DB_NAME}")
    print(f"  - Collection: {COLL}")
    print(f"  - Available databases: {mongo.list_database_names()}")
    
    # Check if our database exists
    if DB_NAME in mongo.list_database_names():
        print(f"  ✅ Database '{DB_NAME}' exists")
        collections = mongo[DB_NAME].list_collection_names()
        print(f"  - Collections in {DB_NAME}: {collections}")
        
        if COLL in collections:
            print(f"  ✅ Collection '{COLL}' exists")
            total_docs = coll.count_documents({})
            print(f"  - Total documents in {COLL}: {total_docs}")
            
            # Check for any studies with analysisRequested
            requested_count = coll.count_documents({"analysisRequested": True})
            print(f"  - Studies with analysisRequested=True: {requested_count}")
            
            # Check for any studies with pending AI requests
            pending_count = coll.count_documents({
                "aiRequests": {"$elemMatch": {"status": "pending"}}
            })
            print(f"  - Studies with pending AI requests: {pending_count}")
            
            # Show a sample document
            sample = coll.find_one({})
            if sample:
                print(f"  - Sample document structure: {list(sample.keys())}")
        else:
            print(f"  ❌ Collection '{COLL}' not found")
    else:
        print(f"  ❌ Database '{DB_NAME}' not found")
        
except Exception as e:
    print(f"❌ Database debug failed: {e}")
    
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
        3. If abnormal, list findings with up to four possible conditions.
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
