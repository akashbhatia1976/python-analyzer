#!/usr/bin/env python3
# ----------------------------------------------------------------
# ai_worker.py  –  Annotate radiology previews via aiRequests queue
# ----------------------------------------------------------------

import os
import io
import time
import traceback
from datetime import datetime
from typing import List, Dict, Any
from urllib.parse import urlparse

import openai
from PIL import Image, ImageDraw, ImageFont
import boto3
from pymongo import MongoClient
from bson import ObjectId

# -------------------- CONFIG --------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
MONGO_URI      = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME        = os.getenv("MONGO_DB", "medicalReportsTestDB")
COLL           = "imagingStudies"

AWS_REGION     = os.getenv("S3_REGION")
S3_BUCKET      = os.getenv("S3_BUCKET_NAME")
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
# Helper: fetch JPEG bytes from S3
# ------------------------------------------------------------
def fetch_jpeg_from_s3(presigned_url: str) -> bytes:
    parsed = urlparse(presigned_url)
    parts = parsed.path.lstrip("/").split("/", 1)
    key = parts[1] if parts[0] == S3_BUCKET and len(parts) > 1 else parsed.path.lstrip("/")
    return s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()

# ------------------------------------------------------------
# 1. OpenAI – per-image analysis
# ------------------------------------------------------------
def analyse_image(url: str) -> Dict[str, Any]:
    prompt = f"You are a radiologist. Analyze: {url}"  # simplified for brevity
    resp = openai.chat.completions.create(
        model=MODEL,
        max_tokens=400,
        temperature=0.2,
        messages=[{"role":"user","content":prompt}],
        response_format={"type":"json_object"},
    )
    return resp.choices[0].message.content

# ------------------------------------------------------------
# 2. Draw caption & bbox
# ------------------------------------------------------------
def draw_annotations(jpeg: bytes, caption: str, findings: List[Dict[str,Any]]) -> bytes:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    for f in findings:
        x,y,bw,bh = f.get("bbox", [0,0,0,0])
        left, top = int(x*w), int(y*h)
        right,bot = int((x+bw)*w), int((y+bh)*h)
        draw.rectangle([left, top, right, bot], outline=(255,0,0,255), width=3)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()
    draw.text((10, h-30), caption, font=font, fill=(255,255,255,255))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

# ------------------------------------------------------------
# 3. Summarize findings
# ------------------------------------------------------------
def summarise_study(arrays: List[List[Dict[str,Any]]]) -> str:
    js = json.dumps(arrays)
    resp = openai.chat.completions.create(
        model=MODEL,
        max_tokens=250,
        temperature=0.3,
        messages=[{"role":"user","content":f"Summary: {js}"}],
    )
    return resp.choices[0].message.content

# ------------------------------------------------------------
# 4. Process single pending request
# ------------------------------------------------------------
def process_study_request(study: dict) -> None:
    sid = study['_id']
    pending = next((r for r in study.get('aiRequests', []) if r.get('status') == 'pending'), None)
    if not pending: return
    req_ts = pending['requestedAt']

    new_caps, all_findings = [], []
    for url in study.get('imageUrls', []):
        try:
            imgb = fetch_jpeg_from_s3(url)
            res = analyse_image(url)
            cap = res.get('caption',''); finds = res.get('findings',[])
            img_ann = draw_annotations(imgb, cap, finds)
            key = f"annotated/{study['userId']}/{int(time.time()*1000)}.jpg"
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=img_ann, ContentType='image/jpeg')
            ann_url = s3.generate_presigned_url('get_object', Params={'Bucket':S3_BUCKET,'Key':key}, ExpiresIn=3600)
            new_caps.append({'url':url,'caption':cap,'annotatedUrl':ann_url,'timestamp':datetime.utcnow()})
            all_findings.append(finds)
        except Exception as e:
            print(f"Error processing {url}: {e}"); continue

    if not new_caps:
        return

    summary = summarise_study(all_findings)
    coll.update_one(
        {'_id': ObjectId(sid), 'aiRequests.requestedAt': req_ts},
        {'$set':{
            'aiRequests.$.interpretation':{'enhancedCaptions':new_caps,'aggregateSummary':summary},
            'aiRequests.$.status':'completed','aiRequests.$.completedAt':datetime.utcnow()
        }}
    )
    # clear flag if no pending
    rem = coll.count_documents({'_id':ObjectId(sid),'aiRequests':{'$elemMatch':{'status':'pending'}}})
    if rem==0:
        coll.update_one({'_id':ObjectId(sid)},{'$set':{'analysisRequested':False}})
    print(f"Processed request {req_ts}; {rem} pending left.")

# ------------------------------------------------------------
# 5. Poll loop
# ------------------------------------------------------------
if __name__=='__main__':
    print(f"Worker started, polling every {POLL_INTERVAL}s.")
    while True:
        try:
            studies = list(coll.find({'analysisRequested':True,'aiRequests':{'$elemMatch':{'status':'pending'}}}).limit(5))
            print(f"[Worker] Polled, {len(studies)} pending.")
            for s in studies: process_study_request(s)
        except Exception as e:
            print(f"Worker error: {e}"); traceback.print_exc()
        time.sleep(POLL_INTERVAL)

