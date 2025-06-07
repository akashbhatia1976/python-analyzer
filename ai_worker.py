#!/usr/bin/env python3
# ----------------------------------------------------------------
# ai_worker.py  –  Annotate radiology previews with OpenAI GPT-4o(Vision)
# ----------------------------------------------------------------

import os
import io
import json
import time
import textwrap
import traceback
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
    parts = parsed.path.lstrip("/").split("/", 1)
    key = parts[1] if parts[0] == S3_BUCKET and len(parts) > 1 else parsed.path.lstrip("/")
    return s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()

# ------------------------------------------------------------
# 1. OpenAI – per-image analysis with bounding boxes
# ------------------------------------------------------------
def analyse_image(url: str) -> Dict[str, Any]:
    prompt = f"""
You are an experienced radiologist explaining an image to a non-specialist.
URL: {url}
1. Describe main anatomical structures.
2. State normal or abnormal.
3. If abnormal, list findings with up to two possible conditions.
4. For each, provide bbox [x,y,width,height] normalized.
5. Reply ≤80 words.
Return ONLY this JSON:
{{
  "caption":"...",
  "findings":[{{"observation":"...","possibleConditions":["...","..."],"bbox":[x,y,w,h]}}]
}}
""".strip()
    try:
        resp = openai.chat.completions.create(
            model=MODEL,
            max_tokens=400,
            temperature=0.2,
            messages=[
                {"role":"user","content":[
                    {"type":"image_url","image_url":{"url":url,"detail":"high"}},
                    {"type":"text","text":prompt}
                ]}
            ],
            response_format={"type":"json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"❌ analyse_image failed for {url}: {repr(e)}")
        traceback.print_exc()
        raise

# ------------------------------------------------------------
# 2. Draw caption & highlight
# ------------------------------------------------------------
def draw_annotations(jpeg: bytes, caption: str, findings: List[Dict[str,Any]]) -> bytes:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    for f in findings:
        bbox = f.get("bbox")
        if bbox:
            x,y,bw,bh = bbox
            left, top = int(x*w), int(y*h)
            right,bot = int((x+bw)*w), int((y+bh)*h)
            draw.rectangle([left, top, right, bot], outline=(255,0,0,255), width=3)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()
    text = textwrap.fill(caption, width=50)
    tb = draw.multiline_textbbox((0,0), text, font=font)
    tx, ty = 10, h - (tb[3]-tb[1]) - 10
    for dx,dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
        draw.multiline_text((tx+dx,ty+dy), text, font=font, fill=(0,0,0,255))
    draw.multiline_text((tx,ty), text, font=font, fill=(255,255,255,255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

# ------------------------------------------------------------
# 3. Study summary
# ------------------------------------------------------------
def summarise_study(arrays: List[List[Dict[str,Any]]]) -> str:
    js = json.dumps(arrays, indent=2)
    resp = openai.chat.completions.create(
        model=MODEL,
        max_tokens=250,
        temperature=0.3,
        messages=[{"role":"user","content":f"Summary of findings: {js}"}],
    )
    return resp.choices[0].message.content.strip()

# ------------------------------------------------------------
# 4. Main processing per study
# ------------------------------------------------------------
def process_study(study: dict) -> None:
    sid, uid = str(study["_id"]), study["userId"]
    # init preview count and fail count on first run
    if study.get("previewCount") is None:
        count = len(study.get("imageUrls", []))
        coll.update_one({"_id":ObjectId(sid)}, {"$set":{"previewCount":count, "aiInterpretation.failCount":{{}}}})
        study["previewCount"] = count
        study.setdefault("aiInterpretation", {})["failCount"] = {}
    prev_count = study["previewCount"]

    done = {c["url"] for c in study.get("aiInterpretation", {}).get("enhancedCaptions", [])}
    fails = study.get("aiInterpretation", {}).get("failCount", {})
    new_caps, finds_list = [], []

    for url in study.get("imageUrls", []):
        if url in done:
            continue
        fail_count = fails.get(url, 0)
        if fail_count >= 3:
            print(f"⚠️ Skipping {url} after {fail_count} failures")
            continue

        try:
            imgb = fetch_jpeg_from_s3(url)
        except Exception as e:
            print(f"❌ fetch {url}: {e}")
            fails[url] = fail_count + 1
            coll.update_one({"_id":ObjectId(sid)}, {"$set":{f"aiInterpretation.failCount.{url}":fails[url]}})
            continue

        try:
            res = analyse_image(url)
            cap = res.get("caption", "")
            finds = res.get("findings", [])
            finds_list.append(finds)
        except Exception:
            fails[url] = fail_count + 1
            coll.update_one({"_id":ObjectId(sid)}, {"$set":{f"aiInterpretation.failCount.{url}":fails[url]}})
            continue

        try:
            ann = draw_annotations(imgb, cap, finds)
            key = f"annotated/{uid}/{int(time.time()*1000)}.jpg"
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=ann, ContentType="image/jpeg")
            ann_url = s3.generate_presigned_url("get_object", Params={"Bucket":S3_BUCKET,"Key":key}, ExpiresIn=3600)
        except Exception as e:
            print(f"❌ upload {url}: {e}")
            fails[url] = fail_count + 1
            coll.update_one({"_id":ObjectId(sid)}, {"$set":{f"aiInterpretation.failCount.{url}":fails[url]}})
            continue

        new_caps.append({"url":url,"caption":cap,"raw":res,"annotatedUrl":ann_url,"timestamp":datetime.utcnow()})

    if not new_caps:
        print("No new images to annotate.")
        return

    summary = ""
    try:
        summary = summarise_study(finds_list)
    except Exception as e:
        print(f"❌ summarise_study failed: {e}")

    coll.update_one({"_id":ObjectId(sid)}, {
        "$push": {"aiInterpretation.enhancedCaptions": {"$each":new_caps}},
        "$set": {"aiInterpretation.aggregateSummary":summary, "aiInterpretation.updatedAt":datetime.utcnow()}
    })

    # check if done
    doc = coll.find_one({"_id":ObjectId(sid)}, {"aiInterpretation.enhancedCaptions":1, "previewCount":1})
    cap_count = len(doc.get("aiInterpretation", {}).get("enhancedCaptions", []))
    if cap_count + sum(1 for c in doc.get("aiInterpretation", {}).get("failCount", {}).values() if c >= 3) >= doc.get("previewCount", 0):
        coll.update_one({"_id":ObjectId(sid)}, {"$set":{"analysisRequested":False}})
        print(f"🔒 Study {sid} fully annotated or skipped, stopping.")

    print(f"✅ Annotated {len(new_caps)} image(s) for study {sid}")

# ------------------------------------------------------------
# 5. Poll loop
# ------------------------------------------------------------
def poll_forever() -> None:
    while True:
        try:
            studies = list(coll.find({"analysisRequested":True}).limit(5))
            if not studies:
                time.sleep(POLL_INTERVAL)
                continue
            for s in studies:
                process_study(s)
        except Exception as e:
            print(f"⚠️ Worker error: {e}")
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    print(f"[{datetime.utcnow().isoformat()}] 🟢 Worker started, polling every {POLL_INTERVAL}s.")
    poll_forever()

