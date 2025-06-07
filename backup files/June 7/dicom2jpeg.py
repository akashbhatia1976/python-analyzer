# python-ai-service/dicom2jpeg.py

import os
import io

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import pydicom
import numpy as np
from PIL import Image

# ------------------------------------------------------------------------------
# 1) Read AWS credentials and S3 bucket from environment variables
# ------------------------------------------------------------------------------
AWS_ACCESS_KEY_ID     = os.getenv("S3_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
AWS_REGION            = os.getenv("S3_REGION")
S3_BUCKET_NAME        = os.getenv("S3_BUCKET_NAME")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME]):
    raise RuntimeError("Missing one of S3_* environment variables in Python service")

# boto3 S3 client (same IAM user that Node uses, or a read-only user)
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# ------------------------------------------------------------------------------
# 2) FastAPI app
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Aether Imaging AI Service",
    description="Fetch DICOM from S3, convert → JPEG, run AI, return findings",
    version="0.1.0",
)

# Add this root route so HEAD / or GET / returns 200 instead of 404
@app.get("/")
async def root():
    return {"message": "Aether AI service is running"}

# ------------------------------------------------------------------------------
# 3) Request & Response Models
# ------------------------------------------------------------------------------

class InterpretRequest(BaseModel):
    """
    Called by Node: provides the S3 key of the DICOM file to interpret.
    """
    dicomS3Key: str     # e.g. "dicom/user123/1684714978123-chest1.dcm"
    studyId: str        # the Mongo studyId (just so Node can match results)


class Finding(BaseModel):
    label: str
    confidence: float


class InterpretResponse(BaseModel):
    """
    Sent back to Node: a list of findings, a confidence score, and a presigned JPEG preview URL.
    """
    findings: list[Finding]
    confidence: float
    previewUrl: str


# ------------------------------------------------------------------------------
# 4) Helper: Convert DICOM → JPEG Bytes AND Upload Preview to S3
# ------------------------------------------------------------------------------
def dicom_s3_to_jpeg_bytes_and_upload(dicom_key: str) -> tuple[bytes, str]:
    """
    1) Download the DICOM object from S3 (using its key).
    2) Load pixel data via pydicom.
    3) Normalize to 8-bit grayscale.
    4) Write to JPEG via Pillow.
    5) Upload that JPEG under 'previews/...' in S3.
    6) Return (jpeg_bytes, preview_url).
    """
    print(f"🔎 Python: fetching from S3 bucket={S3_BUCKET_NAME}, key={dicom_key}")
    try:
        obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=dicom_key)
    except s3.exceptions.NoSuchKey:
        print(f"❌ Python: S3 returned NoSuchKey for bucket={S3_BUCKET_NAME}, key={dicom_key}")
        raise HTTPException(status_code=404, detail=f"DICOM not found: {dicom_key}")

    dicom_bytes = obj["Body"].read()
    dataset = pydicom.dcmread(io.BytesIO(dicom_bytes))
    pixel_array = dataset.pixel_array.astype(np.float32)
    min_val, max_val = float(np.min(pixel_array)), float(np.max(pixel_array))
    if max_val - min_val == 0:
        scaled = np.zeros_like(pixel_array, dtype=np.uint8)
    else:
        scaled = ((pixel_array - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

    image = Image.fromarray(scaled)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    jpeg_bytes = buffer.getvalue()
    buffer.close()

    # Build a “previews/…” key from the original DICOM key:
    # e.g. dicom/user123/1234-abc.dcm → previews/user123/1234-abc.jpg
    preview_key = dicom_key.replace("dicom/", "previews/").replace(".dcm", ".jpg")

    # Upload that JPEG back to S3
    s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=preview_key,
        Body=jpeg_bytes,
        ContentType="image/jpeg",
        ACL="private"
    )
    print(f"✅ Python: uploaded preview JPEG to S3 key={preview_key}")

    # Generate a presigned URL (valid for 1 hour)
    preview_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": preview_key},
        ExpiresIn=3600
    )

    return jpeg_bytes, preview_url


# ------------------------------------------------------------------------------
# 5) (Stub) AI Inference on JPEG Bytes
#    Replace this with a real model later.
# ------------------------------------------------------------------------------
def run_dummy_ai_inference(jpeg_bytes: bytes) -> tuple[list[dict], float]:
    """
    Placeholder: pretend we ran a CNN and found "Pneumonia" with 0.82 confidence.
    Return ([{"label": "...", "confidence": 0.82}, ...], overall_confidence).
    """
    findings = [
        {"label": "Pneumonia", "confidence": 0.82},
        {"label": "Pleural Effusion", "confidence": 0.45},
    ]
    overall_confidence = float(max(f["confidence"] for f in findings))
    return findings, overall_confidence


# ------------------------------------------------------------------------------
# 6) POST /interpret Endpoint
# ------------------------------------------------------------------------------
@app.post("/interpret", response_model=InterpretResponse)
async def interpret(request: InterpretRequest):
    """
    Node will POST JSON: { "dicomS3Key": "dicom/xxx.dcm", "studyId": "..." }
    We: fetch the DICOM from S3, convert → JPEG, upload preview, run AI, and return findings + preview URL.
    """
    dicom_key = request.dicomS3Key
    study_id = request.studyId  # for logging/debug, not strictly needed here

    # 6a) Convert from S3 DICOM → in-memory JPEG and upload preview
    jpeg_bytes, preview_url = dicom_s3_to_jpeg_bytes_and_upload(dicom_key)

    # 6b) Run (dummy) AI inference on jpeg_bytes
    findings, overall_confidence = run_dummy_ai_inference(jpeg_bytes)

    # 6c) Return JSON to Node
    return InterpretResponse(
        findings=[Finding(**f) for f in findings],
        confidence=overall_confidence,
        previewUrl=preview_url
    )


# ------------------------------------------------------------------------------
# 7) Health Check Endpoint (Optional)
# ------------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "python-ai-service"}

