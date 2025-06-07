# python-ai-service/dicom2jpeg.py

import os
import io
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import pydicom
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# 1) Read AWS credentials and S3 bucket from environment variables
# ----------------------------------------------------------------------------
AWS_ACCESS_KEY_ID     = os.getenv("S3_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
AWS_REGION            = os.getenv("S3_REGION")
S3_BUCKET_NAME        = os.getenv("S3_BUCKET_NAME")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME]):
    raise RuntimeError("Missing one of S3_* environment variables in Python service")

# boto3 S3 client
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# ----------------------------------------------------------------------------
# 2) FastAPI app
# ----------------------------------------------------------------------------
app = FastAPI(
    title="Aether Imaging JPEG Service",
    description="Fetch DICOM from S3, convert to JPEG, upload preview, return preview URL",
    version="0.2.0",
)

@app.get("/")
async def root():
    return {"message": "Aether JPEG service is running"}
    
@app.head("/")
async def head_root():
    return Response(status_code=200)

# ----------------------------------------------------------------------------
# 3) Request & Response Models
# ----------------------------------------------------------------------------
class PreviewRequest(BaseModel):
    dicomS3Key: str   # e.g. "dicom/user123/1684714978123-chest1.dcm"
    studyId: str      # MongoDB studyId as hex string

class PreviewResponse(BaseModel):
    previewUrl: str

# ----------------------------------------------------------------------------
# 4) Helper: Convert DICOM → JPEG Bytes AND Upload Preview to S3
# ----------------------------------------------------------------------------
def dicom_s3_to_jpeg_and_upload(dicom_key: str, study_id: str) -> tuple[bytes, str]:
    # fetch DICOM
    try:
        obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=dicom_key)
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"DICOM not found: {dicom_key}")

    dicom_bytes = obj["Body"].read()
    ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
    arr = ds.pixel_array.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    img8 = ((arr - mn) / (mx - mn) * 255).clip(0,255).astype(np.uint8) if mx > mn else np.zeros_like(arr, dtype=np.uint8)

    img = Image.fromarray(img8)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()
    buf.close()

    # shard by studyId hash to avoid S3 hot-spotting
    shard = int(study_id[:8], 16) % 1000
    timestamp = int(time.time() * 1000)
    preview_key = f"previews/{shard}/{study_id}/{timestamp}.jpg"

    # upload to S3
    s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=preview_key,
        Body=jpeg_bytes,
        ContentType="image/jpeg",
        ACL="private"
    )

    # generate presigned URL for display
    preview_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": preview_key},
        ExpiresIn=3600
    )
    return jpeg_bytes, preview_url

# ----------------------------------------------------------------------------
# 5) POST /preview Endpoint
# ----------------------------------------------------------------------------
@app.post("/preview", response_model=PreviewResponse)
async def preview(request: PreviewRequest):
    _, url = dicom_s3_to_jpeg_and_upload(request.dicomS3Key, request.studyId)
    return PreviewResponse(previewUrl=url)

# ----------------------------------------------------------------------------
# 6) Health Check
# ----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "python-ai-service"}

