FROM python:3.11-slim

# System dependencies (preserve existing tesseract/poppler, plus add libraries needed by Pillow)
RUN apt-get update && \
    apt-get install -y \
      tesseract-ocr \
      libtesseract-dev \
      poppler-utils \
      libjpeg-dev \
      zlib1g-dev \
      libpng-dev \
      libcairo2 \
      libpango1.0-0 \
      libgdk-pixbuf2.0-0 \
      libffi-dev && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy app files
COPY . .

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8080 (keep existing port)
EXPOSE 8080

# Start FastAPI (dicom2jpeg.py) using Uvicorn on port 8080
CMD ["uvicorn", "dicom2jpeg:app", "--host", "0.0.0.0", "--port", "8080"]

