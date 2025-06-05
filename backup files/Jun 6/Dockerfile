
FROM python:3.11-slim

# System dependencies
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev poppler-utils && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy app files
COPY . .

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Flask port
EXPOSE 8080

# Start Flask app
CMD ["python", "app.py"]
