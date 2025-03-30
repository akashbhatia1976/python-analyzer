
# Python Analyzer (with Tesseract + OpenAI)

## Setup

1. Deploy this folder as a **Docker-based Render Web Service**
2. Expose POST endpoint `/analyze` that accepts a PDF file
3. The script will:
   - Extract text via Tesseract
   - Process it using OpenAI
   - Return structured JSON

## Rollback Plan

If something doesn't work:
1. Switch `uploadRoutes.js` back to using `spawn('./venv/bin/python3', [...])`
2. Remove this service from Render

You can test locally with:

```bash
curl -X POST http://localhost:8080/analyze \
  -F "file=@your_report.pdf"
```
