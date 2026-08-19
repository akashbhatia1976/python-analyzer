import os
import json
import re
import requests
import tempfile
import time
from datetime import datetime, date
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from PIL import Image, ImageOps


try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    print("🟢 HEIF/HEIC support enabled", flush=True)
except Exception as e:
    print("⚠️ HEIF support not available:", e, flush=True)


# --- Load environment
load_dotenv()

# --- OpenAI + retry config (env-driven) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_RETRIES = int(os.getenv("OPENAI_RETRIES", "3"))
RETRY_BASE_MS  = int(os.getenv("OPENAI_RETRY_BASE_MS", "1500"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}

REQUIRED_CATEGORIES = ["Patient Information", "Medical Parameters", "Doctor's Notes"]


# --- Synonym & Category Mappings
with open("data/synonyms.json", "r") as f:
    nested_synonyms = json.load(f)

with open("data/categories_map.json", "r") as f:
    categories_map = json.load(f)

synonyms_flat = {}
for category, entries in nested_synonyms.items():
    for canonical, synonyms in entries.items():
        if isinstance(synonyms, dict):
            for subcanonical, sublist in synonyms.items():
                for synonym in sublist:
                    synonyms_flat[synonym.lower().strip()] = subcanonical
        else:
            for synonym in synonyms:
                synonyms_flat[synonym.lower().strip()] = canonical


def normalize_test_name(name):
    if not name or not isinstance(name, str):
        return {
            "originalName": name,
            "canonicalName": name,
            "category": None,
            "normalized": False
        }
    key = name.lower().strip()
    canonical = synonyms_flat.get(key)
    category = categories_map.get(canonical) if canonical else None
    return {
        "originalName": name,
        "canonicalName": canonical if canonical else name,
        "category": category if category else None,
        "normalized": bool(canonical)
    }




def extract_text_from_pdf(pdf_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(pdf_path, output_folder=temp_dir, fmt='png')
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
    return text


def extract_text_from_image(image_path):
    img = Image.open(image_path)
    try:
        img = ImageOps.exif_transpose(img)   # respect iPhone EXIF orientation
    except Exception:
        pass
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")             # ensure tesseract-friendly mode
    return pytesseract.image_to_string(img)



def extract_json_content(content):
    try:
        content = re.sub(r'```json|```', '', content).strip()
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _post_openai_with_retry(payload):
    url = "https://api.openai.com/v1/chat/completions"
    for attempt in range(OPENAI_RETRIES):
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        if resp.status_code == 429 and attempt < OPENAI_RETRIES - 1:
            wait = (attempt + 1) * (RETRY_BASE_MS / 1000.0)
            print(f"⏳ OpenAI 429; retrying in {wait:.1f}s (attempt {attempt+1}/{OPENAI_RETRIES})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    return resp  # will have raised already if not OK


def analyze_with_openai(text):
    try:
        prompt = (
            "Analyze the following medical report text and extract details into structured JSON. "
            "Ensure the response contains these categories: 'Patient Information', 'Medical Parameters', and 'Doctor's Notes'. "  # ASCII apostrophe to match your keys
            "Do NOT omit any category, even if some data is missing. "
            "Each parameter in 'Medical Parameters' must be structured as an object with fields: 'Value', 'Reference Range', and 'Unit'. "
            "If the reference range is not provided, return 'Reference Range': 'N/A'. "
            "If the unit is not specified, return 'Unit': 'N/A'. "
            "Ensure numerical values are extracted accurately without extra text. "
            "If there are no doctor's notes, return 'Doctor's Notes': []. "
            "Respond in JSON format ONLY."
        )

        payload = {
            "model": OPENAI_MODEL,  # ← NEW (env-driven; default gpt-4o-mini)
            "messages": [
                {"role": "system", "content": "You are an AI assistant specializing in medical data extraction."},
                {"role": "user", "content": f"{prompt}\n\n{text}"},
            ],
            "temperature": 0,
            # JSON mode for structured output (supported by 4o/4o-mini):
            "response_format": {"type": "json_object"},  # ← NEW
            # no max_tokens → let the model respond fully
        }

        print(f"🧠 OpenAI model: {OPENAI_MODEL}")

        resp = _post_openai_with_retry(payload)
        data = resp.json()

        # --- OpenAI token usage logging ---
        usage = data.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        print("📊 OpenAI token usage:", flush=True)
        print(f"   Input tokens:  {prompt_tokens:,}", flush=True)
        print(f"   Output tokens: {completion_tokens:,}", flush=True)
        print(f"   Total tokens:  {total_tokens:,}", flush=True)

        content = (
            data.get("choices", [])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        
        if not content:
            raise ValueError("Empty content from OpenAI.")
        return extract_json_content(content)
    except Exception as e:
        print(f"Error in analyze_with_openai: {e}")
        return None


def parse_float(val):
    try:
        return float(str(val).replace(",", "").strip())
    except:
        return None


def flatten_nested_parameters(data):
    flat, unmatched = [], []
    for name, details in data.items():
        if isinstance(details, dict):
            val = parse_float(details.get("Value"))
            unit = details.get("Unit", "N/A")
            ref = details.get("Reference Range", "N/A")
            entry = normalize_test_name(name)
            flat.append({"name": name, "value": val, "unit": unit, "referenceRange": ref, **entry})
            if not entry["normalized"]:
                unmatched.append(name)
    return flat, unmatched


def flatten_array_parameters(data):
    flat, unmatched = [], []
    for item in data:
        name = item.get("Test Name") or item.get("Name") or item.get("Parameter")
        val = parse_float(item.get("Value"))
        ref = item.get("Reference Range", "N/A")
        unit = item.get("Unit", "N/A")
        entry = normalize_test_name(name)
        flat.append({"name": name, "value": val, "unit": unit, "referenceRange": ref, **entry})
        if not entry["normalized"]:
            unmatched.append(name)
    return flat, unmatched


def validate_response(resp):
    for key in REQUIRED_CATEGORIES:
        if key not in resp:
            resp[key] = [] if key == "Doctor's Notes" else {}
    flat = []
    if isinstance(resp.get("Medical Parameters"), list):
        flat, _ = flatten_array_parameters(resp["Medical Parameters"])
        grouped = {}
        for p in flat:
            grouped.setdefault(p["category"] or "Unmatched", {})[p["name"]] = {"Value": p["value"], "Unit": p["unit"], "Reference Range": p["referenceRange"]}
        resp["Medical Parameters"] = grouped
    else:
        flat, _ = flatten_nested_parameters(resp.get("Medical Parameters", {}))
    return resp, flat


def analyze_pdf(path, uid, name, report_date):
    resp = analyze_with_openai(extract_text_from_pdf(path)) or {}
    validated, flat = validate_response(resp)
    return {"parameters": flat, "extractedParameters": validated}


def analyze_image(path, uid, name, report_date):
    resp = analyze_with_openai(extract_text_from_image(path)) or {}
    validated, flat = validate_response(resp)
    return {"parameters": flat, "extractedParameters": validated}


def analyze_file(path, uid, name, report_date):
    ext = (os.path.splitext(path)[1] or "").lower()
    print(f"🔍 analyze_file: ext={ext}, path={path}", flush=True)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}

    # 1) Extension-based
    if ext in IMAGE_EXTS:
        return analyze_image(path, uid, name, report_date)
    if ext == ".pdf":
        return analyze_pdf(path, uid, name, report_date)

    # 2) Magic sniff for PDF
    try:
        with open(path, "rb") as f:
            if f.read(5).startswith(b"%PDF-"):
                print("📄 Detected PDF by magic bytes", flush=True)
                return analyze_pdf(path, uid, name, report_date)
    except Exception as e:
        print(f"⚠️ Magic-sniff warning: {e}", flush=True)

    # 3) Last-resort: let Pillow identify images
    try:
        from PIL import Image
        with Image.open(path) as im:
            fmt = (im.format or "").upper()
        print(f"🖼️ Pillow identified format: {fmt}", flush=True)
        return analyze_image(path, uid, name, report_date)
    except Exception as e:
        print(f"⚠️ Pillow identify failed: {e}", flush=True)

    raise ValueError(f"Unsupported image format/type: ext={ext}")


# --- Main runner
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python script.py <file_path> <userId> [fileName] [reportDate]")
        exit(1)
    path = sys.argv[1]
    uid = sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(path)
    report_date = sys.argv[4] if len(sys.argv) > 4 else None

    result = analyze_file(path, uid, name, report_date)
    print(json.dumps(result or {"parameters": [], "extractedParameters": {}}, default=str))

