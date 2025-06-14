import os
import json
import re
import requests
import tempfile
from datetime import datetime, date
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from dotenv import load_dotenv

# --- Load environment
load_dotenv()

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

# --- OpenAI Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}
REQUIRED_CATEGORIES = ["Patient Information", "Medical Parameters", "Doctor's Notes"]


def extract_text_from_pdf(pdf_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(pdf_path, output_folder=temp_dir, fmt='png')
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
    return text


def extract_text_from_image(image_path):
    """Extract text from an image file using OCR."""
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)


def extract_json_content(content):
    try:
        content = re.sub(r'```json|```', '', content).strip()
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def analyze_with_openai(text):
    try:
        prompt = (
            "Analyze the following medical report text and extract details into structured JSON. "
            "Ensure the response contains these categories: 'Patient Information', 'Medical Parameters', and 'Doctor’s Notes'. "
            "Do NOT omit any category, even if some data is missing. "
            "Each parameter in 'Medical Parameters' must be structured as an object with fields: 'Value', 'Reference Range', and 'Unit'. "
            "If the reference range is not provided, return 'Reference Range': 'N/A'. "
            "If the unit is not specified, return 'Unit': 'N/A'. "
            "Ensure numerical values are extracted accurately without extra text. "
            "If there are no doctor’s notes, return 'Doctor’s Notes': []. "
            "Respond in JSON format ONLY."
        )
        data = {
            "model": "gpt-4-turbo-2024-04-09",
            "messages": [
                {"role": "system", "content": "You are an AI assistant specializing in medical data extraction."},
                {"role": "user", "content": f"{prompt}\n\n{text}"},
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json=data)
        resp.raise_for_status()
        content = resp.json().get("choices", [])[0].get("message", {}).get("content", "").strip()
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
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
        return analyze_image(path, uid, name, report_date)
    return analyze_pdf(path, uid, name, report_date)

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

