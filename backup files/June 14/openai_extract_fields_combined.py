import os
import json
import re
import requests
import tempfile
from datetime import datetime, date
from pdf2image import convert_from_path
import pytesseract
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
        extracted_text = ""
        for image in images:
            extracted_text += pytesseract.image_to_string(image)
    return extracted_text

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
            "If the reference range is not provided in the report, return 'Reference Range': 'N/A'. "
            "If the unit is not specified, return 'Unit': 'N/A'. "
            "Ensure numerical values are extracted accurately without additional text. "
            "If there are no doctor’s notes, return an empty list: 'Doctor’s Notes': []. "
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

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json=data)
        response.raise_for_status()
        raw_json = response.json()

        content = raw_json.get("choices", [])[0].get("message", {}).get("content", "").strip()
        if not content:
            raise ValueError("Empty content in OpenAI response.")

        return extract_json_content(content)

    except requests.exceptions.RequestException as e:
        print(f"❌ Request error with OpenAI: {e}")
        return None
    except Exception as e:
        print(f"❌ Error analyzing with OpenAI: {e}")
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
            ref = details.get("Reference Range", "N/A")
            unit = details.get("Unit", "N/A")
            entry = normalize_test_name(name)
            flat.append({
                "name": name,
                "value": val,
                "unit": unit,
                "referenceRange": ref,
                **entry
            })
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
        flat.append({
            "name": name,
            "value": val,
            "unit": unit,
            "referenceRange": ref,
            **entry
        })
        if not entry["normalized"]:
            unmatched.append(name)
    return flat, unmatched

def validate_response(resp):
    for key in REQUIRED_CATEGORIES:
        if key not in resp:
            resp[key] = {} if key != "Doctor's Notes" else []
    if isinstance(resp["Medical Parameters"], list):
        flat, unmatched = flatten_array_parameters(resp["Medical Parameters"])
        grouped = {}
        for param in flat:
            grouped.setdefault(param["category"] or "Unmatched", {})[param["name"]] = {
                "Value": param["value"],
                "Unit": param["unit"],
                "Reference Range": param["referenceRange"]
            }
        resp["Medical Parameters"] = grouped
    else:
        flat, unmatched = flatten_nested_parameters(resp["Medical Parameters"])
    return resp, flat

# --- Main runner
def analyze_pdf(path, user_id, report_name=None, report_date=None):
    text = extract_text_from_pdf(path)
    ai_resp = analyze_with_openai(text)
    if not ai_resp:
        print("❌ Failed to get OpenAI response.")
        return
    full, flat = validate_response(ai_resp)

    return {
        "parameters": flat,
        "extractedParameters": full.get("Medical Parameters", {})
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python script.py <pdf_path> <userId> [fileName] [reportDate]")
        exit(1)

    pdf = sys.argv[1]
    uid = sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(pdf)
    report_date = sys.argv[4] if len(sys.argv) > 4 else None

    result = analyze_pdf(pdf, uid, name, report_date)

    if result:
        print(json.dumps(result, default=str))
    else:
        print(json.dumps({ "parameters": [], "extractedParameters": {} }))

