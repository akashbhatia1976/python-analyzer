
import os
import json
import re
import requests
import tempfile
from pdf2image import convert_from_path
import pytesseract
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load synonyms and category mappings
with open(os.path.join(os.path.dirname(__file__), "data", "synonyms.json"), "r") as f:
    nested_synonyms = json.load(f)

# Flatten synonyms
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

# Load category mapping
with open(os.path.join(os.path.dirname(__file__), "data", "categories_map.json"), "r") as f:
    categories_map = json.load(f)

# Normalize name
def normalize_test_name(name):
    key = name.lower().strip()
    canonical = synonyms_flat.get(key)
    category = categories_map.get(canonical) if canonical else None
    return {
        "originalName": name,
        "canonicalName": canonical if canonical else name,
        "category": category if category else None,
        "normalized": bool(canonical)
    }

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}

REQUIRED_CATEGORIES = ["Patient Information", "Medical Parameters", "Doctor's Notes"]

def log(message):
    print(message)

def extract_json_content(content):
    try:
        content = re.sub(r'```json|```', '', content).strip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        log(f"Error decoding JSON: {e}")
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
            "model": "gpt-3.5-turbo-0125",
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

        log("Raw OpenAI Response:")
        log(json.dumps(raw_json, indent=4))

        content = raw_json.get("choices", [])[0].get("message", {}).get("content", "").strip()
        if not content:
            raise ValueError("Empty content in OpenAI response.")

        return extract_json_content(content)
    except requests.exceptions.RequestException as e:
        log(f"Request error with OpenAI: {e}")
        return None
    except Exception as e:
        log(f"Error analyzing with OpenAI: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(pdf_path, output_folder=temp_dir, fmt='png')
            extracted_text = ""
            for image in images:
                extracted_text += pytesseract.image_to_string(image)

        log("Extracted text from PDF:")
        log(extracted_text[:500])
        return extracted_text
    except Exception as e:
        log(f"Error extracting text from PDF: {e}")
        return None

def parse_float(val):
    try:
        return float(val.replace(",", "").strip())
    except (ValueError, TypeError, AttributeError):
        return None

def flatten_parameters(data):
    flat = []
    unmatched = []

    if not isinstance(data, dict):
        return flat

    for category, params in data.items():
        if isinstance(params, dict):
            for name, detail in params.items():
                if isinstance(detail, dict):
                    flat.append({
                        "category": category,
                        "name": name,
                        "value": parse_float(detail.get("Value")),
                        "unit": detail.get("Unit", "N/A"),
                        "referenceRange": detail.get("Reference Range", "N/A")
                    })
                    norm = normalize_test_name(name)
                    if not norm["normalized"]:
                        print(f"⚠️ Unmatched parameter: {name}")
                        unmatched.append(name)
                    flat[-1].update({
                        "originalName": norm["originalName"],
                        "canonicalName": norm["canonicalName"],
                        "ontologyCategory": norm["category"],
                        "normalized": norm["normalized"]
                    })

    print(f"✅ Flattened {len(flat)} parameters. Normalized {sum(1 for p in flat if p.get('normalized'))}, Unmatched: {sum(1 for p in flat if not p.get('normalized'))}")
    if unmatched:
        with open("unmatched_parameters.log", "w") as f:
            for name in unmatched:
                f.write(f"{name}\n")
        print("🚨 Unmatched parameters saved to unmatched_parameters.log")
    return flat

def validate_and_fix_response(parsed_response):
    if not parsed_response:
        return {
            "success": False,
            "categories": [],
            "extractedparameters": {},
            "parameters": [],
            "message": "No data extracted"
        }

    for category in REQUIRED_CATEGORIES:
        if category not in parsed_response:
            parsed_response[category] = {} if category != "Doctor's Notes" else []

    if isinstance(parsed_response.get("Doctor's Notes"), dict):
        parsed_response["Doctor's Notes"] = []

    print("Parsed Response:", json.dumps(parsed_response, indent=4))

    medical_params = parsed_response.get("Medical Parameters", {})
    flat_params = flatten_parameters(medical_params)

    return {
        "success": True,
        "categories": list(parsed_response.keys()),
        "extractedparameters": parsed_response,
        "parameters": flat_params,
        "message": "Data extraction completed"
    }

def analyze_pdf(file_path):
    try:
        log("Extracting text from PDF...")
        input_text = extract_text_from_pdf(file_path)
        if not input_text:
            raise ValueError("No text extracted from input file.")

        log("Analyzing text with OpenAI...")
        openai_response = analyze_with_openai(input_text)
        if not openai_response:
            raise ValueError("Failed to analyze text with OpenAI.")

        standardized_output = validate_and_fix_response(openai_response)
        return standardized_output
    except Exception as e:
        log(f"Unhandled error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# ✅ MongoDB Insertion
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["medicalReportsDB"]

def clean_and_parse_range(range_str):
    try:
        parts = range_str.replace(",", "").split("-")
        if len(parts) == 2:
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return low, high
    except Exception:
        pass
    return None, None

def save_to_mongo(user_id, report_name, standardized_data):
    
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)


    abnormal_count = 0
    for p in standardized_data["parameters"]:
        val = p.get("value")
        ref_range = p.get("referenceRange", "")
        if val is not None and "-" in ref_range:
            low, high = clean_and_parse_range(ref_range)
            if low is not None and (val < low or val > high):
                abnormal_count += 1

    report_data = {
        "userId": user_id,
        "reportName": report_name,
        "timestamp": now,
        "extractedParameters": standardized_data["extractedparameters"],
        "abnormalCount": abnormal_count
    }

    report_result = db.reports.insert_one(report_data)
    report_id = report_result.inserted_id

    parameters = []
    graph_edges = []

    for p in standardized_data["parameters"]:
        p["reportId"] = report_id
        p["userId"] = user_id
        p["healthId"] = f"AETHER-{user_id.upper()}"
        if not p.get("category"):
            p["category"] = "Unmatched"
        parameters.append(p)

        if p.get("loincCode"):
            graph_edges.append({
                "source": f"report:{str(report_id)}",
                "target": f"loinc:{p['loincCode']}",
                "type": "parameter-maps-to",
                "parameter": p.get("canonicalName"),
                "timestamp": now
            })

    if parameters:
        db.parameters.insert_many(parameters)
    if graph_edges:
        db.graph_edges.insert_many(graph_edges)

    print(f"✅ Inserted report '{report_name}' with {len(parameters)} parameters and {len(graph_edges)} graph edges.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("❌ Usage: python openai_extract_fields_combined_complete.py <pdf_path> [userId] [reportName]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    user_id = sys.argv[2] if len(sys.argv) > 2 else "sky001"
    report_name = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(pdf_path)

    result = analyze_pdf(pdf_path)

    if result.get("success"):
        save_to_mongo(user_id, report_name, result)
        print("✅ PDF analysis and MongoDB insert completed successfully.")
    else:
        print(f"❌ Failed: {result.get('error')}")
