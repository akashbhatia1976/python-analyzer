
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

def flatten_parameters(data):
    flat = []
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
    return flat

def parse_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

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
