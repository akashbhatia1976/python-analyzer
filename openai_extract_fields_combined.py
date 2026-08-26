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

from parameter_structure_router import route_parameters, ARRAY


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

# Completeness audit is opt-in and fails safe. If the flag is absent or false,
# the analyzer follows the exact primary-extraction path used before this change.
OPENAI_COMPLETENESS_AUDIT = (
    os.getenv("OPENAI_COMPLETENESS_AUDIT", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

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

        # Every canonical parameter must resolve to itself.
        synonyms_flat[canonical.lower().strip()] = canonical

        if isinstance(synonyms, dict):
            for subcanonical, sublist in synonyms.items():

                # Nested canonicals also resolve to themselves.
                synonyms_flat[subcanonical.lower().strip()] = subcanonical

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
        resp = requests.post(
            url,
            headers=HEADERS,
            json=payload,
            timeout=120
        )

        if resp.status_code == 429 and attempt < OPENAI_RETRIES - 1:
            wait = (attempt + 1) * (RETRY_BASE_MS / 1000.0)
            print(
                f"⏳ OpenAI 429; retrying in {wait:.1f}s "
                f"(attempt {attempt+1}/{OPENAI_RETRIES})"
            )
            time.sleep(wait)
            continue

        if not resp.ok:
            print(
                f"❌ OpenAI HTTP status: {resp.status_code}",
                flush=True
            )
            print(
                f"❌ OpenAI error body: {resp.text}",
                flush=True
            )

        resp.raise_for_status()
        return resp

    return resp  # will have raised already if not OK


def analyze_with_openai(text):
    try:
        prompt = (
            "Analyze the following medical report text and extract the patient's clinical data into structured JSON. "

            "The response MUST contain exactly these top-level categories: "
            "'Patient Information', 'Medical Parameters', and 'Doctor's Notes'. "
            "Do NOT omit any of these categories, even if some data is missing. "

            "COMPLETENESS IS CRITICAL. "
            "Inspect the entire report from beginning to end, including every page, section, panel, table, subsection, "
            "and patient-specific result. Extract EVERY patient-specific clinical observation or test result that is present. "
            "Do not stop after extracting a representative subset of a panel. "
            "If a panel contains multiple reported observations, extract every reported observation in that panel. "
            "Every row in a clinical results table that contains a patient-specific reported result MUST be extracted, "
            "including morphology, microscopy, smear findings, physical examination, chemical examination, "
            "qualitative observations, and rows whose reported result is only a symbol such as '-', '+', '++', or similar. "
            "A result of '-' is still a reported patient result and MUST NOT be omitted. "

            "Pay particular attention to morphology, smear, and microscopy subsections. "
            "If the report contains headings such as 'RBC Morphology', 'WBC Morphology', 'Platelet Morphology', "
            "'Microscopic Examination', or similar, inspect EVERY row beneath those headings and extract every "
            "patient-specific result before moving to the next section. "
            "This includes rows such as Hypochromia, Microcytosis, Macrocytosis, Anisocytosis, Poikilocytosis, "
            "Polychromasia, Target Cells, Basophilic Stippling, Normoblasts, Others, Platelet Morphology, Comment, "
            "and any equivalent observations present in the report. "
            "Do not omit a row merely because its result is '-', 'Absent', 'Normal', descriptive text, or another "
            "qualitative value. "

            "Do not omit qualitative, categorical, textual, or symbolic results. "
            "Values such as 'Negative', 'Positive', 'Trace', 'Absent', 'Not seen', titres such as '1:80', "
            "plus signs, morphology findings, microscopy findings, urine findings, and similar observations "
            "are valid clinical results and MUST be extracted when present. "

            "Preserve the report's clinical panel hierarchy where it exists. "
            "Panel or section names may be used as grouping objects, but every terminal clinical observation "
            "inside 'Medical Parameters' must be represented as an object with the fields "
            "'Value', 'Reference Range', and 'Unit'. "

            "If a reference range is not provided for a result, return 'Reference Range': 'N/A'. "
            "If a reference range or interpretive range contains multiple bands, thresholds, categories, or lines, "
            "preserve the COMPLETE reference information associated with that result. "
            "Never stop after or return only the first reference category, threshold, band, or line. "
            "For example, if a result includes separate normal, borderline, high, low, deficient, insufficient, "
            "prediabetic, diabetic, cardiovascular-risk, or similar interpretive bands, include ALL of the bands "
            "associated with that result in 'Reference Range'. "
            "If a unit is not provided for a result, return 'Unit': 'N/A'. "

            "Preserve the patient's reported result exactly as shown in the report. "
            "Never substitute the reference range, expected value, normal value, interpretation, or explanatory text for the patient's result. "
            "If the patient's reported result is '-', the 'Value' MUST be '-'. "
            "If the patient's reported result is 'Trace', 'Absent', 'Negative', '+', '++', a range such as '1-2', "
            "or any other qualitative or symbolic result, preserve that result exactly. "
            "Do not infer, calculate, correct, reinterpret, normalize, or replace unusual or abnormal-looking values. "
            "Do not change test names into standardized names; preserve the test name as reported. "

            "If the same test appears more than once in different panels or sections of the report, preserve each occurrence "
            "within its corresponding panel or section. Do not merge or deduplicate clinically separate occurrences. "

            "Do not treat panel headings, explanatory reference text, methodology, general medical information, "
            "or non-patient-specific prose as clinical parameters. "

            "For 'Patient Information', copy information only from fields explicitly identified in the report. "
            "Do not infer a person's role from signatures, credentials, report headers, or report footers. "
            "In particular, do not treat a pathologist, laboratory doctor, signatory, lab director, or report author "
            "as the Consulting Doctor unless that person is explicitly identified as the Consulting Doctor in the report. "
            "If the Consulting Doctor field is blank, '-', or not provided, preserve that rather than inferring a doctor. "

            "'Doctor's Notes' must appear only as the top-level 'Doctor's Notes' category and must never be placed "
            "inside 'Medical Parameters'. "
            "If there are no doctor's notes, return 'Doctor's Notes': []. "

            "Before returning the response, perform a final completeness check against the entire report text. "
            "Verify that every page, table, panel, subsection, and patient-specific reported result has been considered "
            "and that no clinical result has been skipped merely because other results from the same panel were already extracted. "

            "Respond in valid JSON format ONLY. Do not include markdown, commentary, explanation, or text outside the JSON."
        )

        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant specializing in medical data extraction."
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\n{text}"
                },
            ],
            "response_format": {"type": "json_object"},
        }

        # GPT-4o / GPT-4o-mini support deterministic temperature=0.
        # GPT-5.6 reasoning models only support their default temperature.
        if not OPENAI_MODEL.startswith("gpt-5.6"):
            payload["temperature"] = 0

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



def validate_response(resp):
    """Validate top-level extraction categories and route Medical Parameters.

    The raw OpenAI Medical Parameters structure is classified by the dedicated
    router. Each known structure has its own handler. Unknown/ambiguous shapes
    produce zero canonical parameters so the existing Node ingestion review path
    can safely retain the source + raw extraction instead of persisting guesses.
    """

    if not isinstance(resp, dict):
        resp = {}

    for key in REQUIRED_CATEGORIES:
        if key not in resp:
            resp[key] = [] if key == "Doctor's Notes" else {}

    medical_parameters = resp.get("Medical Parameters", {})
    routed = route_parameters(medical_parameters, normalize_test_name)

    flat = routed.get("parameters", [])
    metadata = routed.get("metadata", {})

    print(
        "🧭 Parameter router:",
        metadata.get("parameterStructure", "unknown"),
        f"| count={metadata.get('parameterCount', 0)}",
        f"| unmatched={metadata.get('unmatchedCount', 0)}",
        flush=True,
    )

    if metadata.get("status") != "completed":
        print(
            "⚠️ Parameter router sent extraction to safe review path:",
            metadata.get("reason"),
            flush=True,
        )

    # Preserve legacy array-output behavior: the old analyzer regrouped array
    # parameters into an object before returning extractedParameters. This is
    # intentionally retained to avoid changing downstream consumers in the P1.
    if metadata.get("parameterStructure") == ARRAY:
        grouped = {}
        for p in flat:
            grouped.setdefault(p.get("category") or "Unmatched", {})[p["name"]] = {
                "Value": p.get("value"),
                "Unit": p.get("unit", "N/A"),
                "Reference Range": p.get("referenceRange", "N/A"),
            }
        resp["Medical Parameters"] = grouped

    return resp, flat, metadata


def _build_analysis_result(resp):
    validated, flat, router_metadata = validate_response(resp or {})
    return {
        "parameters": flat,
        "extractedParameters": validated,
        "extractionMetadata": router_metadata,
    }


def _apply_completeness_audit(report_text, primary_result):
    """Run the validated V2.1 audit as a strictly additive second pass."""

    if not OPENAI_COMPLETENESS_AUDIT:
        return primary_result

    primary_metadata = primary_result.get("extractionMetadata", {}) or {}

    # Do not layer an audit result on top of an unsupported primary structure.
    if primary_metadata.get("status") != "completed":
        print(
            "🧾 Completeness audit skipped: primary router did not complete safely",
            flush=True,
        )
        return primary_result

    try:
        # Lazy import keeps the pre-audit path unchanged when the flag is off.
        from completeness_audit_v2_1 import audit_missing_parameters

        audit_result = audit_missing_parameters(report_text, primary_result)
        audit_metadata = audit_result.get("metadata", {}) or {}
        candidates = audit_result.get("candidates", []) or []

        # Preserve the primary extraction and expose the validated audit delta
        # separately for provenance/debugging.
        primary_result["completenessAudit"] = audit_metadata
        primary_result["auditRecovery"] = candidates

        if audit_metadata.get("status") != "completed" or not candidates:
            print("🩹 Completeness audit added 0 parameters", flush=True)
            return primary_result

        # Audit candidates use the existing ARRAY contract:
        # Name + Value + Reference Range + Unit.
        routed_audit = route_parameters(candidates, normalize_test_name)
        audit_router_metadata = routed_audit.get("metadata", {}) or {}

        if audit_router_metadata.get("status") != "completed":
            print(
                "⚠️ Completeness audit delta failed router safety check; "
                "adding 0 parameters",
                flush=True,
            )
            primary_result["completenessAudit"]["mergeStatus"] = (
                "audit_delta_router_failed"
            )
            primary_result["completenessAudit"]["mergeReason"] = (
                audit_router_metadata.get("reason")
            )
            return primary_result

        audit_parameters = routed_audit.get("parameters", []) or []

        if not audit_parameters:
            print("🩹 Completeness audit added 0 parameters", flush=True)
            return primary_result

        primary_parameters = primary_result.get("parameters", []) or []
        primary_count = len(primary_parameters)

        # Additive only: never edit or replace a primary parameter.
        primary_result["parameters"] = [*primary_parameters, *audit_parameters]

        final_metadata = dict(primary_metadata)
        final_metadata["primaryParameterCount"] = primary_count
        final_metadata["auditAddedCount"] = len(audit_parameters)
        final_metadata["parameterCount"] = len(primary_result["parameters"])
        final_metadata["unmatchedCount"] = (
            int(primary_metadata.get("unmatchedCount", 0) or 0)
            + int(audit_router_metadata.get("unmatchedCount", 0) or 0)
        )
        final_metadata["completenessAuditEnabled"] = True
        final_metadata["completenessAuditStatus"] = audit_metadata.get("status")
        primary_result["extractionMetadata"] = final_metadata

        primary_result["completenessAudit"]["mergeStatus"] = "completed"
        primary_result["completenessAudit"]["auditRouterVersion"] = (
            audit_router_metadata.get("routerVersion")
        )
        primary_result["completenessAudit"]["auditRouterStructure"] = (
            audit_router_metadata.get("parameterStructure")
        )

        print(
            f"🩹 Completeness audit added {len(audit_parameters)} parameter(s) "
            f"| final count={len(primary_result['parameters'])}",
            flush=True,
        )
        return primary_result

    except Exception as exc:
        # Audit is an enhancement, never a reason to lose a valid primary result.
        print(
            f"⚠️ Completeness audit integration failed safely: {exc}",
            flush=True,
        )
        primary_result["completenessAudit"] = {
            "status": "failed",
            "reason": str(exc),
        }
        primary_result["auditRecovery"] = []
        return primary_result


def _analyze_extracted_text(text):
    """Reuse one OCR/text pass for primary extraction and optional audit."""

    resp = analyze_with_openai(text) or {}
    primary_result = _build_analysis_result(resp)
    return _apply_completeness_audit(text, primary_result)


def analyze_pdf(path, uid, name, report_date):
    text = extract_text_from_pdf(path)
    return _analyze_extracted_text(text)


def analyze_image(path, uid, name, report_date):
    text = extract_text_from_image(path)
    return _analyze_extracted_text(text)


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

