"""
Aether completeness audit for medical report extraction.

Purpose
-------
This module does NOT perform the primary extraction.

It receives:
1. the full OCR text for a medical report, and
2. the primary extraction already produced by Aether,

then asks the audit model to return ONLY patient-specific clinical observations
that are clearly present in the source text but missing from the primary
extraction.

The audit is intentionally additive:
- it does not modify existing extracted observations;
- it does not delete observations;
- it does not reinterpret or "correct" existing values;
- it returns missing-row candidates for deterministic downstream merging.

This module is designed to be imported by openai_extract_fields_combined.py
after the primary extraction has succeeded.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Keep the audit model independently configurable, but default to the same
# low-cost model currently used by the primary Aether extractor.
OPENAI_AUDIT_MODEL = os.getenv(
    "OPENAI_AUDIT_MODEL",
    os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
)

OPENAI_AUDIT_RETRIES = int(
    os.getenv("OPENAI_AUDIT_RETRIES", os.getenv("OPENAI_RETRIES", "3"))
)

OPENAI_AUDIT_RETRY_BASE_MS = int(
    os.getenv(
        "OPENAI_AUDIT_RETRY_BASE_MS",
        os.getenv("OPENAI_RETRY_BASE_MS", "1500"),
    )
)

OPENAI_AUDIT_TIMEOUT_SEC = int(
    os.getenv("OPENAI_AUDIT_TIMEOUT_SEC", "120")
)

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}" if OPENAI_API_KEY else "",
    "Content-Type": "application/json",
}

AUDIT_RESULT_KEY = "Missing Medical Parameters"
AUDIT_VERSION = "completeness_audit_v2_1"

SOURCE_EVIDENCE_KEYS = (
    "Source Evidence",
    "sourceEvidence",
    "source_evidence",
)

NAME_KEYS = ("Name", "name", "Test Name", "testName", "Parameter", "parameter")
VALUE_KEYS = ("Value", "value")
UNIT_KEYS = ("Unit", "unit")
REFERENCE_RANGE_KEYS = (
    "Reference Range",
    "referenceRange",
    "reference_range",
    "ReferenceRange",
)


class CompletenessAuditError(RuntimeError):
    """Raised when the completeness-audit request or response is invalid."""


def _first_present(mapping: Dict[str, Any], keys: Tuple[str, ...], default=None):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _extract_json_content(content: str) -> Dict[str, Any] | None:
    if not isinstance(content, str):
        return None

    cleaned = re.sub(r"```json|```", "", content).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _unwrap_primary_extraction(primary_extraction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either:
      - extractedParameters itself, or
      - the complete analyzer result containing extractedParameters.
    """

    if not isinstance(primary_extraction, dict):
        return {}

    extracted = primary_extraction.get("extractedParameters")

    if isinstance(extracted, dict):
        return extracted

    return primary_extraction


def _primary_medical_parameters(primary_extraction: Dict[str, Any]) -> Any:
    extracted = _unwrap_primary_extraction(primary_extraction)

    medical = extracted.get("Medical Parameters")

    if medical is None:
        return {}

    return medical


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return " ".join(value.strip().casefold().split())

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return str(value).strip().casefold()


def _collect_primary_observations(
    node: Any,
    path: Tuple[str, ...] = (),
) -> List[Dict[str, Any]]:
    """
    Walk the raw primary Medical Parameters structure and collect terminal
    patient observations.

    This is NOT a replacement for parameter_structure_router.py.
    It is used only to prevent the audit stage from returning obvious
    duplicates of observations that the primary extraction already contains.
    """

    observations: List[Dict[str, Any]] = []

    if isinstance(node, list):
        for item in node:
            if not isinstance(item, dict):
                continue

            name = _first_present(item, NAME_KEYS)
            has_value = any(key in item for key in VALUE_KEYS)

            if name and has_value:
                observations.append(
                    {
                        "Name": name,
                        "Value": _first_present(item, VALUE_KEYS),
                        "Reference Range": _first_present(
                            item,
                            REFERENCE_RANGE_KEYS,
                            "N/A",
                        ),
                        "Unit": _first_present(item, UNIT_KEYS, "N/A"),
                        "_path": path,
                    }
                )
                continue

            observations.extend(_collect_primary_observations(item, path))

        return observations

    if not isinstance(node, dict):
        return observations

    # A terminal parameter leaf in the existing V2.2 extraction.
    if any(key in node for key in VALUE_KEYS):
        inferred_name = path[-1] if path else None

        if inferred_name:
            observations.append(
                {
                    "Name": inferred_name,
                    "Value": _first_present(node, VALUE_KEYS),
                    "Reference Range": _first_present(
                        node,
                        REFERENCE_RANGE_KEYS,
                        "N/A",
                    ),
                    "Unit": _first_present(node, UNIT_KEYS, "N/A"),
                    "_path": path[:-1],
                }
            )

        return observations

    for name, child in node.items():
        observations.extend(
            _collect_primary_observations(
                child,
                (*path, str(name)),
            )
        )

    return observations



def _primary_observation_list(
    primary_extraction: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Produce a compact, deduplicated Name + Value + Unit list for the LLM audit.

    The primary extraction can be deeply nested. Giving the auditor that nested
    structure made comparison unnecessarily difficult and expensive. For
    completeness checking, the important question is simply whether a
    patient-specific observation is already represented by Name + Value.
    """

    medical = _primary_medical_parameters(primary_extraction)
    observations = _collect_primary_observations(medical)

    compact: List[Dict[str, Any]] = []
    seen = set()

    for row in observations:
        name = row.get("Name")

        if not name:
            continue

        signature = (
            _normalize_text(name),
            _normalize_text(row.get("Value")),
            _normalize_text(row.get("Unit")),
        )

        if signature in seen:
            continue

        seen.add(signature)

        compact.append(
            {
                "Name": name,
                "Value": row.get("Value"),
                "Unit": row.get("Unit", "N/A"),
            }
        )

    return compact


def _evidence_is_supported(
    report_text: str,
    evidence: str,
    name: Any,
    value: Any,
) -> bool:
    """
    Require the model to provide a short exact source excerpt for every
    candidate. We normalize only whitespace/case before comparison.

    This prevents invented rows from reaching the additive merge stage.
    """

    if not isinstance(evidence, str) or not evidence.strip():
        return False

    normalized_source = _normalize_text(report_text)
    normalized_evidence = _normalize_text(evidence)

    if not normalized_evidence:
        return False

    if normalized_evidence not in normalized_source:
        return False

    normalized_name = _normalize_text(name)
    normalized_value = _normalize_text(value)

    if normalized_name and normalized_name not in normalized_evidence:
        return False

    if normalized_value and normalized_value not in normalized_evidence:
        return False

    return True



def _is_na_like(value: Any) -> bool:
    normalized = _normalize_text(value)

    return normalized in {
        "",
        "n/a",
        "na",
        "not available",
        "not provided",
        "none",
        "null",
    }


def _strip_unit_from_value(value: Any, unit: Any) -> str:
    """
    Normalize a value that may have swallowed its unit.

    Example:
        value="54 (ml/min/1.73sqm)", unit="ml/min/1.73sqm"
        -> "54"

    This is deliberately narrow. It removes the known unit only when the unit
    appears as a trailing suffix, optionally wrapped in parentheses.
    """

    normalized_value = _normalize_text(value)
    normalized_unit = _normalize_text(unit)

    if not normalized_value or _is_na_like(unit):
        return normalized_value

    unit_pattern = re.escape(normalized_unit)

    stripped = re.sub(
        rf"\s*\(\s*{unit_pattern}\s*\)\s*$",
        "",
        normalized_value,
        flags=re.IGNORECASE,
    ).strip()

    if stripped != normalized_value:
        return stripped

    stripped = re.sub(
        rf"\s+{unit_pattern}\s*$",
        "",
        normalized_value,
        flags=re.IGNORECASE,
    ).strip()

    return stripped


def _observations_equivalent(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> bool:
    """
    Determine whether two rows represent the same already-extracted
    observation.

    This catches formatting duplicates where one pass has placed the unit
    inside Value while the other has kept Value and Unit separate.

    Example:
        primary: Value="54", Unit="ml/min/1.73sqm"
        audit:   Value="54 (ml/min/1.73sqm)", Unit="N/A"
    """

    left_name = _normalize_text(left.get("Name"))
    right_name = _normalize_text(right.get("Name"))

    if not left_name or left_name != right_name:
        return False

    left_value = _normalize_text(left.get("Value"))
    right_value = _normalize_text(right.get("Value"))

    if left_value == right_value:
        return True

    left_unit = left.get("Unit")
    right_unit = right.get("Unit")

    left_forms = {
        left_value,
        _strip_unit_from_value(left.get("Value"), left_unit),
        _strip_unit_from_value(left.get("Value"), right_unit),
    }

    right_forms = {
        right_value,
        _strip_unit_from_value(right.get("Value"), left_unit),
        _strip_unit_from_value(right.get("Value"), right_unit),
    }

    left_forms.discard("")
    right_forms.discard("")

    return bool(left_forms & right_forms)


def _candidate_signature(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    Conservative duplicate signature.

    Name + patient value is deliberately used instead of name alone:
    repeated tests with different values must remain eligible for recovery.
    """

    return (
        _normalize_text(row.get("Name")),
        _normalize_text(row.get("Value")),
    )


def _existing_primary_observations(
    primary_extraction: Dict[str, Any],
) -> List[Dict[str, Any]]:
    medical = _primary_medical_parameters(primary_extraction)

    return [
        row
        for row in _collect_primary_observations(medical)
        if row.get("Name")
    ]


def _validate_candidate(
    item: Any,
    report_text: str,
) -> Dict[str, Any] | None:
    """
    Enforce the narrow audit-row contract before anything is allowed to leave
    this module.

    A recovery candidate must:
    - have an explicit patient result;
    - never use N/A as a substitute patient result;
    - include exact source evidence copied from the OCR text;
    - have both its Name and Value represented in that source evidence.
    """

    if not isinstance(item, dict):
        return None

    name = _first_present(item, NAME_KEYS)

    if not isinstance(name, str) or not name.strip():
        return None

    if not any(key in item for key in VALUE_KEYS):
        return None

    value = _first_present(item, VALUE_KEYS)

    if isinstance(value, (dict, list, tuple, set)):
        return None

    if value is None:
        return None

    if isinstance(value, str):
        normalized_value = _normalize_text(value)

        if normalized_value in {
            "",
            "n/a",
            "na",
            "not available",
            "not provided",
            "none",
            "null",
        }:
            return None

    evidence = _first_present(item, SOURCE_EVIDENCE_KEYS)

    if not _evidence_is_supported(
        report_text,
        evidence,
        name,
        value,
    ):
        return None

    reference_range = _first_present(
        item,
        REFERENCE_RANGE_KEYS,
        "N/A",
    )

    unit = _first_present(
        item,
        UNIT_KEYS,
        "N/A",
    )

    if isinstance(reference_range, (dict, list, tuple, set)):
        return None

    if isinstance(unit, (dict, list, tuple, set)):
        return None

    return {
        "Name": name.strip(),
        "Value": value,
        "Reference Range": (
            "N/A" if reference_range is None else reference_range
        ),
        "Unit": "N/A" if unit is None else unit,
    }


def _filter_existing_candidates(
    primary_extraction: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Remove candidates already represented in the primary extraction.

    v2.1 handles exact duplicates and the common formatting case where a unit
    is embedded inside Value by one pass but stored separately by the other.
    """

    existing = _existing_primary_observations(primary_extraction)
    filtered: List[Dict[str, Any]] = []
    skipped = 0

    for candidate in candidates:
        if any(
            _observations_equivalent(candidate, primary)
            for primary in existing
        ):
            skipped += 1
            continue

        if any(
            _observations_equivalent(candidate, already_new)
            for already_new in filtered
        ):
            skipped += 1
            continue

        filtered.append(candidate)

    return filtered, skipped


def _post_openai_with_retry(payload: Dict[str, Any]) -> requests.Response:
    if not OPENAI_API_KEY:
        raise CompletenessAuditError("OPENAI_API_KEY is not set")

    last_response = None

    for attempt in range(OPENAI_AUDIT_RETRIES):
        response = requests.post(
            OPENAI_CHAT_COMPLETIONS_URL,
            headers=HEADERS,
            json=payload,
            timeout=OPENAI_AUDIT_TIMEOUT_SEC,
        )

        last_response = response

        retryable = response.status_code in {
            429,
            500,
            502,
            503,
            504,
        }

        if retryable and attempt < OPENAI_AUDIT_RETRIES - 1:
            wait = (
                (attempt + 1)
                * (OPENAI_AUDIT_RETRY_BASE_MS / 1000.0)
            )

            print(
                f"⏳ Completeness audit HTTP {response.status_code}; "
                f"retrying in {wait:.1f}s "
                f"(attempt {attempt + 1}/{OPENAI_AUDIT_RETRIES})",
                flush=True,
            )

            time.sleep(wait)
            continue

        if not response.ok:
            print(
                f"❌ Completeness audit HTTP status: "
                f"{response.status_code}",
                flush=True,
            )
            print(
                f"❌ Completeness audit error body: "
                f"{response.text}",
                flush=True,
            )

        response.raise_for_status()
        return response

    if last_response is None:
        raise CompletenessAuditError(
            "Completeness audit failed before receiving a response."
        )

    last_response.raise_for_status()
    return last_response


def _build_audit_prompt(
    report_text: str,
    primary_observations: List[Dict[str, Any]],
) -> str:
    """
    Build a comparison prompt using a flat list of already-extracted
    observations rather than the full nested primary JSON.
    """

    primary_json = json.dumps(
        primary_observations,
        ensure_ascii=False,
        default=str,
    )

    return (
        "You are a medical-report extraction COMPLETENESS AUDITOR. "
        "You are NOT performing the primary extraction again. "

        "You will receive two inputs: "
        "(1) the complete OCR text of the medical report, and "
        "(2) a FLAT LIST of patient observations already extracted by the "
        "primary pass, represented as Name + Value + Unit rows. "

        "Your ONLY task is to identify patient-specific clinical result rows "
        "that are clearly present in the source OCR but are NOT represented in "
        "the primary observation list. "

        "Before returning any source row, compare its test/observation identity "
        "AND patient value/unit representation against the ENTIRE primary observation list. "
        "If the same clinical observation and patient value are already present, "
        "it is NOT missing, even if capitalization, punctuation, abbreviation, "
        "panel placement, grouping, or formatting differs. "

        "Return ONLY genuinely missing source rows. "
        "Do NOT modify, correct, reinterpret, replace, or delete any existing "
        "primary observation. "
        "Do NOT return a more complete reference range for an observation that "
        "already exists; reference-range repair is outside this audit. "

        "Inspect the ENTIRE report text from beginning to end. "
        "Patient-specific results can be numeric, text, qualitative, or symbolic. "
        "A literal '-', '+', '++', 'Trace', 'Absent', 'Negative', 'Positive', "
        "'Not seen', titre, or descriptive morphology result is a valid patient "
        "result when it appears in the report's result position. "

        "However, NEVER invent a patient value. "
        "If the source does not contain an explicit result for a candidate row, "
        "do not return it. "
        "In particular, NEVER use 'N/A' as the Value of a recovered observation. "
        "'N/A' may be used only for Reference Range or Unit when those fields are "
        "not provided in the source. "

        "Do not convert a section heading, panel name, or summary label into a "
        "new observation merely because a descriptive result appears elsewhere "
        "under that section. "
        "For example, a source row 'COMMENT Leucopenia' is a Comment observation "
        "with value 'Leucopenia'; do not invent a separate observation called "
        "'Leucopenia' with value 'N/A'. "

        "Pay particular attention to commonly omitted rows such as morphology, "
        "smear findings, microscopy, physical examination, chemical examination, "
        "qualitative observations, symbolic results, and associated calculated "
        "results such as estimated average glucose. "

        "Every recovered source result row must be returned as a separate item. "
        "Do not summarize or collapse multiple missing rows into one item. "

        "For EVERY candidate you return, also include 'Source Evidence'. "
        "'Source Evidence' must be a SHORT EXACT EXCERPT copied verbatim from the "
        "provided OCR text that contains BOTH the candidate Name and the candidate "
        "Value. Do not paraphrase the evidence. "
        "If you cannot provide exact source evidence containing both, do not "
        "return that candidate. "

        "For each missing observation return exactly these fields: "
        "'Name', 'Value', 'Reference Range', 'Unit', and 'Source Evidence'. "

        "'Name' must preserve the observation/test name as reported in the source. "
        "'Value' must preserve the patient's reported value. "
        "'Reference Range' must preserve the complete reference information for "
        "that missing row when present, otherwise 'N/A'. "
        "'Unit' must preserve the reported unit when present, otherwise 'N/A'. "

        "Do not treat methodologies, explanatory prose, generic medical "
        "information, signatures, historical narrative, or reference-range-only "
        "text as missing patient observations. "

        f"Return valid JSON with exactly one top-level key: "
        f"'{AUDIT_RESULT_KEY}'. "
        f"The value of '{AUDIT_RESULT_KEY}' MUST be an array. "
        "If nothing is missing, return an empty array. "
        "Do not include markdown or commentary outside the JSON. "

        "\n\n===== COMPLETE REPORT OCR TEXT =====\n"
        f"{report_text}"
        "\n\n===== PRIMARY EXTRACTED OBSERVATIONS (ALREADY PRESENT) =====\n"
        f"{primary_json}"
    )


def audit_missing_parameters(
    report_text: str,
    primary_extraction: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Find patient-specific clinical rows missing from the primary extraction.

    Returns:
        {
            "candidates": [...],
            "rawCandidates": [...],
            "metadata": {
                "status": "completed" | "failed",
                "model": "...",
                "rawCandidateCount": 0,
                "candidateCount": 0,
                "duplicatesSkipped": 0,
                "usage": {...},
                "reason": null | "..."
            }
        }

    This function never mutates primary_extraction.
    """

    if not isinstance(report_text, str) or not report_text.strip():
        return {
            "candidates": [],
            "rawCandidates": [],
            "metadata": {
                "status": "failed",
                "model": OPENAI_AUDIT_MODEL,
                "rawCandidateCount": 0,
                "candidateCount": 0,
                "duplicatesSkipped": 0,
                "usage": {},
                "reason": "EMPTY_REPORT_TEXT",
            },
        }

    if not isinstance(primary_extraction, dict):
        return {
            "candidates": [],
            "rawCandidates": [],
            "metadata": {
                "status": "failed",
                "model": OPENAI_AUDIT_MODEL,
                "rawCandidateCount": 0,
                "candidateCount": 0,
                "duplicatesSkipped": 0,
                "usage": {},
                "reason": "INVALID_PRIMARY_EXTRACTION",
            },
        }

    primary_observations = _primary_observation_list(primary_extraction)

    prompt = _build_audit_prompt(
        report_text,
        primary_observations,
    )

    payload: Dict[str, Any] = {
        "model": OPENAI_AUDIT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict medical data completeness auditor. "
                    "Recover only source-supported observations omitted from "
                    "an existing extraction."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        # Keep this aligned with the known-good primary extractor for the first
        # audit experiment. We validate the returned delta strictly in Python.
        "response_format": {"type": "json_object"},
    }

    # gpt-4o / gpt-4o-mini accept temperature=0. GPT-5.6 reasoning models
    # currently require their default temperature in this configuration.
    if not OPENAI_AUDIT_MODEL.startswith("gpt-5.6"):
        payload["temperature"] = 0

    print(
        f"🧾 Completeness audit model: {OPENAI_AUDIT_MODEL}",
        flush=True,
    )

    try:
        response = _post_openai_with_retry(payload)
        data = response.json()

        usage = data.get("usage", {}) or {}

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        parsed = _extract_json_content(content)

        if not parsed:
            raise CompletenessAuditError(
                "Completeness audit returned invalid JSON."
            )

        raw_candidates = parsed.get(AUDIT_RESULT_KEY, [])

        if not isinstance(raw_candidates, list):
            raise CompletenessAuditError(
                f"'{AUDIT_RESULT_KEY}' is not an array."
            )

        validated: List[Dict[str, Any]] = []

        for item in raw_candidates:
            row = _validate_candidate(item, report_text)

            if row is not None:
                validated.append(row)

        filtered, duplicates_skipped = _filter_existing_candidates(
            primary_extraction,
            validated,
        )

        print("📊 Completeness audit token usage:", flush=True)
        print(
            f"   Input tokens:  "
            f"{usage.get('prompt_tokens', 0):,}",
            flush=True,
        )
        print(
            f"   Output tokens: "
            f"{usage.get('completion_tokens', 0):,}",
            flush=True,
        )
        print(
            f"   Total tokens:  "
            f"{usage.get('total_tokens', 0):,}",
            flush=True,
        )

        print(
            "🩹 Completeness audit:",
            f"raw={len(raw_candidates)}",
            f"validated={len(validated)}",
            f"validationRejected={len(raw_candidates) - len(validated)}",
            f"duplicatesSkipped={duplicates_skipped}",
            f"candidates={len(filtered)}",
            flush=True,
        )

        return {
            "candidates": filtered,
            "rawCandidates": raw_candidates,
            "metadata": {
                "status": "completed",
                "auditVersion": AUDIT_VERSION,
                "model": OPENAI_AUDIT_MODEL,
                "primaryObservationCount": len(primary_observations),
                "rawCandidateCount": len(raw_candidates),
                "validatedCandidateCount": len(validated),
                "validationRejectedCount": len(raw_candidates) - len(validated),
                "candidateCount": len(filtered),
                "duplicatesSkipped": duplicates_skipped,
                "usage": usage,
                "reason": None,
            },
        }

    except Exception as exc:
        print(
            f"⚠️ Completeness audit failed safely: {exc}",
            flush=True,
        )

        return {
            "candidates": [],
            "rawCandidates": [],
            "metadata": {
                "status": "failed",
                "model": OPENAI_AUDIT_MODEL,
                "rawCandidateCount": 0,
                "candidateCount": 0,
                "duplicatesSkipped": 0,
                "usage": {},
                "reason": str(exc),
            },
        }


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)

    if not isinstance(parsed, dict):
        raise CompletenessAuditError(
            f"JSON file must contain an object: {path}"
        )

    return parsed


def main() -> int:
    """
    Standalone test mode.

    Example:
        python completeness_audit.py \
          --text-file tests/akash1_ocr.txt \
          --primary-json tests/router_benchmark_runs/.../result.json \
          --output tests/audit_result.json

    Production integration should import audit_missing_parameters() directly
    rather than launching this module as a subprocess.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Audit an existing Aether medical-report extraction for missing "
            "patient-specific clinical rows."
        )
    )

    parser.add_argument(
        "--text-file",
        required=True,
        help="Path to the complete OCR text used for the primary extraction.",
    )

    parser.add_argument(
        "--primary-json",
        required=True,
        help=(
            "Path to extractedParameters JSON or a complete analyzer result.json."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the audit JSON result.",
    )

    args = parser.parse_args()

    text_path = Path(args.text_file)
    primary_path = Path(args.primary_json)

    report_text = text_path.read_text(encoding="utf-8")
    primary_extraction = _load_json(primary_path)

    result = audit_missing_parameters(
        report_text,
        primary_extraction,
    )

    output_json = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json + "\n", encoding="utf-8")
        print(f"✅ Audit result written to: {output_path}", flush=True)
    else:
        print(output_json)

    return 0 if result.get("metadata", {}).get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
