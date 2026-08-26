"""Aether parameter structure handlers.

Each handler accepts exactly one known Medical Parameters shape and emits the
same canonical parameter row contract expected by reportIngestionService.js.

Handlers do not classify input structure. Classification belongs to
parameter_structure_router.py.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Tuple


Normalizer = Callable[[Any], Dict[str, Any]]

VALUE_KEYS = ("Value", "value")
UNIT_KEYS = ("Unit", "unit")
REFERENCE_RANGE_KEYS = (
    "Reference Range",
    "referenceRange",
    "reference_range",
    "ReferenceRange",
)
NAME_KEYS = ("Test Name", "Name", "Parameter", "testName", "name", "parameter")

KNOWN_METADATA_KEYS = frozenset({
    "doctor's notes",
    "doctors notes",
    "doctor notes",
    "patient information",
})


def is_known_metadata_key(name: Any) -> bool:
    """Recognize only explicitly approved non-parameter metadata keys.

    These keys may occasionally be misplaced by the extractor inside
    `Medical Parameters`. Do not make this list broad: unknown structures
    must continue to fail safe.
    """

    if not isinstance(name, str):
        return False

    return name.strip().lower() in KNOWN_METADATA_KEYS



# Deliberately strict. Values such as 1:80, <1.10, "Negative", "Not seen",
# and "No agglutination" must remain strings rather than being coerced.
_NUMERIC_RE = re.compile(
    r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?$"
)


class StructureHandlerError(ValueError):
    """Raised when a handler receives data outside its declared contract."""


def _first_present(mapping: Dict[str, Any], keys: Tuple[str, ...], default=None):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def has_value_field(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in VALUE_KEYS)


def is_parameter_leaf(value: Any) -> bool:
    """Return True only for the V1 parameter-leaf contract.

    A parameter leaf must be an object containing a Value field. Unit and
    Reference Range may be absent because the upstream prompt allows N/A.
    """

    return has_value_field(value)


def normalize_parameter_value(value: Any) -> Any:
    """Preserve clinical qualitative values while normalizing true numerics.

    - JSON ints/floats stay numeric.
    - Strict numeric strings become floats (matching legacy behavior).
    - All other strings survive exactly (trimmed).
    - None stays None.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        # bool is a subclass of int in Python; do not silently turn it into 1/0.
        return value

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        if _NUMERIC_RE.fullmatch(text):
            try:
                return float(text.replace(",", ""))
            except ValueError:
                pass

        return text

    # Unknown scalar/object value types are preserved rather than guessed.
    return value


def build_canonical_parameter(
    name: Any,
    details: Dict[str, Any],
    normalize_test_name: Normalizer,
) -> Dict[str, Any]:
    if not isinstance(details, dict) or not is_parameter_leaf(details):
        raise StructureHandlerError(
            f"Parameter '{name}' does not satisfy the parameter-leaf contract."
        )

    raw_value = _first_present(details, VALUE_KEYS)
    unit = _first_present(details, UNIT_KEYS, "N/A")
    reference_range = _first_present(details, REFERENCE_RANGE_KEYS, "N/A")

    normalized = normalize_test_name(name)

    return {
        "name": name,
        "value": normalize_parameter_value(raw_value),
        "unit": "N/A" if unit is None else unit,
        "referenceRange": "N/A" if reference_range is None else reference_range,
        **normalized,
    }


def _append_leaf(
    output: List[Dict[str, Any]],
    unmatched: List[Any],
    name: Any,
    details: Dict[str, Any],
    normalize_test_name: Normalizer,
) -> None:
    row = build_canonical_parameter(name, details, normalize_test_name)
    output.append(row)
    if not row.get("normalized"):
        unmatched.append(name)


def handle_flat_parameters(
    data: Dict[str, Any],
    normalize_test_name: Normalizer,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Handle a pure flat object: {parameter_name: parameter_leaf, ...}."""

    if not isinstance(data, dict) or not data:
        raise StructureHandlerError("Flat handler requires a non-empty object.")

    flat: List[Dict[str, Any]] = []
    unmatched: List[Any] = []

    for name, details in data.items():
        if not is_parameter_leaf(details):
            raise StructureHandlerError(
                f"Flat handler received non-leaf branch at '{name}'."
            )
        _append_leaf(flat, unmatched, name, details, normalize_test_name)

    return flat, unmatched


def handle_array_parameters(
    data: List[Any],
    normalize_test_name: Normalizer,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Handle the known list shape with Test Name/Name/Parameter + Value."""

    if not isinstance(data, list) or not data:
        raise StructureHandlerError("Array handler requires a non-empty list.")

    flat: List[Dict[str, Any]] = []
    unmatched: List[Any] = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise StructureHandlerError(
                f"Array item {index} is not an object."
            )

        name = _first_present(item, NAME_KEYS)
        if not name:
            raise StructureHandlerError(
                f"Array item {index} has no recognized parameter-name field."
            )

        if not is_parameter_leaf(item):
            raise StructureHandlerError(
                f"Array item {index} has no recognized Value field."
            )

        _append_leaf(flat, unmatched, name, item, normalize_test_name)

    return flat, unmatched


def _walk_nested_container(
    data: Dict[str, Any],
    normalize_test_name: Normalizer,
    flat: List[Dict[str, Any]],
    unmatched: List[Any],
    path: Tuple[str, ...],
) -> None:
    if not isinstance(data, dict) or not data:
        joined = " > ".join(path) or "<root>"
        raise StructureHandlerError(f"Empty/invalid nested container at {joined}.")

    for name, value in data.items():
        current_path = (*path, str(name))

        if is_parameter_leaf(value):
            _append_leaf(flat, unmatched, name, value, normalize_test_name)
            continue

        if isinstance(value, dict) and value:
            _walk_nested_container(
                value,
                normalize_test_name,
                flat,
                unmatched,
                current_path,
            )
            continue

        joined = " > ".join(current_path)
        raise StructureHandlerError(
            f"Unsupported nested branch at {joined}: {type(value).__name__}."
        )


def handle_nested_parameters(
    data: Dict[str, Any],
    normalize_test_name: Normalizer,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Handle a pure nested object whose top-level branches are containers.

    This handler may recurse internally because the router has already declared
    the whole object to be NESTED_OBJECT. It does not attempt to reclassify the
    overall shape.
    """

    if not isinstance(data, dict) or not data:
        raise StructureHandlerError("Nested handler requires a non-empty object.")

    # Pure nested means no top-level parameter leaves.
    for name, value in data.items():
        if is_parameter_leaf(value):
            raise StructureHandlerError(
                f"Nested handler received top-level parameter leaf '{name}'."
            )
        if not isinstance(value, dict) or not value:
            raise StructureHandlerError(
                f"Nested handler received unsupported top-level branch '{name}'."
            )

    flat: List[Dict[str, Any]] = []
    unmatched: List[Any] = []
    _walk_nested_container(data, normalize_test_name, flat, unmatched, ())
    return flat, unmatched


def handle_mixed_parameters(
    data: Dict[str, Any],
    normalize_test_name: Normalizer,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Handle a top-level mix of parameter leaves and nested containers.

    The mixed handler is intentionally an explicit dispatcher: direct leaves use
    flat semantics; container branches use nested semantics.
    """

    if not isinstance(data, dict) or not data:
        raise StructureHandlerError("Mixed handler requires a non-empty object.")

    flat: List[Dict[str, Any]] = []
    unmatched: List[Any] = []
    saw_leaf = False
    saw_container = False

    for name, value in data.items():
        if is_parameter_leaf(value):
            saw_leaf = True
            _append_leaf(flat, unmatched, name, value, normalize_test_name)
            continue

        if isinstance(value, dict) and value:
            saw_container = True
            _walk_nested_container(
                value,
                normalize_test_name,
                flat,
                unmatched,
                (str(name),),
            )
            continue

        raise StructureHandlerError(
            f"Mixed handler received unsupported top-level branch '{name}'."
        )

    if not (saw_leaf and saw_container):
        raise StructureHandlerError(
            "Mixed handler requires at least one direct leaf and one container."
        )

    return flat, unmatched


def is_scalar_group(value: Any) -> bool:
    """Return True for a grouped set of direct scalar observations.

    Example from a real report:

        "WBC Morphology": {
            "Hypochromia": "-",
            "Microcytosis": "5",
            "Anisocytosis": "Mild"
        }

    This is intentionally distinct from a parameter leaf (which has a Value
    field) and from a standard nested panel (whose children are parameter
    leaves/containers).
    """

    if not isinstance(value, dict) or not value or is_parameter_leaf(value):
        return False

    for child in value.values():
        if isinstance(child, (dict, list, tuple, set)):
            return False

    return True


def _preserve_scalar_observation_value(value: Any) -> Any:
    """Preserve scalar-group observations without clinical reinterpretation.

    Numeric-looking strings such as morphology grades/codes remain strings.
    The raw extracted JSON is the source of truth for this structure.
    """

    if isinstance(value, str):
        return value.strip()
    return value


def _append_scalar_group_observation(
    output: List[Dict[str, Any]],
    unmatched: List[Any],
    group_name: Any,
    observation_name: Any,
    raw_value: Any,
    normalize_test_name: Normalizer,
) -> None:
    if isinstance(raw_value, (dict, list, tuple, set)):
        raise StructureHandlerError(
            f"Scalar group '{group_name}' contains non-scalar observation "
            f"'{observation_name}'."
        )

    normalized = normalize_test_name(observation_name)
    row = {
        "name": observation_name,
        "value": _preserve_scalar_observation_value(raw_value),
        "unit": "N/A",
        "referenceRange": "N/A",
        "sourceGroup": group_name,
        **normalized,
    }
    output.append(row)

    if not row.get("normalized"):
        unmatched.append(observation_name)


def handle_mixed_scalar_group_parameters(
    data: Dict[str, Any],
    normalize_test_name: Normalizer,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Handle standard parameter branches plus grouped scalar observations.

    This is a separate handler for the structure first observed in Report 3.
    Existing flat/array/nested/mixed handlers are not changed or reused as a
    universal parser. This handler explicitly dispatches only the branch types
    allowed by its own contract:

    - direct parameter leaves
    - standard nested containers of parameter leaves
    - scalar observation groups such as WBC Morphology
    """

    if not isinstance(data, dict) or not data:
        raise StructureHandlerError(
            "Mixed scalar-group handler requires a non-empty object."
        )

    flat: List[Dict[str, Any]] = []
    unmatched: List[Any] = []
    saw_scalar_group = False
    saw_standard_branch = False

    for name, value in data.items():
        if is_parameter_leaf(value):
            saw_standard_branch = True
            _append_leaf(flat, unmatched, name, value, normalize_test_name)
            continue

        if is_scalar_group(value):
            saw_scalar_group = True
            for observation_name, raw_value in value.items():
                _append_scalar_group_observation(
                    flat,
                    unmatched,
                    name,
                    observation_name,
                    raw_value,
                    normalize_test_name,
                )
            continue

        if isinstance(value, dict) and value:
            saw_standard_branch = True
            _walk_nested_container(
                value,
                normalize_test_name,
                flat,
                unmatched,
                (str(name),),
            )
            continue

        raise StructureHandlerError(
            f"Mixed scalar-group handler received unsupported top-level branch '{name}'."
        )

    if not saw_scalar_group:
        raise StructureHandlerError(
            "Mixed scalar-group handler requires at least one scalar group."
        )

    if not saw_standard_branch:
        raise StructureHandlerError(
            "Mixed scalar-group handler requires at least one standard parameter branch."
        )

    return flat, unmatched


def handle_parameter_object_with_metadata(
    data: Dict[str, Any],
    normalize_test_name: Normalizer,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Handle valid parameter branches contaminated by known metadata.

    Example:

        {
            "CBC": {
                "Haemoglobin": {...},
                "RBC": {...}
            },
            "Doctor's Notes": []
        }

    Only explicitly recognized metadata keys are ignored. Any other
    unsupported branch causes the handler to fail rather than guess.
    """

    if not isinstance(data, dict) or not data:
        raise StructureHandlerError(
            "Metadata-contaminated handler requires a non-empty object."
        )

    metadata_keys = [
        name for name in data
        if is_known_metadata_key(name)
    ]

    if not metadata_keys:
        raise StructureHandlerError(
            "Metadata-contaminated handler received no known metadata."
        )

    parameter_data = {
        name: value
        for name, value in data.items()
        if not is_known_metadata_key(name)
    }

    if not parameter_data:
        raise StructureHandlerError(
            "Metadata-contaminated object contains no parameter branches."
        )

    flat: List[Dict[str, Any]] = []
    unmatched: List[Any] = []

    for name, value in parameter_data.items():

        # Standard direct parameter
        if is_parameter_leaf(value):
            _append_leaf(
                flat,
                unmatched,
                name,
                value,
                normalize_test_name,
            )
            continue

        # Scalar observation group, e.g. WBC Morphology
        if is_scalar_group(value):
            for observation_name, raw_value in value.items():
                _append_scalar_group_observation(
                    flat,
                    unmatched,
                    name,
                    observation_name,
                    raw_value,
                    normalize_test_name,
                )
            continue

        # Standard nested panel
        if isinstance(value, dict) and value:
            _walk_nested_container(
                value,
                normalize_test_name,
                flat,
                unmatched,
                (str(name),),
            )
            continue

        # Anything else remains unsafe.
        raise StructureHandlerError(
            f"Metadata-contaminated handler received unsupported "
            f"parameter branch '{name}'."
        )

    return flat, unmatched 

def handle_nested_object_with_scalar_groups(
    data: Dict[str, Any],
    normalize_test_name: Normalizer,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Handle nested parameter trees that contain scalar groups deeper inside.

    Example:

        {
            "CBC+ESR": {
                "RBC PARAMETERS": {
                    "Haemoglobin": {
                        "Value": 6.5,
                        "Reference Range": "12.0-15.0",
                        "Unit": "g/dL"
                    }
                },
                "RBC Morphology": {
                    "Hypochromia": "-",
                    "Microcytosis": 5,
                    "Anisocytosis": "Mild"
                },
                "ESR": {
                    "Value": 51,
                    "Reference Range": "2-30",
                    "Unit": "mm at 1 hr"
                }
            }
        }

    This handler is deliberately separate from the standard nested handler.
    """

    if not isinstance(data, dict) or not data:
        raise StructureHandlerError(
            "Nested scalar-group handler requires a non-empty object."
        )

    flat: List[Dict[str, Any]] = []
    unmatched: List[Any] = []
    saw_scalar_group = False

    def walk(value: Dict[str, Any], path: Tuple[str, ...]) -> None:
        nonlocal saw_scalar_group

        if not isinstance(value, dict) or not value:
            joined = " > ".join(path) or "<root>"
            raise StructureHandlerError(
                f"Empty/invalid nested scalar-group container at {joined}."
            )

        for name, child in value.items():
            current_path = (*path, str(name))

            if is_parameter_leaf(child):
                _append_leaf(
                    flat,
                    unmatched,
                    name,
                    child,
                    normalize_test_name,
                )
                continue

            if is_scalar_group(child):
                saw_scalar_group = True

                for observation_name, raw_value in child.items():
                    _append_scalar_group_observation(
                        flat,
                        unmatched,
                        name,
                        observation_name,
                        raw_value,
                        normalize_test_name,
                    )

                continue

            if isinstance(child, dict) and child:
                walk(child, current_path)
                continue

            joined = " > ".join(current_path)

            raise StructureHandlerError(
                f"Unsupported nested scalar-group branch at "
                f"{joined}: {type(child).__name__}."
            )

    walk(data, ())

    if not saw_scalar_group:
        raise StructureHandlerError(
            "Nested scalar-group handler requires at least one scalar group."
        )

    return flat, unmatched    


