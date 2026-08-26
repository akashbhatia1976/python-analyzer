"""Aether Medical Parameters structure router (P1).

The router classifies the raw `Medical Parameters` payload into one explicitly
supported shape and dispatches to a narrow handler. Unknown/ambiguous shapes
fail safely with zero canonical parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional

from parameter_handlers import (
    StructureHandlerError,
    handle_array_parameters,
    handle_flat_parameters,
    handle_mixed_parameters,
    handle_mixed_scalar_group_parameters,
    handle_nested_parameters,
    handle_nested_object_with_scalar_groups,
    handle_parameter_object_with_metadata,
    is_known_metadata_key,
    is_parameter_leaf,
    is_scalar_group,
)


ROUTER_VERSION = "parameter_router_v1_3"

FLAT_OBJECT = "flat_object"
ARRAY = "array"
NESTED_OBJECT = "nested_object"
MIXED_OBJECT = "mixed_object"
MIXED_SCALAR_GROUP_OBJECT = "mixed_scalar_group_object"
PARAMETER_OBJECT_WITH_METADATA = "parameter_object_with_metadata"
NESTED_OBJECT_WITH_SCALAR_GROUPS = "nested_object_with_scalar_groups"
UNKNOWN = "unknown"


@dataclass
class RouterMetadata:
    parameterStructure: str
    routerVersion: str = ROUTER_VERSION
    parameterCount: int = 0
    unmatchedCount: int = 0
    status: str = "completed"
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_supported_nested_container(value: Any) -> bool:
    """Validate a standard nested container subtree without extracting.

    Standard nested containers may contain only parameter leaves or other
    non-empty dict containers. Scalar observation groups are deliberately not
    accepted here; they are a separate routed structure with a separate
    handler.
    """

    if not isinstance(value, dict) or not value:
        return False

    for child in value.values():
        if is_parameter_leaf(child):
            continue
        if isinstance(child, dict) and child:
            if not _is_supported_nested_container(child):
                return False
            continue
        return False

    return True


def _is_supported_array(data: Any) -> bool:
    if not isinstance(data, list) or not data:
        return False

    name_keys = {"Test Name", "Name", "Parameter", "testName", "name", "parameter"}

    for item in data:
        if not isinstance(item, dict):
            return False
        if not any(key in item and item.get(key) for key in name_keys):
            return False
        if not is_parameter_leaf(item):
            return False

    return True

def _contains_nested_scalar_group(data: Any, depth: int = 0) -> bool:
    """Return True when a scalar group exists below the top level.

    A scalar group at depth 0 belongs to the existing
    MIXED_SCALAR_GROUP_OBJECT route.

    This route is specifically for scalar groups occurring inside
    a nested parameter tree.
    """

    if not isinstance(data, dict) or not data:
        return False

    for value in data.values():
        if is_parameter_leaf(value):
            continue

        if is_scalar_group(value):
            if depth >= 1:
                return True
            continue

        if isinstance(value, dict) and value:
            if _contains_nested_scalar_group(value, depth + 1):
                return True

    return False


def detect_parameter_structure(data: Any) -> str:
    """Deterministically classify one known router shape.

    This function never extracts or normalizes parameters.
    """

    if _is_supported_array(data):
        return ARRAY

    if not isinstance(data, dict) or not data:
        return UNKNOWN

        # Known metadata occasionally leaks into Medical Parameters.
    # Route it explicitly rather than weakening the existing handlers.
    metadata_keys = [
        name for name in data
        if is_known_metadata_key(name)
    ]

    if metadata_keys:
        parameter_data = {
            name: value
            for name, value in data.items()
            if not is_known_metadata_key(name)
        }

        # Metadata by itself is not a valid parameter structure.
        if not parameter_data:
            return UNKNOWN

        # Every remaining branch must still be a structure we explicitly know.
        for value in parameter_data.values():
            if is_parameter_leaf(value):
                continue

            if is_scalar_group(value):
                continue

            if (
                isinstance(value, dict)
                and value
                and _is_supported_nested_container(value)
            ):
                continue

            return UNKNOWN

        return PARAMETER_OBJECT_WITH_METADATA

    if _contains_nested_scalar_group(data):
        return NESTED_OBJECT_WITH_SCALAR_GROUPS    
            

    leaf_count = 0
    container_count = 0
    scalar_group_count = 0

    for value in data.values():
        if is_parameter_leaf(value):
            leaf_count += 1
            continue

        if is_scalar_group(value):
            scalar_group_count += 1
            continue

        if isinstance(value, dict) and value and _is_supported_nested_container(value):
            container_count += 1
            continue

        return UNKNOWN

    # Existing four routes remain exactly as before when no scalar group exists.
    if scalar_group_count == 0:
        if leaf_count == len(data):
            return FLAT_OBJECT

        if container_count == len(data):
            return NESTED_OBJECT

        if leaf_count > 0 and container_count > 0 and leaf_count + container_count == len(data):
            return MIXED_OBJECT

        return UNKNOWN

    # New explicit structure: one or more scalar observation groups mixed with
    # at least one standard leaf/container branch. This is the shape observed in
    # Report 3 (e.g. WBC Morphology alongside ordinary lab parameters).
    if (
        scalar_group_count > 0
        and (leaf_count > 0 or container_count > 0)
        and leaf_count + container_count + scalar_group_count == len(data)
    ):
        return MIXED_SCALAR_GROUP_OBJECT

    return UNKNOWN


def route_parameters(
    data: Any,
    normalize_test_name: Callable[[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    """Classify + dispatch and return canonical rows plus router metadata.

    Unknown/ambiguous data deliberately returns zero parameters. This lets the
    existing Node ingestion safety path queue the report for review rather than
    persisting guessed clinical data.
    """

    structure = detect_parameter_structure(data)

    if structure == UNKNOWN:
        metadata = RouterMetadata(
            parameterStructure=UNKNOWN,
            status="needs_review",
            reason="UNSUPPORTED_OR_AMBIGUOUS_PARAMETER_STRUCTURE",
        )
        return {"parameters": [], "unmatched": [], "metadata": metadata.to_dict()}

    handlers = {
        FLAT_OBJECT: handle_flat_parameters,
        ARRAY: handle_array_parameters,
        NESTED_OBJECT: handle_nested_parameters,
        MIXED_OBJECT: handle_mixed_parameters,
        MIXED_SCALAR_GROUP_OBJECT: handle_mixed_scalar_group_parameters,
        PARAMETER_OBJECT_WITH_METADATA: handle_parameter_object_with_metadata,
        NESTED_OBJECT_WITH_SCALAR_GROUPS: handle_nested_object_with_scalar_groups,
    }

    try:
        flat, unmatched = handlers[structure](data, normalize_test_name)
    except StructureHandlerError as exc:
        metadata = RouterMetadata(
            parameterStructure=UNKNOWN,
            status="needs_review",
            reason=f"HANDLER_CONTRACT_FAILED: {exc}",
        )
        return {"parameters": [], "unmatched": [], "metadata": metadata.to_dict()}

    metadata = RouterMetadata(
        parameterStructure=structure,
        parameterCount=len(flat),
        unmatchedCount=len(unmatched),
        status="completed",
    )

    return {
        "parameters": flat,
        "unmatched": unmatched,
        "metadata": metadata.to_dict(),
    }
