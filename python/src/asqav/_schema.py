"""Dependency-free context-schema validator for ``Agent.sign``.
A ``{field: {"type": ..., "required": ...}}`` dict or a raising callable."""

from __future__ import annotations

from typing import Any

# Docs URL stamped on AsqavValidationError instances from this module.
_SCHEMA_DOCS_URL = "https://asqav.com/docs/structured-receipts"

# Maps schema "type" strings to the Python runtime type(s) they check against.
_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}

_VALID_TYPE_NAMES = frozenset(_TYPE_MAP)


def validate_context_schema(
    context: dict[str, Any] | None,
    schema: dict[str, Any],
) -> None:
    """Validate *context* against the built-in *schema* descriptor.
    ``None`` counts as ``{}``; raises ``AsqavValidationError`` on a bad one."""
    # Lazy import to avoid circular dependency with client.py.
    from .client import AsqavValidationError

    ctx = context or {}

    for field, descriptor in schema.items():
        required = descriptor.get("required", False)
        type_name = descriptor.get("type")

        if type_name is not None and type_name not in _VALID_TYPE_NAMES:
            raise ValueError(
                f"context_schema_error: unknown type '{type_name}' for field '{field}'; "
                f"must be one of {sorted(_VALID_TYPE_NAMES)}"
            )

        if field not in ctx:
            if required:
                raise AsqavValidationError(
                    f"context_schema_error: field '{field}' is required but missing",
                    docs_url=_SCHEMA_DOCS_URL,
                )
            continue

        if type_name is None:
            continue

        value = ctx[field]
        expected = _TYPE_MAP[type_name]

        # bool is a subclass of int; reject it from "number" explicitly so
        # {"active": True} does not silently pass a "number" field.
        if type_name == "number" and isinstance(value, bool):
            raise AsqavValidationError(
                f"context_schema_error: field '{field}' expected type number, got bool",
                docs_url=_SCHEMA_DOCS_URL,
            )

        if not isinstance(value, expected):
            actual = type(value).__name__
            raise AsqavValidationError(
                f"context_schema_error: field '{field}' expected type {type_name}, got {actual}",
                docs_url=_SCHEMA_DOCS_URL,
            )


def normalize_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return *context* with keys in stable sorted order, so signed bytes match.
    ``None`` is returned unchanged."""
    if context is None:
        return None
    return dict(sorted(context.items()))
