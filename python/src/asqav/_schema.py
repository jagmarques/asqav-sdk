"""Lightweight, dependency-free context-schema validator (criterion 328).

Schema spec: a dict mapping field names to a field descriptor.

Field descriptor keys:
    type     - one of "string" | "number" | "boolean" | "object" | "array"
    required - bool, defaults to False

Example::

    schema = {
        "user_id": {"type": "string", "required": True},
        "score":   {"type": "number"},
        "tags":    {"type": "array"},
    }

Pass to ``Agent.sign(..., context_schema=schema)``. The SDK validates the
context dict against the schema and raises ``AsqavValidationError`` on
mismatch before any network call. On success the context is returned with
keys in stable sorted order so audit trails are consistent and queryable.

You may instead pass a callable as ``context_schema``:

    def my_validator(ctx: dict) -> None:
        import jsonschema
        jsonschema.validate(ctx, MY_FULL_JSON_SCHEMA)

    agent.sign("action", ctx, context_schema=my_validator)

A callable receives the (possibly None-coerced-to-{}) context dict and must
raise on invalid input. The built-in path needs no new runtime dependency.
"""

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

    Raises ``AsqavValidationError`` on the first offending field so the
    developer sees one clear message rather than a list dump.

    Args:
        context: The context dict passed to ``Agent.sign()``. ``None`` is
            treated as an empty dict (no user keys supplied).
        schema:  A ``{field_name: {"type": ..., "required": ...}}`` mapping.

    Raises:
        AsqavValidationError: When a required field is absent or a value
            does not match its declared type.
        ValueError: When the schema itself uses an unknown type name; this
            is a programmer error, not a runtime data error.
    """
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
    """Return *context* with keys in stable sorted order.

    Stable key order means two sign() calls with identical data but
    different insertion orders produce the same signed bytes, making
    receipts consistent and queryable.

    Returns ``None`` unchanged; an empty dict becomes ``{}``.
    """
    if context is None:
        return None
    return dict(sorted(context.items()))
