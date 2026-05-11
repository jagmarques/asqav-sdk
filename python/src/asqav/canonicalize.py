"""Fingerprint helpers for the Asqav SDK.

This module produces the bytes the Asqav backend signs (and, in
hash-only mode, the bytes the SDK hashes locally before sending to the
cloud).

Three helpers are provided:

* :func:`canonicalize` returns standard JSON bytes for any
  JSON-serializable Python value, following the conventions in
  ``conformance/vectors.json`` (JCS subset: keys sorted, no whitespace,
  raw UTF-8, no trailing newline).
* :func:`canonicalize_tool_args` is a user-facing pre-validator over
  ``tool_args`` payloads that rejects floats with a JSON-pointer-locating
  error before delegating to :func:`canonicalize`. The Compliance
  Receipts profile excludes floats from the signed canonical scope
  because canonical JSON only fixes byte-stable numeric output for
  safe-range integers; floats drift across runtimes.
* :func:`hash_action` returns a self-describing hash string of the form
  ``"sha256:<64hex>"`` built from a ``{action_type, context}`` dict. An
  optional ``salt`` switches it to HMAC-SHA-256 for organizations that
  use a per-org keyed hash.

Customers without the Asqav SDK can reproduce these helpers in a few lines
of stdlib Python (see ``docs/fingerprint-spec.md``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

__all__ = [
    "canonicalize",
    "canonicalize_action",
    "canonicalize_tool_args",
    "hash_action",
]


def canonicalize_action(
    action_type: str,
    context: "dict[str, Any] | None" = None,
) -> bytes:
    """Canonical bytes of ``{action_type, context}`` for hashing or signing."""
    return canonicalize({"action_type": action_type, "context": context or {}})


def canonicalize(obj: Any) -> bytes:
    """Return standard JSON bytes for ``obj``.

    Implementation: ``json.dumps`` with ``sort_keys=True``,
    ``separators=(",", ":")`` and ``ensure_ascii=False``. This is a
    faithful JCS subset for the value types the Asqav cloud actually
    accepts (``dict``, ``list``, ``str``, ``int``, ``float``, ``bool``,
    ``None``). The conformance vectors at
    ``conformance/vectors.json`` are generated with the exact same
    rules so SDKs written against this helper agree with the backend
    byte-for-byte.

    Notes:
        * Object keys are sorted lexicographically (Python ``sorted``).
        * No whitespace anywhere.
        * Unicode characters above U+001F are emitted as raw UTF-8.
        * ``NaN`` / ``Infinity`` are not legal JSON; ``json.dumps`` will
          raise unless the caller pre-filtered them.

    Args:
        obj: Any JSON-serializable value.

    Returns:
        UTF-8 encoded JSON bytes in standard form.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _find_float_pointer(value: Any, pointer: str) -> str | None:
    """Return JSON pointer to the first float found in ``value``, else None.

    Walks dicts and lists depth-first in insertion order so the reported
    pointer is stable for a given input. ``bool`` is excluded because
    Python's ``isinstance(True, int)`` is True and ``bool`` is JCS-safe.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        return pointer
    if isinstance(value, dict):
        for key, sub in value.items():
            token = str(key).replace("~", "~0").replace("/", "~1")
            hit = _find_float_pointer(sub, f"{pointer}/{token}")
            if hit is not None:
                return hit
        return None
    if isinstance(value, list):
        for idx, sub in enumerate(value):
            hit = _find_float_pointer(sub, f"{pointer}/{idx}")
            if hit is not None:
                return hit
        return None
    return None


def canonicalize_tool_args(args: dict[str, Any] | list[Any]) -> bytes:
    """Return JCS-canonical bytes of ``tool_args``, rejecting floats.

    Floats are not byte-stable across runtimes, so the Compliance Receipts
    profile keeps them out of the signed canonical scope. Serialize
    numerics as strings (``"1.5"``) or integer-rational pairs before
    calling this helper. Booleans and integers (including arbitrarily
    large ``int``) are accepted because JCS pins their byte form.
    See ``docs/fingerprint-spec.md`` for the canonical-form rules.

    Args:
        args: A dict or list of tool arguments. Leaves may be ``str``,
            ``int``, ``bool``, ``None``, ``dict``, or ``list``. ``float``
            is rejected.

    Returns:
        UTF-8 encoded JCS-canonical bytes.

    Raises:
        ValueError: If any leaf is a ``float``. The message includes the
            JSON pointer to the offending value so the caller can fix
            the input.
    """
    pointer = _find_float_pointer(args, "")
    if pointer is not None:
        raise ValueError(
            f"tool_args float at {pointer or '/'}: floats are not byte-stable "
            "across runtimes per RFC 8785 section 3.2.2; serialize numerics "
            "as strings or rationals before passing to canonicalize_tool_args"
        )
    return canonicalize(args)


def hash_action(
    action_type: str,
    context: dict[str, Any] | None = None,
    salt: bytes | None = None,
) -> str:
    """Return ``"sha256:<hex>"`` for the fingerprint bytes of an action.

    The hashed object is ``{"action_type": action_type, "context": context}``
    formatted via :func:`canonicalize`. This matches the conformance
    vectors and the backend's hash-only ingestion shape.

    With ``salt`` set, returns HMAC-SHA-256 instead of plain SHA-256.
    Use the per-organization salt fetched from the Asqav dashboard.

    Args:
        action_type: The action type string (e.g. ``"api:call"``).
        context: The action's context dict (any JSON-serializable shape).
        salt: Optional 32-byte per-org salt for keyed hashing.

    Returns:
        Self-describing hash string ``"sha256:<64 hex chars>"``.
    """
    canonical = canonicalize_action(action_type, context)
    if salt is not None:
        digest = hmac.new(salt, canonical, hashlib.sha256).hexdigest()
    else:
        digest = hashlib.sha256(canonical).hexdigest()
    return f"sha256:{digest}"
