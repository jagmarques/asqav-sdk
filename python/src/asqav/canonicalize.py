"""Canonical JSON + content hashing for the asqav SDK.

This module exposes the byte-for-byte canonicalization used by the asqav
backend to produce the bytes that get signed (and, in hash-only mode, the
bytes that the SDK hashes locally before sending to the cloud).

Two helpers are provided:

* :func:`canonicalize` returns canonical JSON bytes for any
  JSON-serializable Python value, following the conventions in
  ``conformance/vectors.json`` (RFC 8785 JCS subset: keys sorted, no
  whitespace, raw UTF-8, no trailing newline).
* :func:`hash_action` returns a self-describing hash string of the form
  ``"sha256:<64hex>"`` over a canonical ``{action_type, context}`` dict.
  An optional ``salt`` switches it to HMAC-SHA-256 for organizations that
  use a per-org keyed hash.

Customers without an asqav SDK can reproduce these helpers in a few lines
of stdlib Python (see ``docs/canonicalization.md``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

__all__ = ["canonicalize", "hash_action"]


def canonicalize(obj: Any) -> bytes:
    """Return canonical JSON bytes for ``obj``.

    Implementation: ``json.dumps`` with ``sort_keys=True``,
    ``separators=(",", ":")`` and ``ensure_ascii=False``. This is a
    faithful subset of RFC 8785 (JCS) for the value types the asqav
    cloud actually accepts (``dict``, ``list``, ``str``, ``int``,
    ``float``, ``bool``, ``None``). The conformance vectors at
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
        UTF-8 encoded canonical JSON bytes.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hash_action(
    action_type: str,
    context: dict[str, Any] | None = None,
    salt: bytes | None = None,
) -> str:
    """Return ``"sha256:<hex>"`` for the canonical bytes of an action.

    The hashed object is ``{"action_type": action_type, "context": context}``
    canonicalized via :func:`canonicalize`. This matches the conformance
    vectors and the backend's hash-only ingestion shape.

    With ``salt`` set, returns HMAC-SHA-256 instead of plain SHA-256.
    Use the per-organization salt fetched from the asqav dashboard.

    Args:
        action_type: The action type string (e.g. ``"api:call"``).
        context: The action's context dict (any JSON-serializable shape).
        salt: Optional 32-byte per-org salt for keyed hashing.

    Returns:
        Self-describing hash string ``"sha256:<64 hex chars>"``.
    """
    payload = {"action_type": action_type, "context": context or {}}
    canonical = canonicalize(payload)
    if salt is not None:
        digest = hmac.new(salt, canonical, hashlib.sha256).hexdigest()
    else:
        digest = hashlib.sha256(canonical).hexdigest()
    return f"sha256:{digest}"
