"""Fingerprint helpers producing the bytes the Asqav backend signs.

In hash-only mode the SDK hashes these same bytes locally. See
``docs/fingerprint-spec.md`` for canonical-form rules.
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
    """Return JCS-subset JSON bytes for ``obj``; matches the conformance vectors byte-for-byte."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _find_float_pointer(value: Any, pointer: str) -> str | None:
    """Return JSON pointer to the first float in ``value`` (depth-first); ``bool`` is JCS-safe."""
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
    """Return JCS-canonical bytes of ``tool_args``; raises ``ValueError`` (with JSON
    pointer) on any float leaf since floats aren't byte-stable across runtimes.
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
    """Return ``"sha256:<hex>"`` of canonicalized ``{action_type, context}``; with
    ``salt`` set, uses HMAC-SHA-256 keyed by that caller-held salt instead of plain
    SHA-256. The prefix reads ``sha256`` either way, so a verifier that recomputes
    per ``docs/fingerprint-spec.md`` cannot tell a salted digest from a plain one.
    """
    canonical = canonicalize_action(action_type, context)
    if salt is not None:
        digest = hmac.new(salt, canonical, hashlib.sha256).hexdigest()
    else:
        digest = hashlib.sha256(canonical).hexdigest()
    return f"sha256:{digest}"
