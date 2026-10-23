"""Caller-held keyed commitments for opaque claims (draft-ietf-keytrans).
The caller holds the key; this module never generates, stores or sends it."""

from __future__ import annotations

import hashlib
import hmac
import secrets

__all__ = ["new_opening", "commit"]

_OPENING_BYTES = 16


def new_opening() -> bytes:
    """Return 16 fresh random bytes for use as a commitment opening.

    The opening is caller entropy that makes each commitment unlinkable.
    It is generated locally with the OS CSPRNG and never leaves the caller.
    """
    return secrets.token_bytes(_OPENING_BYTES)


def commit(
    key: bytes,
    opening: bytes,
    label: str,
    version: int,
    data: bytes,
) -> str:
    """Return hex HMAC-SHA256 over ``opening || label || version || data``.
    The caller supplies ``key``; it is never generated, stored or sent."""
    # Key material is caller-owned. It is read here and nowhere else, and it
    # never leaves the caller's process. Do not add a default or a fallback.
    framed = opening + label.encode("utf-8") + version.to_bytes(4, "big") + data
    return hmac.new(key, framed, hashlib.sha256).hexdigest()
