"""Caller-held keyed commitments for opaque-claim attestations.

Builds a randomised keyed commitment locally so a caller can seal a claim
before submitting it to asqav's /attestations route. The commitment is an
HMAC the caller seals with a key only they hold, so asqav can never open it.

The keyed framing (opening, label, version, data) follows the IETF KeyTrans
protocol draft: https://datatracker.ietf.org/doc/draft-ietf-keytrans-protocol/

SECURITY: the key is generated and held by the CALLER. This module never
sends, logs, persists, or default-generates key material. Key bytes only
ever live in the caller's memory for the lifetime of the HMAC call.
"""

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
    """Return lowercase hex of HMAC-SHA256 over the framed commitment input.

    The MAC input is ``opening || label || version || data`` with the label
    UTF-8 encoded and the version as a 4-byte big-endian integer, the keyed
    framing from the IETF KeyTrans draft. The caller supplies ``key``. This
    function never generates, stores, logs, or transmits the key.
    """
    # Key material is caller-owned. It is read here and nowhere else, and it
    # never leaves the caller's process. Do not add a default or a fallback.
    framed = opening + label.encode("utf-8") + version.to_bytes(4, "big") + data
    return hmac.new(key, framed, hashlib.sha256).hexdigest()
