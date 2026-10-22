"""Oracle failure classes (574 D5a, from 418): INVALID a resolved key and failed
binding, UNVERIFIABLE a stopped recompute. _axis delegates to classify."""
from __future__ import annotations

from . import crypto

#: Failure classes (criterion 418). Literals, not logic: pinned equal
#: to the oracle core's by the vocabulary test below in spirit.
FAILURE_INVALID = "invalid"
FAILURE_UNVERIFIABLE = "unverifiable"

#: Axes whose FAIL proves a binding failed (418 invalid; see classify).
_ALWAYS_INVALID_AXES = frozenset(
    {
        "signature",
        "key_status",
        "issuer_bind",
        "key_binding",
        "counterparty",
        "payload_digest",
        "outer_copies",
        "nonce",
        "seq",
        "parent_signature",
        "pdp_signature",
        "anchors",
    }
)


def classify(axis: str, result: str, note: str) -> str | None:
    """Map an axis outcome to its failure class; FAIL needs resolved inputs.
    FAILs: seq a bad counter, parent/pdp no sig, key_binding a bad claim."""
    if result == crypto.PASS:
        return None
    if result == crypto.SKIPPED:
        return FAILURE_UNVERIFIABLE
    if axis in _ALWAYS_INVALID_AXES:
        return FAILURE_INVALID
    if axis == "chain":
        # A mismatched link or a cross-format predecessor is a proven
        # break; a predecessor the canonicaliser cannot walk is a stop.
        if note.startswith("chain break:") or note == "predecessor is a different receipt format":
            return FAILURE_INVALID
        return FAILURE_UNVERIFIABLE
    if axis == "skew":
        if note.startswith("unparseable issued_at"):
            return FAILURE_UNVERIFIABLE
        return FAILURE_INVALID
    if axis == "structure":
        if note.startswith("unsupported ACTA alg") or note.startswith(
            "unsupported signature algorithm"
        ):
            return FAILURE_INVALID
        if note.startswith("key_purpose mismatch"):
            return FAILURE_INVALID
        if "signing-key DID != issuer DID" in note:
            return FAILURE_INVALID
        if note.startswith("proof @context is not a prefix"):
            return FAILURE_INVALID
        if note.startswith("unregistered type ") or note.startswith(
            "malformed type "
        ):
            return FAILURE_INVALID
        return FAILURE_UNVERIFIABLE
    if axis == "expiry":
        if note.startswith("unreadable expires_at"):
            return FAILURE_UNVERIFIABLE
        return FAILURE_INVALID
    # issuer_key / ingest and anything unlisted: never a proven break.
    return FAILURE_UNVERIFIABLE
