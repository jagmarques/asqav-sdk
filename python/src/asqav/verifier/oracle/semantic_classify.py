"""Independent failure classification for the oracle (574 D5a). Criterion 418, not a verifier's table:
INVALID means a resolved key and a binding failed; UNVERIFIABLE means the recompute could not complete.
A FAIL needs resolved inputs, so an unresolvable one SKIPs upstream; the per-axis preconditions are
in ``classify``. The axes in _ALWAYS_INVALID_AXES FAIL invalid; chain, skew, structure and expiry
split proven breaks from stops by note; an unlisted axis is never a break."""
# anchors FAILs on malformed or untrusted material, SKIPs when absent.
from __future__ import annotations

from . import crypto

#: Failure classes (criterion 418). Literals, not logic: pinned equal
#: to the oracle core's by the vocabulary test below in spirit.
FAILURE_INVALID = "invalid"
FAILURE_UNVERIFIABLE = "unverifiable"

#: Axes whose FAIL proves a binding failed (418 invalid; see docstring).
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
    """Map one axis outcome to its failure class (criterion 418). Preconditions: signature SKIPs with no key
    resolved; key_status/issuer_bind weigh a resolved entry; key_binding SKIPs when unrecomputable, FAILs a
    malformed claim; counterparty SKIPs what it cannot resolve, FAILs a mismatch; payload_digest FAILs on a
    malformed shape or a context disagreement; outer_copies FAILs on a disagreement with the signed bytes;
    nonce FAILs on a proven duplicate pair; seq FAILs on a withheld counter; parent/pdp FAIL on a failed
    verification with resolved keys; anchors FAILs on malformed or untrusted material, SKIPs when absent."""
    # The oracle's ``_axis`` constructor uses this; a standing corpus gate proves
    # behavior is preserved.
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
