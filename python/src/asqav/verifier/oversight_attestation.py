from binascii import Error
from typing import Any

from asqav.canonicalize import canonicalize
from asqav.verifier.oracle.crypto import PASS, verify_signature
from asqav.verifier.verify_receipt import _strict_b64

MISSING = "missing/unverified"
ASSERTED = "asserted/unverified"
SIGNATURE_VERIFIED = "signature-verified"
NOT_APPLICABLE = "not_applicable"
METHODS = frozenset({"manual_assertion", "sso", "hardware_key", "signed_statement"})


def oversight_attestation(
    payload: dict[str, Any], *, reviewer_key: tuple[str, bytes] | None = None,
) -> str:
    if payload.get("type") != "protectmcp:lifecycle:oversight_ruling":
        return NOT_APPLICABLE
    reviewer = payload.get("reviewer")
    att = reviewer.get("attestation") if isinstance(reviewer, dict) else None
    if not isinstance(att, dict) or att.get("method") not in tuple(METHODS):
        return MISSING
    if att["method"] == "manual_assertion" or reviewer_key is None:
        return ASSERTED
    sig = att.get("signature")
    if not isinstance(sig, dict):
        return ASSERTED
    if any(not isinstance(sig.get(k), str) or not sig[k].strip() for k in ("alg", "kid", "sig")):
        return ASSERTED
    if sig["alg"] != reviewer_key[0]:
        return ASSERTED
    try:
        raw = _strict_b64(sig["sig"])
        message = canonicalize({k: v for k, v in att.items() if k != "signature"})
        if raw is not None and verify_signature(*reviewer_key, message, raw)[0] == PASS:
            return SIGNATURE_VERIFIED
    except (ValueError, TypeError, Error):
        return ASSERTED
    return ASSERTED
