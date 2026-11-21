"""Agent card checks: freshness, nonce binding, ML-DSA-65 signature."""

from __future__ import annotations

import hmac
from datetime import datetime

from asqav.verifier.verify_receipt import _parse_stamp, _strict_b64, canonical_json

CARD_TTL_SECONDS = 300
CARD_EXPIRED = "card_expired"
NONCE_MISMATCH = "nonce_mismatch"
NONCE_UNBOUND = "nonce_unbound"
SIGNATURE_INVALID = "signature_invalid"


def _expired(card: dict, now) -> bool:
    """Expiry stamp: card_expires_at, else card_signed_at + the TTL."""
    field = "card_expires_at" if card.get("card_expires_at") is not None else "card_signed_at"
    stamp = _parse_stamp(card.get(field))
    if stamp is None or not isinstance(now, datetime) or now.tzinfo is None:
        return True
    return now >= stamp if field == "card_expires_at" else (now - stamp).total_seconds() >= CARD_TTL_SECONDS


def _nonce_failure(card: dict, expected_nonce) -> str | None:
    if expected_nonce is None:
        return None
    if card.get("nonce_bound") is False or not card.get("client_nonce"):
        return NONCE_UNBOUND
    try:
        return None if hmac.compare_digest(card["client_nonce"], expected_nonce) else NONCE_MISMATCH
    except TypeError:
        return NONCE_MISMATCH


def _signature_failure(card: dict) -> str | None:
    sig = _strict_b64(card.get("card_signature"))
    pk = _strict_b64(card.get("agent_public_key"))
    if sig is None or pk is None:
        return SIGNATURE_INVALID
    try:
        from dilithium_py.ml_dsa import ML_DSA_65

        msg = canonical_json({k: v for k, v in card.items() if k != "card_signature"})
        ok = ML_DSA_65.verify(pk, msg, sig)
    except Exception:
        return SIGNATURE_INVALID
    return None if ok else SIGNATURE_INVALID


def verify_card(card, expected_nonce, now) -> list[str]:
    """Failures in fixed order; [] passes. Bad input fails closed."""
    card = card if isinstance(card, dict) else {}
    failures = [CARD_EXPIRED] if _expired(card, now) else []
    for failure in (_nonce_failure(card, expected_nonce), _signature_failure(card)):
        if failure is not None:
            failures.append(failure)
    return failures
