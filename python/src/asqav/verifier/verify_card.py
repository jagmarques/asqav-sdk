"""Agent card checks: freshness, nonce binding, ML-DSA-65 signature."""

from __future__ import annotations

import hmac
from datetime import datetime, timedelta

from asqav.verifier.verify_receipt import _parse_stamp, _strict_b64, canonical_json

CARD_TTL_SECONDS = 300
CARD_EXPIRED = "card_expired"
NONCE_MISMATCH = "nonce_mismatch"
NONCE_UNBOUND = "nonce_unbound"
SIGNATURE_INVALID = "signature_invalid"


def _expiry(card: dict):
    expires = card.get("card_expires_at")
    if expires is not None:
        return _parse_stamp(expires)
    signed = _parse_stamp(card.get("card_signed_at"))
    return None if signed is None else signed + timedelta(seconds=CARD_TTL_SECONDS)


def _nonce_failure(card: dict, expected_nonce) -> str | None:
    if expected_nonce is None:
        return None
    if card.get("nonce_bound") is False or not card.get("client_nonce"):
        return NONCE_UNBOUND
    try:
        matched = hmac.compare_digest(card["client_nonce"], expected_nonce)
    except TypeError:
        matched = False
    return None if matched else NONCE_MISMATCH


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
    """Failures in fixed order; [] only when every check passes. Never raises;
    unreadable input fails closed with named failures."""
    if not isinstance(card, dict):
        card = {}
    failures = []
    expiry = _expiry(card)
    if not isinstance(now, datetime) or now.tzinfo is None or expiry is None or now >= expiry:
        failures.append(CARD_EXPIRED)
    nonce = _nonce_failure(card, expected_nonce)
    if nonce is not None:
        failures.append(nonce)
    signature = _signature_failure(card)
    if signature is not None:
        failures.append(signature)
    return failures
