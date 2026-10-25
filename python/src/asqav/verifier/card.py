"""Offline check of a signed agent card: signature, freshness and nonce echo.
Every failure is named; a card passes only with none of them."""

from __future__ import annotations

import base64
import binascii
import hmac
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from asqav._jcs import canonical_json
from asqav.verifier import verify_receipt as _vr

__all__ = [
    "CARD_EXPIRED",
    "NONCE_MISMATCH",
    "NONCE_UNBOUND",
    "SIGNATURE_INVALID",
    "DEFAULT_MAX_AGE_SECONDS",
    "CardVerdict",
    "verify_card",
]

CARD_EXPIRED = "card_expired"
NONCE_MISMATCH = "nonce_mismatch"
NONCE_UNBOUND = "nonce_unbound"
SIGNATURE_INVALID = "signature_invalid"

#: A stamp's zone designator, which a card's timestamps must carry.
_ZONE_RE = re.compile(r"(?:Z|[+-]\d{2}(?::?\d{2})?)$")

#: The cloud's card TTL; a card older than this is stale even if it claims more.
DEFAULT_MAX_AGE_SECONDS = 300


@dataclass(frozen=True)
class CardVerdict:
    failures: tuple[str, ...]
    details: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


    # A card field through the receipt verifier's one stamp parser; a stamp
    # with no zone is refused here, where that parser would read it as UTC.
def _instant(value: Any) -> datetime | None:
    if not isinstance(value, str) or _ZONE_RE.search(value) is None:
        return None
    return _vr._parse_stamp(value)


    # The signed bytes are the card minus card_signature, JCS-canonicalized.
def _signature_problem(card: dict[str, Any]) -> str | None:
    try:
        pk = base64.b64decode(card["agent_public_key"], validate=True)
        sig = base64.b64decode(card["card_signature"], validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error):
        return "agent_public_key or card_signature missing or not base64"
    signed = {k: v for k, v in card.items() if k != "card_signature"}
    verdict, detail = _vr.verify_signature(
        pk, canonical_json(signed), sig, card.get("algorithm")
    )
    return None if verdict == "PASS" else f"{verdict}: {detail}"


def _expiry_problem(
    card: dict[str, Any], at_time: datetime, max_age_seconds: int
) -> str | None:
    signed_at = _instant(card.get("card_signed_at"))
    if signed_at is None:
        return "card_signed_at missing or not a tz-aware timestamp"
    # A card signed ahead of the check would stay fresh as long as its stamp
    ahead = (signed_at - at_time).total_seconds()
    if ahead > _vr.SKEW_BOUND_SECONDS:
        return f"card_signed_at {ahead:.0f}s ahead of the check (> {_vr.SKEW_BOUND_SECONDS}s)"
    limit = signed_at + timedelta(seconds=max_age_seconds)
    if "card_expires_at" in card:
        expires_at = _instant(card["card_expires_at"])
        if expires_at is None:
            return "card_expires_at is not a tz-aware timestamp"
        limit = min(limit, expires_at)
    # Expired at the instant itself, the same boundary as receipt expiry.
    if at_time >= limit:
        return f"expired at {limit.isoformat()}"
    return None


def _nonce_failures(card: dict[str, Any], expected_nonce: str) -> list[str]:
    echoed = card.get("client_nonce")
    out = []
    if isinstance(echoed, str) and not hmac.compare_digest(
        echoed.encode(), expected_nonce.encode()
    ):
        out.append(NONCE_MISMATCH)
    if card.get("nonce_bound") is not True or not isinstance(echoed, str):
        out.append(NONCE_UNBOUND)
    return out


def verify_card(
    card: dict[str, Any],
    expected_nonce: str | None,
    at_time: datetime,
    *,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
) -> CardVerdict:
    """Check ``card`` at ``at_time`` against the nonce the caller sent.
    ``expected_nonce`` None: the caller sent none, so no nonce check runs."""
    if at_time.tzinfo is None:
        raise ValueError("at_time must be timezone-aware")
    failures: list[str] = []
    details: list[str] = []
    problem = _signature_problem(card)
    if problem is not None:
        failures.append(SIGNATURE_INVALID)
        details.append(problem)
    problem = _expiry_problem(card, at_time, max_age_seconds)
    if problem is not None:
        failures.append(CARD_EXPIRED)
        details.append(problem)
    if expected_nonce is not None:
        for name in _nonce_failures(card, expected_nonce):
            failures.append(name)
            details.append(f"{name}: card does not echo the expected nonce")
    return CardVerdict(tuple(failures), tuple(details))
