"""verify_card over the stale_card, nonce_mismatch and card_version_downgrade
vectors, plus cards signed here with a real ML-DSA-65 key."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from dilithium_py.ml_dsa import ML_DSA_65

from asqav._jcs import canonical_json
from asqav.verifier.card import (
    CARD_EXPIRED,
    DEFAULT_MAX_AGE_SECONDS,
    NONCE_MISMATCH,
    NONCE_UNBOUND,
    SIGNATURE_INVALID,
    verify_card,
)

VECTORS_PATH = Path(__file__).parent.parent.parent / "conformance" / "vectors.json"


def _vector(name: str) -> dict:
    doc = json.loads(VECTORS_PATH.read_text())
    return next(v for v in doc["vectors"] if v["name"] == name)


@pytest.fixture(scope="module")
def keypair() -> tuple[bytes, bytes]:
    return ML_DSA_65.keygen()


    # Re-sign a vector's card with a real key at the vector's own instant.
def _signed(vector: str, keypair, **fields) -> tuple[dict, datetime]:
    pk, sk = keypair
    card = dict(_vector(vector)["input"], agent_public_key=base64.b64encode(pk).decode())
    signed_at = datetime.fromisoformat(card["card_signed_at"])
    card["card_expires_at"] = (
        signed_at + timedelta(seconds=DEFAULT_MAX_AGE_SECONDS)
    ).isoformat()
    card.update(fields)
    sig = ML_DSA_65.sign(sk, canonical_json(card))
    card["card_signature"] = base64.b64encode(sig).decode()
    return card, signed_at + timedelta(seconds=1)


def test_stale_card_vector_is_card_expired() -> None:
    vec = _vector("stale_card")
    at = datetime.fromisoformat(_vector("nonce_mismatch")["input"]["card_signed_at"])
    verdict = verify_card(vec["input"], None, at)
    assert not verdict.ok
    assert CARD_EXPIRED in verdict.failures


def test_negctl_card_expires_at_in_the_past_is_card_expired_never_pass(
    keypair,
) -> None:
    """NEGCTL: a validly signed card whose card_expires_at is past fails."""
    card, at = _signed("stale_card", keypair)
    assert verify_card(card, None, at).ok
    expires_at = datetime.fromisoformat(card["card_expires_at"])
    verdict = verify_card(card, None, expires_at)
    assert verdict.failures == (CARD_EXPIRED,)
    assert not verdict.ok


def test_short_card_expires_at_wins_over_the_age_window(keypair) -> None:
    card, at = _signed("stale_card", keypair, card_expires_at="2020-01-01T00:00:01+00:00")
    assert verify_card(card, None, at - timedelta(seconds=1)).ok
    assert verify_card(card, None, at).failures == (CARD_EXPIRED,)


def test_nonce_mismatch_vector_is_nonce_mismatch() -> None:
    vec = _vector("nonce_mismatch")
    at = datetime.fromisoformat(vec["input"]["card_signed_at"])
    verdict = verify_card(vec["input"], vec["peer_sent_nonce"], at)
    assert NONCE_MISMATCH in verdict.failures
    assert not verdict.ok


def test_signed_nonce_echo_passes_and_a_peer_nonce_mismatches(keypair) -> None:
    vec = _vector("nonce_mismatch")
    card, at = _signed("nonce_mismatch", keypair, nonce_bound=True)
    assert verify_card(card, vec["input"]["client_nonce"], at).ok
    verdict = verify_card(card, vec["peer_sent_nonce"], at)
    assert verdict.failures == (NONCE_MISMATCH,)


def test_unbound_card_with_an_expected_nonce_is_nonce_unbound(keypair) -> None:
    vec = _vector("nonce_mismatch")
    card, at = _signed("stale_card", keypair, nonce_bound=False)
    assert verify_card(card, None, at).ok
    verdict = verify_card(card, vec["peer_sent_nonce"], at)
    assert verdict.failures == (NONCE_UNBOUND,)


def test_card_version_downgrade_vector_is_signature_invalid(keypair) -> None:
    vec = _vector("card_version_downgrade")
    card, at = _signed("card_version_downgrade", keypair, card_version="a2a.v1")
    assert verify_card(card, None, at).ok
    downgraded = dict(card, card_version=vec["input"]["card_version"])
    assert downgraded != card
    assert verify_card(downgraded, None, at).failures == (SIGNATURE_INVALID,)


def test_unsigned_vector_card_is_signature_invalid() -> None:
    vec = _vector("card_version_downgrade")
    at = datetime.fromisoformat(vec["input"]["card_signed_at"])
    assert verify_card(vec["input"], None, at).failures == (SIGNATURE_INVALID,)


def test_naive_verification_time_is_refused() -> None:
    vec = _vector("stale_card")
    with pytest.raises(ValueError, match="timezone-aware"):
        verify_card(vec["input"], None, datetime(2026, 4, 22, 20, 0, 0))
