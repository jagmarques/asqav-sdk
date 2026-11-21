"""verify_card freshness, nonce binding and signature negative controls."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone

import pytest

from asqav.verifier.verify_card import (
    CARD_EXPIRED,
    CARD_TTL_SECONDS,
    NONCE_MISMATCH,
    NONCE_UNBOUND,
    SIGNATURE_INVALID,
    verify_card,
)
from asqav.verifier.verify_receipt import canonical_json

ML_DSA_65 = pytest.importorskip("dilithium_py.ml_dsa").ML_DSA_65
SIGNED_AT = datetime(2026, 4, 22, 20, 0, tzinfo=timezone.utc)
NONCE = "n" * 32


@pytest.fixture(scope="module")
def keypair():
    return ML_DSA_65.keygen()


def _signed_card(keypair, **fields):
    pk, sk = keypair
    card = {
        "agent_id": "agt_demo",
        "algorithm": "ML-DSA-65",
        "agent_public_key": base64.b64encode(pk).decode(),
        "card_signed_at": SIGNED_AT.isoformat(),
        "card_expires_at": (SIGNED_AT + timedelta(seconds=CARD_TTL_SECONDS)).isoformat(),
        "card_version": "a2a.v1",
        "name": "demo-agent",
        "schemaVersion": "1.0",
        "nonce_bound": True,
        "client_nonce": NONCE,
    }
    for name, value in fields.items():
        if value is None:
            card.pop(name, None)
        else:
            card[name] = value
    card["card_signature"] = base64.b64encode(ML_DSA_65.sign(sk, canonical_json(card))).decode()
    return card


def test_signed_fresh_card_passes(keypair):
    card = _signed_card(keypair)
    assert verify_card(card, NONCE, SIGNED_AT + timedelta(seconds=60)) == []


# NEGCTL
def test_expired_card_returns_card_expired(keypair):
    now = SIGNED_AT + timedelta(seconds=60)
    card = _signed_card(keypair, card_expires_at=(now - timedelta(seconds=1)).isoformat())
    assert verify_card(card, NONCE, now) == [CARD_EXPIRED]
    assert verify_card(_signed_card(keypair, card_expires_at="?"), NONCE, now) == [CARD_EXPIRED]
    stale = _signed_card(keypair, card_expires_at=None)
    past_ttl = SIGNED_AT + timedelta(seconds=CARD_TTL_SECONDS + 1)
    assert verify_card(stale, NONCE, past_ttl) == [CARD_EXPIRED]


# NEGCTL
def test_unbound_card_with_expected_nonce_returns_nonce_unbound(keypair):
    now = SIGNED_AT + timedelta(seconds=60)
    unbound = _signed_card(keypair, nonce_bound=False)
    assert verify_card(unbound, NONCE, now) == [NONCE_UNBOUND]
    unmarked = _signed_card(keypair, nonce_bound=False, client_nonce=None)
    assert verify_card(unmarked, "a" * 32, now) == [NONCE_UNBOUND]


def test_mismatched_nonce_returns_nonce_mismatch(keypair):
    card = _signed_card(keypair)
    assert verify_card(card, "b" * 32, SIGNED_AT + timedelta(seconds=60)) == [NONCE_MISMATCH]


def test_tampered_card_returns_signature_invalid(keypair):
    card = _signed_card(keypair)
    card["card_version"] = "a2a.v0"
    assert verify_card(card, NONCE, SIGNED_AT + timedelta(seconds=60)) == [SIGNATURE_INVALID]
