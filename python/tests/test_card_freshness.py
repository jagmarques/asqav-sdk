"""verify_card freshness, nonce binding and signature negative controls."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
VECTORS = Path(__file__).resolve().parents[2] / "conformance" / "vectors.json"
ASQAV_SHA256 = {
    "stale_card": "0dcced0b29db5e9c29f5b71b4693c01443371042dd523e97fb917dad3b4040e9",
    "nonce_mismatch": "0d3916bae22c889c9bf77c2edecf91849366a8b3b8fb64a1d291e9a319904c9a",
    "card_version_downgrade": "9d62c8f11b8d3922e838edfecd866eae20e4865b38cdd58092e8b592aa789c30",
}


def _sdk_vector(name):
    corpus = json.loads(VECTORS.read_text())["vectors"]
    return next(vector for vector in corpus if vector["name"] == name)


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


def test_sdk_card_vectors_match_asqav():
    for name, digest in ASQAV_SHA256.items():
        vector = _sdk_vector(name)
        canonical = hashlib.sha256(canonical_json(vector["input"])).hexdigest()
        assert canonical == vector["sha256"] == digest


# NEGCTL
def test_stale_card_vector_returns_card_expired():
    vector = _sdk_vector("stale_card")
    now = datetime(2026, 4, 22, 20, 1, tzinfo=timezone.utc)
    assert CARD_EXPIRED in verify_card(dict(vector["input"]), None, now)


# NEGCTL
def test_nonce_mismatch_vector_returns_nonce_mismatch():
    vector = _sdk_vector("nonce_mismatch")
    now = datetime.fromisoformat(vector["input"]["card_signed_at"]) + timedelta(seconds=60)
    failures = verify_card(dict(vector["input"]), vector["peer_sent_nonce"], now)
    assert NONCE_MISMATCH in failures
    assert CARD_EXPIRED not in failures


# NEGCTL
def test_downgrade_vector_returns_signature_invalid():
    vector = _sdk_vector("card_version_downgrade")
    now = datetime.fromisoformat(vector["input"]["card_signed_at"]) + timedelta(seconds=60)
    assert SIGNATURE_INVALID in verify_card(dict(vector["input"]), None, now)
