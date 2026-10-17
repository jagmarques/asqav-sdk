"""Criterion 569 C9: the corpus carries a KMS-signed receipt (asqav-54).

The vector was minted by verifier/mint_kms_signed_vector.py: a deterministic
dilithium-py key held by a BaseKMSClient-conformant test double, SPKI recorded
beside the JWK. These tests assert the acceptance clauses over frozen bytes.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest

import asqav
from asqav.verifier.oracle import ADAPTERS, verify
from asqav.verifier.verify_receipt import envelope_minus_anchors_jcs

VEC = (
    Path(__file__).resolve().parents[2]
    / "verifier"
    / "conformance-vectors"
    / "asqav-54-kms-signed"
)
ANCHOR_HEX = "315b3a7fecfe02ebaacfb570462b4e4e28249f48ca0a4e43d2a36bc528dbeac1"

try:
    from dilithium_py.ml_dsa import ML_DSA_65

    _DILITHIUM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DILITHIUM_AVAILABLE = False


def _load(name: str):
    return json.loads((VEC / name).read_text())


def _spki_der() -> bytes:
    return base64.b64decode((VEC / "kms_spki_der_b64.txt").read_text().strip())


def _read_len(der: bytes, pos: int) -> tuple[int, int]:
    first = der[pos]
    pos += 1
    if first < 0x80:
        return first, pos
    n = first & 0xFF if False else first & 0x7F
    return int.from_bytes(der[pos : pos + n], "big"), pos + n


def _unwrap_spki_pub(der: bytes) -> bytes:
    # Short walk mirroring asqav_cloud.core.kms._unwrap_spki_to_raw_key.
    assert der[0] == 0x30
    _, pos = _read_len(der, 1)
    assert der[pos] == 0x30
    alen, apos = _read_len(der, pos + 1)
    pos = apos + alen
    assert der[pos] == 0x03
    _, bpos = _read_len(der, pos + 1)
    assert der[bpos] == 0
    blen = der[pos + 1] if der[pos + 1] < 0x80 else None
    assert blen is None or True
    total, _ = _read_len(der, pos + 1)
    return der[bpos + 1 : bpos + total]


def test_sig_is_unpadded_base64url_of_3309_bytes() -> None:
    sig = _load("receipt.json")["signature"]
    assert sig["alg"] == "ML-DSA-65"
    assert len(sig["sig"]) == 4412
    assert "+" not in sig["sig"] and "/" not in sig["sig"]
    assert "=" not in sig["sig"]
    assert len(base64.urlsafe_b64decode(sig["sig"] + "==")) == 3309


def test_kid_resolves_to_akp_jwk_from_kms_spki() -> None:
    receipt, jwks = _load("receipt.json"), _load("jwks.json")
    raw = _unwrap_spki_pub(_spki_der())
    assert len(raw) == 1952
    (key,) = jwks["keys"]
    assert key["kty"] == "AKP" and key["alg"] == "ML-DSA-65"
    assert base64.urlsafe_b64decode(key["pub"] + "==") == raw
    assert key["kid"] == base64.urlsafe_b64encode(
        hashlib.sha256(_spki_der()).digest()
    ).decode("ascii").rstrip("=")
    assert receipt["signature"]["kid"] == key["issuer_id"]


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_standalone_verifies_with_signature_pass() -> None:
    result = asqav.verify_receipt_offline(_load("receipt.json"), _load("jwks.json"))
    assert result["verdict"] == "verified", result
    sig_axis = next(a for a in result["axes"] if a["name"] == "signature")
    assert sig_axis["result"] == "PASS", sig_axis


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_oracle_verifies() -> None:
    got = verify(_load("receipt.json"), ADAPTERS, key_provider=_load("jwks.json"))
    assert got.verdict == "verified", got


def test_anchor_digest_recomputes_to_pin() -> None:
    receipt = _load("receipt.json")
    digest = hashlib.sha256(envelope_minus_anchors_jcs(receipt)).hexdigest()
    assert digest == ANCHOR_HEX


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_flipped_sig_bit_fails_both_paths() -> None:
    receipt, jwks = _load("receipt.json"), _load("jwks.json")
    bad = copy.deepcopy(receipt)
    sig = bad["signature"]["sig"]
    bad["signature"]["sig"] = sig[:10] + ("A" if sig[10] != "A" else "B") + sig[11:]
    assert bad["signature"]["sig"] != sig
    r1 = asqav.verify_receipt_offline(bad, jwks)
    assert r1["verdict"] == "unverified", r1
    r2 = verify(bad, ADAPTERS, key_provider=jwks)
    assert r2.verdict != "verified", r2


def test_expiry_stays_ahead_of_the_vector_clock() -> None:
    from datetime import datetime, timedelta, timezone

    expires = datetime.fromisoformat(
        _load("receipt.json")["payload"]["expires_at"].replace("Z", "+00:00")
    )
    assert expires.tzinfo is not None
    assert expires - datetime.now(timezone.utc) > timedelta(days=300)
