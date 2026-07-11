"""Offline receipt verification helpers: fetch_jwks + verify_receipt_offline.

All tests in this file must pass with NO network access. urllib.request.urlopen
is monkeypatched at the module level to hard-fail so any accidental HTTP call
turns into an immediate test failure.
"""

from __future__ import annotations

import copy
import json
import urllib.request
from pathlib import Path

import pytest

import asqav
from asqav.verifier import verify_receipt as vr

# --- Paths to the conformance vectors shipped in the repo ---
VECTORS = Path(__file__).parent.parent.parent / "verifier" / "conformance-vectors"
PASS_VECTOR = VECTORS / "asqav-01-genesis-permit"
TAMPER_VECTOR = VECTORS / "asqav-04-tamper-sig"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Hard-ban all HTTP while these tests run.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch):
    """Raise if anything tries to open a network connection."""

    def _no_network(*args, **kwargs):
        raise RuntimeError(f"Network access forbidden in offline tests: args={args!r}")

    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    yield


# ---------------------------------------------------------------------------
# fetch_jwks public API smoke-test (network forbidden -> expect RuntimeError).
# This proves the helper is exported and wired to the network call.
# ---------------------------------------------------------------------------


def test_fetch_jwks_exported():
    """fetch_jwks must be importable from the top-level asqav namespace."""
    assert callable(asqav.fetch_jwks)


def test_fetch_jwks_hits_network(monkeypatch):
    """fetch_jwks() must invoke urllib.request.urlopen (not silently bypass it)."""
    called = []

    def _capture(req, **kw):
        called.append(req.full_url if hasattr(req, "full_url") else str(req))
        raise RuntimeError("intercepted")

    monkeypatch.setattr(urllib.request, "urlopen", _capture)
    with pytest.raises(RuntimeError, match="intercepted"):
        asqav.fetch_jwks()
    assert called, "fetch_jwks must call urlopen"
    assert "jwks.json" in called[0] or ".well-known" in called[0]


# ---------------------------------------------------------------------------
# verify_receipt_offline: PASS path using the conformance-vector jwks.
# ---------------------------------------------------------------------------


def test_verify_receipt_offline_pass():
    """A valid Ed25519-signed receipt verifies PASS against the vector JWKS."""
    receipt = _load(PASS_VECTOR / "receipt.json")
    jwks = _load(PASS_VECTOR / "jwks.json")
    result = asqav.verify_receipt_offline(receipt, jwks)
    assert result["verdict"] == "PASS", f"axes: {result['axes']}"
    sig_axis = next(a for a in result["axes"] if a["name"] == "signature")
    assert sig_axis["result"] == "PASS"


def test_verify_receipt_offline_axes_structure():
    """Result dict must have the documented shape."""
    receipt = _load(PASS_VECTOR / "receipt.json")
    jwks = _load(PASS_VECTOR / "jwks.json")
    result = asqav.verify_receipt_offline(receipt, jwks)
    assert isinstance(result["verdict"], str)
    assert isinstance(result["axes"], list)
    assert all("name" in a and "result" in a and "note" in a for a in result["axes"])
    assert result["fmt"] == "asqav-native"


# ---------------------------------------------------------------------------
# verify_receipt_offline: FAIL on tampered receipt.
# ---------------------------------------------------------------------------


def test_verify_receipt_offline_tampered_receipt_fails():
    """A receipt with a flipped decision field must come back FAIL."""
    receipt = _load(TAMPER_VECTOR / "receipt.json")
    jwks = _load(TAMPER_VECTOR / "jwks.json")
    result = asqav.verify_receipt_offline(receipt, jwks)
    assert result["verdict"] == "FAIL", f"axes: {result['axes']}"
    sig_axis = next((a for a in result["axes"] if a["name"] == "signature"), None)
    assert sig_axis is not None
    assert sig_axis["result"] == "FAIL"


def test_verify_receipt_offline_tampered_payload_directly():
    """Mutating the payload after signing must also produce FAIL."""
    receipt = _load(PASS_VECTOR / "receipt.json")
    jwks = _load(PASS_VECTOR / "jwks.json")
    tampered = copy.deepcopy(receipt)
    tampered["payload"]["decision"] = "deny"
    result = asqav.verify_receipt_offline(tampered, jwks)
    assert result["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# verify_receipt_offline: INCOMPLETE when JWKS has no matching key.
# ---------------------------------------------------------------------------


def test_verify_receipt_offline_no_key_in_jwks():
    """Missing key in JWKS -> signature SKIPPED -> INCOMPLETE (never a PASS)."""
    receipt = _load(PASS_VECTOR / "receipt.json")
    result = asqav.verify_receipt_offline(receipt, {"keys": []})
    # Oracle SKIPs the signature when the key is absent; verdict is INCOMPLETE.
    assert result["verdict"] in ("FAIL", "INCOMPLETE")
    assert result["verdict"] != "PASS"
    sig_axis = next((a for a in result["axes"] if a["name"] == "signature"), None)
    assert sig_axis is not None
    assert sig_axis["result"] in ("SKIPPED", "FAIL")


# ---------------------------------------------------------------------------
# ML-DSA-65 path: PASS when dilithium-py is present, SKIPPED otherwise.
#
# NOTE: these are same-library interop round-trips (dilithium-py sign +
# dilithium-py verify). They confirm the wiring works but do NOT prove
# interop with real Asqav-cloud ML-DSA-65 signatures. A real-cloud
# payload-mode known-answer conformance vector is a documented follow-up.
# ---------------------------------------------------------------------------


def _make_mldsa_receipt_and_jwks():
    """Generate a real ML-DSA-65 receipt + JWKS using dilithium-py."""
    from dilithium_py.ml_dsa import ML_DSA_65

    pk, sk = ML_DSA_65.keygen()
    pk_b64 = __import__("base64").b64encode(pk).decode()
    kid = "test-mldsa-key-01"

    payload = {
        "type": "protectmcp:decision",
        "issued_at": "2026-06-19T00:00:00.000000Z",
        "issuer_id": kid,
        "action_ref": "sha256:" + "a" * 64,
        "payload_digest": {"hash": "b" * 64, "size": 128},
        "policy_digest": "sha256:" + "c" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
    }
    msg = vr.canonical_json(payload)
    sig = ML_DSA_65.sign(sk, msg)
    import base64

    sig_b64 = base64.b64encode(sig).decode()

    envelope = {
        "payload": payload,
        "signature": {"alg": "ML-DSA-65", "kid": kid, "sig": sig_b64},
        "anchors": [],
    }
    jwks = {
        "keys": [
            {
                "kid": kid,
                "issuer_id": kid,
                "alg": "ML-DSA-65",
                "status": "active",
                "public_key": pk_b64,
            }
        ]
    }
    return envelope, jwks


try:
    from dilithium_py.ml_dsa import ML_DSA_65 as _ML_DSA_65_CHECK  # noqa: F401

    _DILITHIUM_AVAILABLE = True
except ImportError:
    _DILITHIUM_AVAILABLE = False


@pytest.mark.skipif(
    not _DILITHIUM_AVAILABLE,
    reason="dilithium-py not installed (pip install asqav[verify])",
)
def test_verify_receipt_offline_mldsa65_pass():
    """ML-DSA-65 signed receipt verifies PASS when dilithium-py is present."""
    envelope, jwks = _make_mldsa_receipt_and_jwks()
    result = asqav.verify_receipt_offline(envelope, jwks)
    assert result["verdict"] == "PASS", f"axes: {result['axes']}"
    sig_axis = next(a for a in result["axes"] if a["name"] == "signature")
    assert sig_axis["result"] == "PASS"


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_verify_receipt_offline_mldsa65_tampered_fails():
    """A tampered ML-DSA-65 receipt must return FAIL, never PASS."""
    envelope, jwks = _make_mldsa_receipt_and_jwks()
    envelope["payload"]["decision"] = "deny"
    result = asqav.verify_receipt_offline(envelope, jwks)
    assert result["verdict"] == "FAIL"
    sig_axis = next(a for a in result["axes"] if a["name"] == "signature")
    assert sig_axis["result"] == "FAIL"


# Real-cloud ML-DSA-65 payload-mode known-answer test (asqav-06-mldsa65-payload-prod).
# Uses a receipt + JWKS minted from api.asqav.com with mode=full-payload.
# The ML-DSA-65 signature axis must be PASS, not SKIPPED or INCOMPLETE.
MLDSA_KAT_VECTOR = VECTORS / "asqav-06-mldsa65-payload-prod"


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_verify_receipt_offline_mldsa65_real_cloud_kat():
    """Real-cloud ML-DSA-65 payload-mode KAT: signature axis must be PASS."""
    receipt = _load(MLDSA_KAT_VECTOR / "receipt.json")
    jwks = _load(MLDSA_KAT_VECTOR / "jwks.json")
    result = asqav.verify_receipt_offline(receipt, jwks)
    assert (
        result["verdict"] == "PASS"
    ), f"Expected PASS, got {result['verdict']}; axes: {result['axes']}"
    sig_axis = next(a for a in result["axes"] if a["name"] == "signature")
    # Must be PASS - SKIPPED or INCOMPLETE means the ML-DSA-65 check was bypassed.
    assert (
        sig_axis["result"] == "PASS"
    ), f"ML-DSA-65 signature axis must be PASS (not SKIPPED/INCOMPLETE/FAIL); got: {sig_axis}"


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_verify_receipt_offline_mldsa65_real_cloud_kat_tamper_payload():
    """Tampered KAT payload byte must return FAIL - anti-vacuous guard."""
    receipt = _load(MLDSA_KAT_VECTOR / "receipt.json")
    jwks = _load(MLDSA_KAT_VECTOR / "jwks.json")
    tampered = copy.deepcopy(receipt)
    # Flip one payload field - the ML-DSA sig over canonical bytes must not match.
    tampered["payload"]["decision"] = "deny"
    result = asqav.verify_receipt_offline(tampered, jwks)
    assert (
        result["verdict"] == "FAIL"
    ), f"Tampered receipt must be FAIL, got {result['verdict']}; axes: {result['axes']}"
    sig_axis = next((a for a in result["axes"] if a["name"] == "signature"), None)
    assert sig_axis is not None
    assert (
        sig_axis["result"] == "FAIL"
    ), f"Signature axis on tampered receipt must be FAIL, got: {sig_axis}"


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_verify_receipt_offline_mldsa65_real_cloud_kat_tamper_sig():
    """Swapped-sig KAT receipt must return FAIL - anti-vacuous guard."""
    receipt = _load(MLDSA_KAT_VECTOR / "receipt.json")
    jwks = _load(MLDSA_KAT_VECTOR / "jwks.json")
    tampered = copy.deepcopy(receipt)
    # Zero-out the first bytes of the signature - will not verify.
    import base64

    raw_sig = base64.b64decode(
        tampered["signature"]["sig"].replace("-", "+").replace("_", "/") + "=="
    )
    bad_sig = bytes([0] * 16) + raw_sig[16:]
    tampered["signature"]["sig"] = base64.b64encode(bad_sig).decode()
    result = asqav.verify_receipt_offline(tampered, jwks)
    assert (
        result["verdict"] == "FAIL"
    ), f"Corrupted-sig receipt must be FAIL, got {result['verdict']}; axes: {result['axes']}"
    sig_axis = next((a for a in result["axes"] if a["name"] == "signature"), None)
    assert sig_axis is not None
    assert sig_axis["result"] == "FAIL"


# ---------------------------------------------------------------------------
# run_structured is also available directly on the verifier module.
# run_structured uses the standalone verify_receipt engine (ML-DSA-65 only).
# The Ed25519 vector has an unsupported alg for the standalone engine, so it
# returns INCOMPLETE (not PASS); that is correct and expected behaviour.
# ---------------------------------------------------------------------------


def test_run_structured_directly():
    """run_structured() is callable from the verifier module and returns a dict."""
    receipt = _load(PASS_VECTOR / "receipt.json")
    jwks = _load(PASS_VECTOR / "jwks.json")
    result = vr.run_structured(receipt, jwks)
    # The standalone engine only verifies ML-DSA-65; Ed25519 is SKIPPED -> INCOMPLETE.
    assert result["verdict"] in ("PASS", "INCOMPLETE")
    assert isinstance(result["axes"], list)
    assert result["canonical_sha256"] is not None


# criterion 446: a malformed JWKS must not crash the public offline API. Each
# reached verify_receipt.resolve_key via asqav_native.py and raised before the guard.


def test_offline_api_missing_public_key_no_crash():
    """A JWK with a matching kid but no public_key field resolves as a failure,
    not a KeyError out of the public API."""
    receipt = _load(PASS_VECTOR / "receipt.json")
    kid = receipt["signature"]["kid"]
    jwks = {"keys": [{"kid": kid, "kty": "OKP", "crv": "Ed25519", "x": "abc"}]}
    result = asqav.verify_receipt_offline(receipt, jwks)
    assert result["verdict"] in ("FAIL", "INCOMPLETE")
    sig_axis = next(a for a in result["axes"] if a["name"] == "signature")
    assert sig_axis["result"] in ("FAIL", "SKIPPED")


def test_offline_api_keys_is_string_no_crash():
    receipt = _load(PASS_VECTOR / "receipt.json")
    result = asqav.verify_receipt_offline(receipt, {"keys": "abc"})
    assert isinstance(result["verdict"], str)


def test_offline_api_keys_is_list_of_ints_no_crash():
    receipt = _load(PASS_VECTOR / "receipt.json")
    result = asqav.verify_receipt_offline(receipt, {"keys": [123]})
    assert isinstance(result["verdict"], str)
