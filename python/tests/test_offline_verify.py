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


def _jwks_key(kid: str, issuer_id: str, pub) -> dict:
    import base64

    return {
        "kid": kid,
        "agent_id": "agt_attacker",
        "issuer_id": issuer_id,
        "alg": "ML-DSA-65",
        "status": "active",
        "public_key": base64.b64encode(pub).decode(),
    }


def _cross_issuer_forgery_and_jwks():
    """Attacker-signed receipt claiming the victim's issuer_id, plus the shared
    JWKS publishing both orgs' keys. Returns (envelope, jwks)."""
    import base64

    from dilithium_py.ml_dsa import ML_DSA_65

    attacker_pk, attacker_sk = ML_DSA_65.keygen()
    victim_pk, _victim_sk = ML_DSA_65.keygen()
    envelope, _ = _make_mldsa_receipt_and_jwks()
    payload = envelope["payload"]
    payload["issuer_id"], payload["agent_id"] = "org-victim", "agt_attacker"
    sig = ML_DSA_65.sign(attacker_sk, vr.canonical_json(payload))
    # kid points at the attacker's OWN published key, not the victim's.
    envelope["signature"] = {
        "alg": "ML-DSA-65",
        "kid": "attacker-key",
        "sig": base64.b64encode(sig).decode(),
    }
    jwks = {
        "keys": [
            _jwks_key("victim-key", "org-victim", victim_pk),
            _jwks_key("attacker-key", "org-attacker", attacker_pk),
        ]
    }
    return envelope, jwks


# Production org ids are organization UUIDs, so the fixtures use that shape.
_VICTIM_ORG = "f94f66c0-c580-432d-a041-29374f7aee07"
_ATTACKER_ORG = "0b6c2b1e-9f7a-4d3c-8a11-5c2e7d904f38"

_HASH_SIGNED_FIELDS = (
    "v", "mode", "hash", "hash_algo", "metadata", "server_timestamp",
    "action_id", "agent_id", "org_id", "policy_digest", "policy_decision",
)


def _hash_mode_receipt(org_id: str, key_id: str, secret_key):
    """A real hash-mode /sign receipt: the 11 signed fields plus the wire fields.
    Signed over the bytes the cloud signs, so only the bind weighs the org."""
    import base64

    from dilithium_py.ml_dsa import ML_DSA_65

    from asqav.verifier.oracle.canonical import asqav_jcs

    doc = {
        "v": 1,
        "mode": "hash",
        "hash": "sha256:" + "a" * 64,
        "hash_algo": "sha256",
        "metadata": {},
        "server_timestamp": "2026-06-19T00:00:00.000000Z",
        "action_id": "act_probe",
        "agent_id": "agt_attacker",
        "org_id": org_id,
        "policy_digest": "sha256:" + "c" * 64,
        "policy_decision": "allow",
    }
    signed = {f: doc[f] for f in _HASH_SIGNED_FIELDS}
    doc["key_id"] = key_id
    doc["algorithm"] = "ML-DSA-65"
    # Real /sign output carries an explicit null payload and an anchors list.
    # Omitting them built a fixture shape production never emits.
    doc["payload"] = None
    doc["anchors"] = []
    doc["signature_b64"] = base64.b64encode(
        ML_DSA_65.sign(secret_key, asqav_jcs(signed))
    ).decode()
    return doc


def test_hash_mode_fixture_matches_the_prod_receipt_shape():
    """Fixture practice: a hand-built receipt must carry the prod field set, so a
    test can never pass on a shape production does not emit."""
    from dilithium_py.ml_dsa import ML_DSA_65

    real = set(_load(VECTORS / "asqav-05-hash-mode-prod" / "receipt.json"))
    built = set(_hash_mode_receipt("org-x", "kid-x", ML_DSA_65.keygen()[1]))
    assert real - built == set(), f"fixture omits prod fields: {sorted(real - built)}"


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_verify_receipt_offline_rejects_hash_mode_claiming_another_org():
    """FORGERY in hash mode, the default /sign shape: org_id is a required signed
    field, so a receipt claiming the victim's org must be refused."""
    _envelope_unused, jwks = _cross_issuer_forgery_and_jwks()
    from dilithium_py.ml_dsa import ML_DSA_65

    import base64

    attacker_pk, attacker_sk = ML_DSA_65.keygen()
    jwks = {
        "keys": [
            {
                "kid": "attacker-key",
                "agent_id": "agt_attacker",
                "issuer_id": _ATTACKER_ORG,
                "alg": "ML-DSA-65",
                "status": "active",
                "public_key": base64.b64encode(attacker_pk).decode(),
            }
        ]
    }
    doc = _hash_mode_receipt(_VICTIM_ORG, "attacker-key", attacker_sk)
    result = asqav.verify_receipt_offline(doc, jwks)
    assert result["verdict"] == "FAIL", f"axes: {result['axes']}"
    bind = next(a for a in result["axes"] if a["name"] == "issuer_bind")
    assert bind["result"] == "FAIL", f"issuer_bind axis: {bind}"
    sig_axis = next(a for a in result["axes"] if a["name"] == "signature")
    assert sig_axis["result"] == "PASS", "the hash-mode signature verifies; the bind refuses it"


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_verify_receipt_offline_hash_mode_cannot_shed_axes_via_unsigned_fields():
    """A hash-mode receipt that also displays issuer_id or previousReceiptHash is
    presenting claims its signature does not cover, so it is refused."""
    import base64

    from dilithium_py.ml_dsa import ML_DSA_65

    attacker_pk, attacker_sk = ML_DSA_65.keygen()
    jwks = {
        "keys": [
            {
                "kid": "attacker-key",
                "agent_id": "agt_attacker",
                "issuer_id": _ATTACKER_ORG,
                "alg": "ML-DSA-65",
                "status": "active",
                "public_key": base64.b64encode(attacker_pk).decode(),
            }
        ]
    }
    clean = _hash_mode_receipt(_ATTACKER_ORG, "attacker-key", attacker_sk)
    # Control: the honest hash-mode receipt still verifies.
    assert asqav.verify_receipt_offline(clean, jwks)["verdict"] == "PASS"

    doc = dict(clean)
    # Unsigned in hash mode, so a consumer reading these fields sees a victim claim.
    doc["issuer_id"] = "org-victim"
    doc["previousReceiptHash"] = "0" * 64
    result = asqav.verify_receipt_offline(doc, jwks)
    assert result["verdict"] == "FAIL", f"axes: {result['axes']}"
    # The compliance rule set applied, so the doc is judged on the claim it displays.
    structure = next(a for a in result["axes"] if a["name"] == "structure")
    assert structure["result"] == "FAIL", f"structure axis: {structure}"


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_verify_receipt_offline_rejects_a_key_from_another_issuer():
    """FORGERY through the public offline API: the attacker signs with their own
    published ML-DSA-65 key, points signature.kid at it, and claims the victim's
    issuer_id. The signature is genuine, so only the issuer bind can refuse it."""
    envelope, jwks = _cross_issuer_forgery_and_jwks()
    result = asqav.verify_receipt_offline(envelope, jwks)
    assert result["verdict"] == "FAIL", f"axes: {result['axes']}"
    bind = next(a for a in result["axes"] if a["name"] == "issuer_bind")
    assert bind["result"] == "FAIL", f"issuer_bind axis: {bind}"
    sig_axis = next(a for a in result["axes"] if a["name"] == "signature")
    assert sig_axis["result"] == "PASS", "the forged signature verifies; the bind is what refuses it"


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


def _hash_mode_jwks(issuer_id: str, pub, org_id: str | None = None) -> dict:
    key = _jwks_key("attacker-key", issuer_id, pub)
    if org_id is not None:
        key["org_id"] = org_id
    return {"keys": [key]}


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_hash_mode_org_with_a_legal_entity_does_not_false_fail():
    """issuer_id is legal_entity or org.id while org_id is always org.id, so they
    diverge once an org sets a legal entity. The honest receipt must not FAIL, and
    verifies once the directory publishes org_id per key."""
    from dilithium_py.ml_dsa import ML_DSA_65

    pk, sk = ML_DSA_65.keygen()
    doc = _hash_mode_receipt(_ATTACKER_ORG, "attacker-key", sk)

    aliased = asqav.verify_receipt_offline(doc, _hash_mode_jwks("Acme Compliance Ltd", pk))
    bind = next(a for a in aliased["axes"] if a["name"] == "issuer_bind")
    assert bind["result"] == "SKIPPED", f"honest receipt false-FAILed: {bind}"
    assert aliased["verdict"] == "INCOMPLETE", f"axes: {aliased['axes']}"

    published = asqav.verify_receipt_offline(
        doc, _hash_mode_jwks("Acme Compliance Ltd", pk, org_id=_ATTACKER_ORG)
    )
    assert published["verdict"] == "PASS", f"axes: {published['axes']}"


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_hash_mode_cross_org_still_fails_when_the_directory_names_an_org():
    """The legal-entity allowance must not soften the forgery: when the directory
    names a real org for the key, a receipt claiming a different one FAILs."""
    from dilithium_py.ml_dsa import ML_DSA_65

    pk, sk = ML_DSA_65.keygen()
    doc = _hash_mode_receipt(_VICTIM_ORG, "attacker-key", sk)
    for jwks in (
        _hash_mode_jwks(_ATTACKER_ORG, pk),
        _hash_mode_jwks("Acme Compliance Ltd", pk, org_id=_ATTACKER_ORG),
    ):
        result = asqav.verify_receipt_offline(doc, jwks)
        assert result["verdict"] == "FAIL", f"cross-org forgery survived: {result['axes']}"


def test_unmodified_prod_vector_with_unsigned_claims_does_not_crash():
    """Regression guard: the real prod receipt plus unsigned claim fields gives a
    clean FAIL, never a TypeError out of the compliance path."""
    V = VECTORS / "asqav-05-hash-mode-prod"
    doc = _load(V / "receipt.json")
    doc["issuer_id"] = "org-victim"
    doc["previousReceiptHash"] = "0" * 64
    result = asqav.verify_receipt_offline(doc, _load(V / "jwks.json"))
    assert result["verdict"] == "FAIL", f"axes: {result['axes']}"
    structure = next(a for a in result["axes"] if a["name"] == "structure")
    assert "does not cover" in structure["note"], structure
