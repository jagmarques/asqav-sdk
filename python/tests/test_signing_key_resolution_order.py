"""The standalone verifier resolves the signing key by what the receipt signed, not list position.

Criterion 564. The envelope kid of a cloud receipt is the org id, which every sibling key of the
org publishes as issuer_id, so a loose match answers with whichever sibling is listed first. The
signed key_thumbprint and the signed agent_id name the real key; the matcher reads them first.
Every directory here publishes more than one key under one issuer, the shape that exposes a
matcher which returns whatever it reaches first.
"""

from __future__ import annotations

import base64

import pytest

from asqav.verifier import verify_receipt as v
from asqav.verifier.oracle.adapters.asqav_native import AsqavNativeAdapter
from asqav.verifier.oracle.core import verify as oracle_verify

ORG = "f94f66c0-c580-432d-a041-29374f7aee07"
OTHER_ORG = "0b6c2b1e-9f7a-4d3c-8a11-5c2e7d904f38"


def _ml_dsa_65():
    return pytest.importorskip("dilithium_py.ml_dsa").ML_DSA_65


def _key(kid: str, agent_id: str, pk: bytes, *, issuer_id: str = ORG, status: str = "active",
         thumbprint: bool = True) -> dict:
    entry = {
        "kid": kid,
        "agent_id": agent_id,
        "issuer_id": issuer_id,
        "org_id": issuer_id,
        "alg": "ML-DSA-65",
        "kty": "AKP",
        "public_key": base64.b64encode(pk).decode(),
        "status": status,
        "revoked_at": "2026-07-11T14:57:49Z" if status == "revoked" else None,
    }
    if thumbprint:
        entry["key_thumbprint"] = v.thumbprint_for_key(alg="ML-DSA-65", public_key=pk)
    return entry


def _payload(agent_id: str = "agt_two", issuer_id: str = ORG, thumbprint: str | None = None) -> dict:
    p = {
        "type": "protectmcp:decision",
        "issued_at": "2026-06-19T00:00:00+00:00",
        "issuer_id": issuer_id,
        "agent_id": agent_id,
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
    }
    if thumbprint is not None:
        p["key_thumbprint"] = thumbprint
    return p


def _envelope(payload: dict, sig: bytes, alg: str = "ML-DSA-65", kid: str = ORG) -> dict:
    return {
        "payload": payload,
        "signature": {"alg": alg, "kid": kid, "sig": base64.b64encode(sig).decode()},
        "anchors": [],
    }


def _axes(report: dict) -> dict[str, tuple[str, str]]:
    return {a["name"]: (a["result"], a["note"]) for a in report["axes"]}


def test_unsupported_alg_resolves_the_named_agent_key_not_the_revoked_sibling() -> None:
    """Signature SKIPPED (an algorithm outside the profile), so no failure ever triggered the
    old agent-bind fallback and the revoked sibling's status was reported. Mirrors the live
    canary of 2026-09-02."""
    ml44 = pytest.importorskip("dilithium_py.ml_dsa").ML_DSA_44
    old_pk, _ = ml44.keygen()
    two_pk, two_sk = ml44.keygen()
    payload = _payload("agt_two")
    sig = ml44.sign(two_sk, v.canonical_json(payload))
    old = _key("k-old", "agt_one", old_pk, status="revoked", thumbprint=False)
    two = _key("k-two", "agt_two", two_pk, thumbprint=False)
    old["alg"] = two["alg"] = "ML-DSA-44"
    jwks = {"keys": [old, two]}
    report = v.run_structured(_envelope(payload, sig, alg="ML-DSA-44"), jwks, None)
    axes = _axes(report)
    assert axes["signature"][0] == "SKIPPED"
    assert axes["key_status"][0] == "PASS", axes["key_status"]
    assert axes["issuer_bind"][0] == "PASS"


def test_rotated_agent_resolves_by_signed_thumbprint_across_two_published_keys() -> None:
    """Two active keys for one agent, the stale one listed first; the thumbprint picks the signer."""
    ml = _ml_dsa_65()
    stale_pk, _ = ml.keygen()
    new_pk, new_sk = ml.keygen()
    thumb = v.thumbprint_for_key(alg="ML-DSA-65", public_key=new_pk)
    payload = _payload("agt_two", thumbprint=thumb)
    sig = ml.sign(new_sk, v.canonical_json(payload))
    jwks = {"keys": [_key("k-stale", "agt_two", stale_pk), _key("k-new", "agt_two", new_pk)]}
    report = v.run_structured(_envelope(payload, sig), jwks, None)
    axes = _axes(report)
    assert axes["signature"][0] == "PASS", axes["signature"]
    assert axes["key_binding"][0] == "PASS", axes["key_binding"]
    assert axes["key_status"][0] == "PASS"


def test_rotated_agent_without_a_thumbprint_still_lands_on_the_first_key() -> None:
    """The documented remaining limit: without the signed thumbprint the agent bind picks by position."""
    ml = _ml_dsa_65()
    stale_pk, _ = ml.keygen()
    new_pk, new_sk = ml.keygen()
    payload = _payload("agt_two")
    sig = ml.sign(new_sk, v.canonical_json(payload))
    jwks = {"keys": [_key("k-stale", "agt_two", stale_pk), _key("k-new", "agt_two", new_pk)]}
    report = v.run_structured(_envelope(payload, sig), jwks, None)
    assert _axes(report)["signature"][0] == "FAIL"


def test_thumbprint_naming_an_unused_key_is_reported_as_substitution() -> None:
    """A thumbprint for key D while key C signed: C verifies, D was bound, the binding breaks."""
    ml = _ml_dsa_65()
    c_pk, c_sk = ml.keygen()
    d_pk, _ = ml.keygen()
    payload = _payload("agt_two", thumbprint=v.thumbprint_for_key(alg="ML-DSA-65", public_key=d_pk))
    sig = ml.sign(c_sk, v.canonical_json(payload))
    jwks = {"keys": [_key("k-c", "agt_two", c_pk), _key("k-d", "agt_two", d_pk)]}
    report = v.run_structured(_envelope(payload, sig), jwks, None)
    axes = _axes(report)
    assert axes["signature"][0] == "PASS"
    assert axes["key_binding"][0] == "FAIL" and "key_substituted" in axes["key_binding"][1]
    assert report["verdict"] == v.VERDICT_UNVERIFIED
    assert report["failure_class"] == v.FAILURE_INVALID


def test_thumbprint_of_a_foreign_org_key_cannot_pass_the_issuer_bind() -> None:
    """An attacker binding their own published key's thumbprint verifies the signature and fails the bind."""
    ml = _ml_dsa_65()
    attacker_pk, attacker_sk = ml.keygen()
    victim_pk, _ = ml.keygen()
    payload = _payload("agt_attacker", issuer_id=ORG,
                       thumbprint=v.thumbprint_for_key(alg="ML-DSA-65", public_key=attacker_pk))
    sig = ml.sign(attacker_sk, v.canonical_json(payload))
    jwks = {"keys": [_key("k-victim", "agt_victim", victim_pk),
                     _key("k-attacker", "agt_attacker", attacker_pk, issuer_id=OTHER_ORG)]}
    report = v.run_structured(_envelope(payload, sig), jwks, None)
    axes = _axes(report)
    assert axes["issuer_bind"][0] == "FAIL", axes["issuer_bind"]
    assert report["verdict"] == v.VERDICT_UNVERIFIED


def test_match_signing_key_order_is_thumbprint_kid_agent_then_issuer() -> None:
    ml = _ml_dsa_65()
    a_pk, _ = ml.keygen()
    b_pk, _ = ml.keygen()
    a, b = _key("k-a", "agt_a", a_pk), _key("k-b", "agt_b", b_pk)
    jwks = {"keys": [a, b]}
    thumb_b = b["key_thumbprint"]
    # The signed thumbprint outranks an exact kid that names another key.
    assert v.match_signing_key(jwks, "k-a", "agt_a", ORG, ORG, thumb_b) is b
    assert v.match_signing_key(jwks, "k-a", "agt_b", ORG, ORG) is a
    assert v.match_signing_key(jwks, ORG, "agt_b", ORG, ORG) is b
    assert v.match_signing_key(jwks, ORG, None, ORG, ORG) is a
    # A malformed thumbprint never matches and never crashes.
    assert v.match_signing_key(jwks, ORG, "agt_b", ORG, ORG, "sha256:nope") is b


def test_oracle_adapter_agrees_with_the_standalone_on_the_rotated_agent() -> None:
    ml = _ml_dsa_65()
    stale_pk, _ = ml.keygen()
    new_pk, new_sk = ml.keygen()
    thumb = v.thumbprint_for_key(alg="ML-DSA-65", public_key=new_pk)
    payload = _payload("agt_two", thumbprint=thumb)
    sig = ml.sign(new_sk, v.canonical_json(payload))
    jwks = {"keys": [_key("k-stale", "agt_two", stale_pk), _key("k-new", "agt_two", new_pk)]}
    res = oracle_verify(_envelope(payload, sig), [AsqavNativeAdapter()], jwks)
    by_name = {a.axis: a.result for a in res.axes}
    assert by_name["signature"] == "PASS", res.axes
