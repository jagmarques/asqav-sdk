"""Rotation succession (628 T-b): a verifying key the retained directory never authorized.

K2 signs and the current directory publishes K2, so signature and binding
rederive. The retained directory pins K1 for the same kid/agent with no
transition authorization to K2. With retained evidence supplied, issuer_key
must report an explicit forged_transition unresolved result. Without it the
legacy verdict is unchanged (T2 safety pin).
"""

from __future__ import annotations

import base64

import pytest

from asqav.verifier import verify_receipt as v

ORG = "f94f66c0-c580-432d-a041-29374f7aee07"


def _ml_dsa_65():
    return pytest.importorskip("dilithium_py.ml_dsa").ML_DSA_65


def _key(kid: str, agent_id: str, pk: bytes) -> dict:
    return {
        "kid": kid,
        "agent_id": agent_id,
        "issuer_id": ORG,
        "org_id": ORG,
        "alg": "ML-DSA-65",
        "kty": "AKP",
        "public_key": base64.b64encode(pk).decode(),
        "status": "active",
        "revoked_at": None,
        "key_thumbprint": v.thumbprint_for_key(alg="ML-DSA-65", public_key=pk),
    }


def _payload(agent_id: str, thumbprint: str) -> dict:
    return {
        "type": "protectmcp:decision",
        "issued_at": "2026-06-19T00:00:00+00:00",
        "issuer_id": ORG,
        "agent_id": agent_id,
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
        "key_thumbprint": thumbprint,
    }


def _envelope(payload: dict, sig: bytes) -> dict:
    return {
        "payload": payload,
        "signature": {"alg": "ML-DSA-65", "kid": ORG,
                      "sig": base64.b64encode(sig).decode()},
        "anchors": [],
    }


def _axes(report: dict) -> dict[str, tuple[str, str]]:
    return {a["name"]: (a["result"], a["note"]) for a in report["axes"]}


def _scenario():
    ml = _ml_dsa_65()
    old_pk, _ = ml.keygen()
    new_pk, new_sk = ml.keygen()
    thumb = v.thumbprint_for_key(alg="ML-DSA-65", public_key=new_pk)
    payload = _payload("agt_two", thumb)
    sig = ml.sign(new_sk, v.canonical_json(payload))
    env = _envelope(payload, sig)
    current = {"keys": [_key("k-same", "agt_two", new_pk)]}
    prior = {"keys": [_key("k-same", "agt_two", old_pk)]}
    return env, current, prior


def test_unevidenced_rotation_reports_unresolved() -> None:
    """K2 verifies and binds, but retained evidence pins K1 with no
    transition authorization. issuer_key reports forged_transition."""
    env, current, prior = _scenario()
    report = v.run_structured(env, current, None, prior_jwks=prior)
    axes = _axes(report)
    assert axes["issuer_key"][0] == "SKIPPED", axes["issuer_key"]
    assert "forged_transition" in axes["issuer_key"][1]
    assert axes["signature"][0] == "PASS", axes["signature"]
    assert report["verdict"] == v.VERDICT_UNVERIFIED


def test_same_rotation_without_prior_evidence_keeps_legacy_verdict() -> None:
    """No retained evidence: issuer_key still resolves K2 cleanly."""
    env, current, _ = _scenario()
    report = v.run_structured(env, current, None)
    axes = _axes(report)
    assert axes["issuer_key"][0] == "PASS", axes["issuer_key"]
    assert "forged_transition" not in axes["issuer_key"][1]
