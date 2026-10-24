"""Removed old key (628 T-c): the retained key the rotation replaces is revoked.

Same shape as T-b (K2 verifies and binds, retained evidence pins K1 for the
same kid/agent, no transition authorization), except the retained K1 carries
revoked status. The report must name removed_old_key rather than the generic
forged_transition: a revoked key cannot authorize succession. Active-status
retained keys keep the T-b tag (pinned by its test).
"""

from __future__ import annotations

import base64

import pytest

from asqav.verifier import verify_receipt as v

ORG = "f94f66c0-c580-432d-a041-29374f7aee07"


def _ml_dsa_65():
    return pytest.importorskip("dilithium_py.ml_dsa").ML_DSA_65


def _key(kid: str, agent_id: str, pk: bytes, status: str = "active") -> dict:
    return {
        "kid": kid,
        "agent_id": agent_id,
        "issuer_id": ORG,
        "org_id": ORG,
        "alg": "ML-DSA-65",
        "kty": "AKP",
        "public_key": base64.b64encode(pk).decode(),
        "status": status,
        "revoked_at": "2026-07-11T14:57:49Z" if status == "revoked" else None,
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


def test_revoked_retained_key_names_removed_old_key() -> None:
    """Retained K1 is revoked; K2 verifies with no transition proof.
    issuer_key reports removed_old_key, not the generic transition tag."""
    ml = _ml_dsa_65()
    old_pk, _ = ml.keygen()
    new_pk, new_sk = ml.keygen()
    thumb = v.thumbprint_for_key(alg="ML-DSA-65", public_key=new_pk)
    payload = _payload("agt_two", thumb)
    sig = ml.sign(new_sk, v.canonical_json(payload))
    env = _envelope(payload, sig)
    current = {"keys": [_key("k-same", "agt_two", new_pk)]}
    prior = {"keys": [_key("k-same", "agt_two", old_pk, status="revoked")]}
    report = v.run_structured(env, current, None, prior_jwks=prior)
    axes = _axes(report)
    assert axes["issuer_key"][0] == "SKIPPED", axes["issuer_key"]
    assert "removed_old_key" in axes["issuer_key"][1]
    assert report["verdict"] == v.VERDICT_UNVERIFIED
