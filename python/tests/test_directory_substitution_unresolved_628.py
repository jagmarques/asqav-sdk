"""Directory substitution (628 T-a): a retained directory pins the signing key.

The receipt's signed key_thumbprint names K1. The current directory publishes
only K2 under the same kid/agent (K1 removed, K2 substituted). With the prior
directory supplied as retained evidence, key_binding must report an explicit
directory_substitution unresolved result instead of a bare binding failure.
Without retained evidence the legacy verdict is unchanged (T2 safety pin).
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
    old_pk, old_sk = ml.keygen()
    new_pk, _ = ml.keygen()
    thumb = v.thumbprint_for_key(alg="ML-DSA-65", public_key=old_pk)
    payload = _payload("agt_two", thumb)
    sig = ml.sign(old_sk, v.canonical_json(payload))
    env = _envelope(payload, sig)
    current = {"keys": [_key("k-same", "agt_two", new_pk)]}
    prior = {"keys": [_key("k-same", "agt_two", old_pk)]}
    return env, current, prior


def test_substituted_directory_reports_unresolved() -> None:
    """Retained evidence pins K1; the current directory binds K2 to the same
    kid/agent. key_binding reports directory_substitution as SKIPPED."""
    env, current, prior = _scenario()
    report = v.run_structured(env, current, None, prior_jwks=prior)
    axes = _axes(report)
    assert axes["key_binding"][0] == "SKIPPED", axes["key_binding"]
    assert "directory_substitution" in axes["key_binding"][1]
    assert report["verdict"] == v.VERDICT_UNVERIFIED


def test_same_scenario_without_prior_evidence_keeps_legacy_verdict() -> None:
    """No retained evidence: legacy path unchanged, no substitution report."""
    env, current, _ = _scenario()
    report = v.run_structured(env, current, None)
    axes = _axes(report)
    assert "directory_substitution" not in axes["key_binding"][1]
    assert report["verdict"] == v.VERDICT_UNVERIFIED
