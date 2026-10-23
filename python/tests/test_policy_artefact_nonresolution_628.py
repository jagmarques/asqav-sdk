"""Policy artefacts never resolve (628 T-d): the digest is unevaluated.

Two receipts from one keypair differ only in policy_digest; both reports carry
the same axes and results, with no policy-resolution claim. A new policy root
does not retro-authorize prior material; full replacement needs a wire field.
"""

from __future__ import annotations

import base64

import pytest

from asqav.verifier import verify_receipt as v

ORG = "f94f66c0-c580-432d-a041-29374f7aee07"

EXPECTED_AXES = [
    "structure", "nonce", "issuer_key", "issuer_bind", "key_status",
    "signature", "key_binding", "counterparty", "payload_digest",
    "outer_copies", "chain", "anchors", "skew", "expiry",
]


def _ml_dsa_65():
    return pytest.importorskip("dilithium_py.ml_dsa").ML_DSA_65


def _report(pk: bytes, sk: bytes, thumbprint: str, policy_digest: str) -> dict:
    ml = _ml_dsa_65()
    payload = {
        "type": "protectmcp:decision",
        "issued_at": "2026-06-19T00:00:00+00:00",
        "issuer_id": ORG,
        "agent_id": "agt_two",
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": policy_digest,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
        "key_thumbprint": thumbprint,
    }
    sig = ml.sign(sk, v.canonical_json(payload))
    env = {
        "payload": payload,
        "signature": {"alg": "ML-DSA-65", "kid": ORG,
                      "sig": base64.b64encode(sig).decode()},
        "anchors": [],
    }
    key = {
        "kid": ORG,
        "agent_id": "agt_two",
        "issuer_id": ORG,
        "org_id": ORG,
        "alg": "ML-DSA-65",
        "kty": "AKP",
        "public_key": base64.b64encode(pk).decode(),
        "status": "active",
        "revoked_at": None,
        "key_thumbprint": thumbprint,
    }
    return v.run_structured(env, {"keys": [key]}, None)


def test_policy_digest_never_affects_evaluation() -> None:
    """One keypair, two policy digests: same axis shape and results, and
    the only "policy" word in any note is signature's static legacy
    encoding-policy note."""
    ml = _ml_dsa_65()
    pk, sk = ml.keygen()
    thumb = v.thumbprint_for_key(alg="ML-DSA-65", public_key=pk)
    rx = _report(pk, sk, thumb, "sha256:" + "3" * 64)
    ry = _report(pk, sk, thumb, "sha256:" + "4" * 64)
    assert [a["name"] for a in rx["axes"]] == EXPECTED_AXES
    assert [a["name"] for a in ry["axes"]] == EXPECTED_AXES
    assert not any(a["name"].startswith("policy") for a in rx["axes"])
    assert [(a["name"], a["result"]) for a in rx["axes"]] == [
        (a["name"], a["result"]) for a in ry["axes"]]
    assert rx["verdict"] == ry["verdict"]
    for report in (rx, ry):
        hits = [(a["name"], a["note"]) for a in report["axes"]
                if "polic" in a["note"].lower()]
        assert len(hits) == 1, hits
        assert hits[0][0] == "signature"
        assert "encoding policy" in hits[0][1]
