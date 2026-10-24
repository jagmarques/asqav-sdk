"""Key resolution: a thumbprint tie is broken by the signed identifiers, and the agent bind verifies.

Offline: every key pair and directory row is minted here, so a green run says nothing about
api.asqav.com and everything about the resolver.
"""

from __future__ import annotations

import base64

import pytest

from asqav.verifier import verify_receipt as vr

ORG = "org-A"
AGENT = "agt-A"


def _ml():
    pytest.importorskip("dilithium_py")
    from dilithium_py.ml_dsa import ML_DSA_65

    return ML_DSA_65


def _payload(**extra) -> dict:
    payload = {
        "type": "protectmcp:decision",
        "issued_at": "2026-09-01T00:00:00+00:00",
        "issuer_id": ORG,
        "agent_id": AGENT,
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
    }
    payload.update(extra)
    return payload


def _row(pk: bytes, kid: str, *, status: str = "active", revoked_at=None, thumbprint=None) -> dict:
    row = {
        "kid": kid,
        "agent_id": AGENT,
        "issuer_id": ORG,
        "org_id": ORG,
        "alg": "ML-DSA-65",
        "kty": "AKP",
        "public_key": base64.b64encode(pk).decode(),
        "status": status,
        "revoked_at": revoked_at,
    }
    if thumbprint is not None:
        row["key_thumbprint"] = thumbprint
    return row


def _envelope(payload: dict, sk: bytes, ml, kid: str) -> dict:
    sig = ml.sign(sk, vr.canonical_json(payload))
    return {
        "payload": payload,
        "signature": {"alg": "ML-DSA-65", "kid": kid, "sig": base64.b64encode(sig).decode()},
        "anchors": [],
    }


def _axes(report: dict) -> dict:
    return {a["name"]: a for a in report["axes"]}


# === a thumbprint shared by several rows is narrowed by the signed identifiers (448a) ===


def test_a_kid_naming_the_revoked_duplicate_reaches_key_status_with_that_row() -> None:
    ml = _ml()
    pk, sk = ml.keygen()
    thumb = vr.thumbprint_for_key(alg="ML-DSA-65", public_key=pk)
    # Two rows publishing one key: the active row first, the revoked row second.
    jwks = {
        "keys": [
            _row(pk, "k-A1", thumbprint=thumb),
            _row(pk, "k-A1-old", status="revoked", revoked_at="2026-01-01T00:00:00+00:00",
                 thumbprint=thumb),
        ]
    }
    payload = _payload(key_thumbprint=thumb)
    report = vr.run_structured(_envelope(payload, sk, ml, "k-A1-old"), jwks, None)
    axes = _axes(report)
    assert axes["signature"]["result"] == "PASS", axes["signature"]
    assert axes["key_status"]["result"] != "PASS", axes["key_status"]
    assert "revoked" in axes["key_status"]["note"]
    assert "k-A1-old" in axes["issuer_key"]["note"]


def test_a_kid_naming_the_active_duplicate_still_reads_active() -> None:
    ml = _ml()
    pk, sk = ml.keygen()
    thumb = vr.thumbprint_for_key(alg="ML-DSA-65", public_key=pk)
    jwks = {
        "keys": [
            _row(pk, "k-A1-old", status="revoked", revoked_at="2026-01-01T00:00:00+00:00",
                 thumbprint=thumb),
            _row(pk, "k-A1", thumbprint=thumb),
        ]
    }
    payload = _payload(key_thumbprint=thumb)
    report = vr.run_structured(_envelope(payload, sk, ml, "k-A1"), jwks, None)
    axes = _axes(report)
    assert axes["signature"]["result"] == "PASS", axes["signature"]
    assert axes["key_status"]["result"] == "PASS", axes["key_status"]
    assert "k-A1" in axes["issuer_key"]["note"]


def test_the_signed_agent_id_narrows_a_thumbprint_shared_across_agents() -> None:
    ml = _ml()
    pk, sk = ml.keygen()
    thumb = vr.thumbprint_for_key(alg="ML-DSA-65", public_key=pk)
    other = _row(pk, "k-other", thumbprint=thumb)
    other["agent_id"] = "agt-B"
    jwks = {"keys": [other, _row(pk, "k-mine", thumbprint=thumb)]}
    entry = vr.match_signing_key(jwks, "", AGENT, ORG, ORG, thumb)
    assert entry is not None and entry["kid"] == "k-mine"


# === the agent bind verifies every candidate before it picks one (448b) ===


def _rotated_directory(old_pk: bytes, new_pk: bytes) -> dict:
    # The live directory shape: no key_thumbprint column, the revoked key listed first.
    return {
        "keys": [
            _row(old_pk, "k-old", status="revoked", revoked_at="2026-05-01T00:00:00+00:00"),
            _row(new_pk, "k-new"),
        ]
    }


def test_a_rotated_agent_verifies_against_the_key_that_signed(ml=None) -> None:
    ml = _ml()
    old_pk, _old_sk = ml.keygen()
    new_pk, new_sk = ml.keygen()
    jwks = _rotated_directory(old_pk, new_pk)
    payload = _payload()
    report = vr.run_structured(_envelope(payload, new_sk, ml, ORG), jwks, None)
    axes = _axes(report)
    assert axes["signature"]["result"] == "PASS", axes["signature"]
    assert axes["key_status"]["result"] == "PASS", axes["key_status"]
    assert "k-new" in axes["issuer_key"]["note"]


def test_a_historical_receipt_still_resolves_the_revoked_key_that_signed_it() -> None:
    ml = _ml()
    old_pk, old_sk = ml.keygen()
    new_pk, _new_sk = ml.keygen()
    jwks = _rotated_directory(old_pk, new_pk)
    payload = _payload(issued_at="2026-04-01T00:00:00+00:00")
    report = vr.run_structured(_envelope(payload, old_sk, ml, ORG), jwks, None)
    axes = _axes(report)
    assert axes["signature"]["result"] == "PASS", axes["signature"]
    # Revocation is the key_status axis's call, never the resolver's.
    assert axes["key_status"]["result"] != "PASS", axes["key_status"]
    assert "revoked" in axes["key_status"]["note"]


def test_a_signature_no_candidate_verifies_reports_the_exhausted_agent_bind() -> None:
    ml = _ml()
    old_pk, _old_sk = ml.keygen()
    new_pk, _new_sk = ml.keygen()
    _stranger_pk, stranger_sk = ml.keygen()
    jwks = _rotated_directory(old_pk, new_pk)
    payload = _payload()
    report = vr.run_structured(_envelope(payload, stranger_sk, ml, ORG), jwks, None)
    axes = _axes(report)
    assert axes["signature"]["result"] == "FAIL", axes["signature"]
    assert "no key published for this agent verified" in axes["issuer_key"]["note"]


def test_the_agent_candidate_set_is_bounded() -> None:
    ml = _ml()
    pks = [ml.keygen() for _ in range(vr._AGENT_BIND_CANDIDATE_CAP + 4)]
    jwks = {"keys": [_row(pk, f"k-{i}") for i, (pk, _sk) in enumerate(pks)]}
    payload = _payload()
    msg = vr.canonical_json(payload)
    entry, _res, exhausted = vr._select_agent_bound_key(
        jwks, AGENT, ORG, ORG, msg, b"\x00" * 64, "ML-DSA-65"
    )
    assert exhausted is True
    assert entry is not None and entry["kid"] == "k-0"
