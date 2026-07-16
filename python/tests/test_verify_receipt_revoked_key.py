"""Offline verifier gates the verdict on the signing key's revocation status.

A receipt signed by a key the JWKS marks revoked must not PASS offline, matching
the hosted /verify. Offline the verifier cannot verify a timestamp anchor, so a
revoked key never rides a forgeable anchor to PASS: the key_status axis is FAIL or
SKIPPED and the verdict is never PASS.
"""

from __future__ import annotations

import pytest

from asqav.verifier import verify_receipt as v
from asqav.verifier.oracle.adapters.asqav_native import AsqavNativeAdapter


def _payload() -> dict:
    return {
        "type": "protectmcp:decision",
        "issued_at": "2026-06-01T19:26:44.289388Z",
        "issuer_id": "agent-revoked-001",
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
    }


def _jwks(status: str, revoked_at=None) -> dict:
    key = {
        "kid": "agent-revoked-001",
        "issuer_id": "agent-revoked-001",
        "alg": "ML-DSA-65",
        "public_key": "QUFBQQ==",  # base64 placeholder, never reached for revoked key gate
        "status": status,
    }
    if revoked_at is not None:
        key["revoked_at"] = revoked_at
    return {"keys": [key]}


# --- unit: the key-status axis ---


def test_check_key_status_active_passes():
    res, _ = v.check_key_status("active", "2026-06-01T00:00:00Z")
    assert res == "PASS"


def test_check_key_status_revoked_fails_without_revoked_at():
    res, note = v.check_key_status("revoked", "2026-06-01T00:00:00Z")
    assert res == "FAIL"
    assert "revoked" in note.lower()


def test_check_key_status_revoked_before_issuance_fails():
    res, _ = v.check_key_status(
        "revoked", "2026-06-01T00:00:00Z", revoked_at="2026-05-01T00:00:00Z"
    )
    assert res == "FAIL"


def test_check_key_status_revoked_after_issuance_skipped_without_anchor():
    """No anchor means issued_at is self-attested; backdating is undetectable."""
    res, note = v.check_key_status(
        "revoked", "2026-05-01T00:00:00Z", revoked_at="2026-06-01T00:00:00Z"
    )
    assert res == "SKIPPED"
    assert "anchor" in note.lower()


def test_check_key_status_revoked_after_issuance_passes_with_anchor():
    """A pre-revocation receipt with a trusted anchor still verifies."""
    res, _ = v.check_key_status(
        "revoked", "2026-05-01T00:00:00Z", revoked_at="2026-06-01T00:00:00Z",
        has_trusted_anchor=True,
    )
    assert res == "PASS"


# --- standalone verify_receipt.run ---


def test_run_revoked_key_does_not_pass(capsys):
    """A revoked-key receipt yields a FAIL verdict, never PASS, from run()."""
    envelope = {
        "payload": _payload(),
        "signature": {"alg": "ML-DSA-65", "kid": "agent-revoked-001", "sig": "AAAA"},
        "anchors": [],
    }
    code = v.run(envelope, _jwks("revoked"), None)
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "key_status" in out
    assert code == 1  # FAIL dominates; never 0 (PASS)


def test_run_active_key_has_key_status_pass(capsys):
    """An active key contributes a key_status PASS line."""
    envelope = {
        "payload": _payload(),
        "signature": {"alg": "ML-DSA-65", "kid": "agent-revoked-001", "sig": "AAAA"},
        "anchors": [],
    }
    v.run(envelope, _jwks("active"), None)
    out = capsys.readouterr().out
    assert "key_status" in out
    assert "active" in out


# --- run_structured path ---


def test_run_structured_revoked_key_has_key_status_fail_axis():
    """run_structured must include a key_status FAIL axis for a revoked key."""
    envelope = {
        "payload": _payload(),
        "signature": {"alg": "ML-DSA-65", "kid": "agent-revoked-001", "sig": "AAAA"},
        "anchors": [],
    }
    result = v.run_structured(envelope, _jwks("revoked"), None)
    names = [a["name"] for a in result["axes"]]
    assert "key_status" in names, f"key_status axis missing; got axes: {names}"
    ks = next(a for a in result["axes"] if a["name"] == "key_status")
    assert ks["result"] == "FAIL", f"expected FAIL, got {ks['result']!r}"


def test_run_structured_active_key_has_key_status_pass_axis():
    """run_structured includes a key_status PASS axis for an active key."""
    envelope = {
        "payload": _payload(),
        "signature": {"alg": "ML-DSA-65", "kid": "agent-revoked-001", "sig": "AAAA"},
        "anchors": [],
    }
    result = v.run_structured(envelope, _jwks("active"), None)
    names = [a["name"] for a in result["axes"]]
    assert "key_status" in names, f"key_status axis missing; got axes: {names}"
    ks = next(a for a in result["axes"] if a["name"] == "key_status")
    assert ks["result"] == "PASS", f"expected PASS, got {ks['result']!r}"


def test_run_structured_revoked_at_after_issuance_skipped_without_anchor():
    """No anchor: key_status SKIPPED (issued_at self-attested, backdating risk).
    The placeholder sig FAILs so the verdict is FAIL (or INCOMPLETE); the point
    is key_status is never PASS and the verdict is never PASS."""
    envelope = {
        "payload": _payload(),
        "signature": {"alg": "ML-DSA-65", "kid": "agent-revoked-001", "sig": "AAAA"},
        "anchors": [],
    }
    result = v.run_structured(
        envelope, _jwks("revoked", revoked_at="2026-07-01T00:00:00Z"), None
    )
    ks = next(a for a in result["axes"] if a["name"] == "key_status")
    assert ks["result"] == "SKIPPED"
    assert result["verdict"] != "PASS"


def test_run_structured_forged_anchor_stays_skipped_offline():
    """Offline a present anchor is not trusted, so key_status stays SKIPPED."""
    envelope = {
        "payload": _payload(),
        "signature": {"alg": "ML-DSA-65", "kid": "agent-revoked-001", "sig": "AAAA"},
        "anchors": [{"type": "rfc3161", "value": "dGVzdA=="}],
    }
    result = v.run_structured(
        envelope, _jwks("revoked", revoked_at="2026-07-01T00:00:00Z"), None
    )
    ks = next(a for a in result["axes"] if a["name"] == "key_status")
    assert ks["result"] == "SKIPPED"
    assert "anchor" in ks["note"].lower()


def test_run_structured_forged_anchor_does_not_upgrade_revoked_key():
    """A forgeable anchor must not upgrade a revoked key to PASS offline.
    Anchors sit outside the signed payload and are only shape-checked, so a
    revoked-key holder can append one; offline it is not trustworthy timing."""
    envelope = {
        "payload": _payload(),
        "signature": {"alg": "ML-DSA-65", "kid": "agent-revoked-001", "sig": "AAAA"},
        "anchors": [{"type": "rfc3161", "value": "dGVzdA=="}],  # base64('test'), not a token
    }
    result = v.run_structured(
        envelope, _jwks("revoked", revoked_at="2026-07-01T00:00:00Z"), None
    )
    ks = next(a for a in result["axes"] if a["name"] == "key_status")
    assert ks["result"] == "SKIPPED", f"forged anchor upgraded key_status to {ks['result']!r}"
    assert result["verdict"] != "PASS"


def test_offline_forged_anchor_revoked_key_verdict_not_pass():
    """End-to-end: a revoked key cannot mint a PASS by appending a forged anchor.
    A real ML-DSA-65 signature over a backdated payload verifies (the holder keeps
    the key), but the forged anchor must not upgrade it, so verdict is INCOMPLETE."""
    import base64

    pytest.importorskip("dilithium_py")
    from dilithium_py.ml_dsa import ML_DSA_65

    pk, sk = ML_DSA_65.keygen()
    payload = _payload()  # issued_at precedes the revoked_at below
    sig = base64.b64encode(ML_DSA_65.sign(sk, v.canonical_json(payload))).decode()
    jwks = _jwks("revoked", revoked_at="2026-07-11T14:57:49Z")
    jwks["keys"][0]["public_key"] = base64.b64encode(pk).decode()
    envelope = {
        "payload": payload,
        "signature": {"alg": "ML-DSA-65", "kid": "agent-revoked-001", "sig": sig},
        "anchors": [{"type": "rfc3161", "value": "dGVzdA=="}],
    }
    result = v.run_structured(envelope, jwks, None)
    assert result["verdict"] == "INCOMPLETE", f"revoked key produced verdict {result['verdict']!r}"


# --- oracle adapter path ---


def test_oracle_revoked_key_axis_fails():
    """The asqav-native adapter surfaces a failing key_status axis for a revoked key."""
    doc = {
        "payload": _payload(),
        "signature": {"alg": "ML-DSA-65", "kid": "agent-revoked-001", "sig": "AAAA"},
        "anchors": [],
    }
    ad = AsqavNativeAdapter()
    axes = ad.extra_axes(doc, _jwks("revoked"))
    assert ("key_status", "FAIL", axes[0][2]) == axes[0]


def test_oracle_active_key_axis_passes():
    doc = {
        "payload": _payload(),
        "signature": {"alg": "ML-DSA-65", "kid": "agent-revoked-001", "sig": "AAAA"},
        "anchors": [],
    }
    ad = AsqavNativeAdapter()
    axes = ad.extra_axes(doc, _jwks("active"))
    assert axes[0][1] == "PASS"


def test_oracle_forged_anchor_revoked_key_axis_skipped():
    """The adapter must not let a forged anchor upgrade a revoked key's axis."""
    doc = {
        "payload": _payload(),
        "signature": {"alg": "ML-DSA-65", "kid": "agent-revoked-001", "sig": "AAAA"},
        "anchors": [{"type": "rfc3161", "value": "dGVzdA=="}],
    }
    ad = AsqavNativeAdapter()
    axes = ad.extra_axes(doc, _jwks("revoked", revoked_at="2026-07-01T00:00:00Z"))
    assert axes[0][1] == "SKIPPED", f"forged anchor upgraded axis to {axes[0][1]!r}"
