"""Offline verifier gates the verdict on the signing key's revocation status.

A receipt signed by a key the JWKS marks revoked must not PASS offline, matching
the hosted /verify. The public JWKS publishes status today (no revoked_at), so a
revoked key fails the key_status axis and the verdict is FAIL, not PASS.
"""

from __future__ import annotations

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


def test_check_key_status_revoked_after_issuance_passes():
    """A receipt signed before the key was revoked still verifies (historical)."""
    res, _ = v.check_key_status(
        "revoked", "2026-05-01T00:00:00Z", revoked_at="2026-06-01T00:00:00Z"
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
