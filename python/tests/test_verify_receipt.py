"""Zero-dep verifier accepts the risk-acceptance and code-authorship receipt types.

verify_receipt.py carries protectmcp:lifecycle:risk_acceptance and
protectmcp:lifecycle:code_authorship in ALLOWED_TYPES so the structure axis
recognises a valid receipt of either type and PASSes on it; an unknown type
still fails closed. These tests pin both behaviours.
"""

from __future__ import annotations

from asqav.verifier import verify_receipt as v

RISK_TYPE = "protectmcp:lifecycle:risk_acceptance"
CODE_TYPE = "protectmcp:lifecycle:code_authorship"


def _risk_payload() -> dict:
    return {
        "type": RISK_TYPE,
        "issued_at": "2026-06-01T19:26:44.289388Z",
        "issuer_id": "c37probe-org-00001",
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "observation",
        "approver_id": "approver:alice@example.com",
        "acceptance_reason": "accepted",
        "sarif_digest": "sha256:" + "9" * 64,
    }


def test_risk_acceptance_in_allowed_types() -> None:
    assert RISK_TYPE in v.ALLOWED_TYPES


def test_structure_accepts_risk_acceptance() -> None:
    """The structure axis PASSes on a real risk-acceptance receipt; the type is
    inside ALLOWED_TYPES so it is recognised, not rejected."""
    res, note = v.check_structure(_risk_payload())
    assert res == "PASS"
    assert RISK_TYPE in note


def test_structure_still_rejects_unknown_type() -> None:
    payload = _risk_payload()
    payload["type"] = "protectmcp:not_a_real_type"
    res, _note = v.check_structure(payload)
    assert res == "FAIL"


def test_run_structure_axis_passes_on_risk_acceptance(capsys) -> None:
    """A full run over a risk-acceptance envelope prints `[  ok] structure`:
    the type is recognised, so the structure axis contributes a PASS and the
    risk-acceptance type never appears in a FAIL line."""
    envelope = {
        "payload": _risk_payload(),
        "signature": {"alg": "ML-DSA-65", "kid": "c37probe-org-00001", "sig": "AAAA"},
        "anchors": [],
    }
    v.run(envelope, {"keys": []}, predecessor_payload=None)
    out = capsys.readouterr().out
    assert "[  ok] structure" in out
    # The risk-acceptance type must never appear in a FAIL line.
    for line in out.splitlines():
        if "[FAIL]" in line:
            assert "structure" not in line


def _code_payload() -> dict:
    return {
        "type": CODE_TYPE,
        "issued_at": "2026-06-02T10:00:00.000000Z",
        "issuer_id": "c39probe-org-00001",
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "observation",
        "repo_ref": "github.com/acme/repo",
        "commit_sha": "c0ffee0000000000000000000000000000000000",
        "change_digest": "sha256:" + "9" * 64,
        "change_class": "write",
    }


def test_code_authorship_in_allowed_types() -> None:
    assert CODE_TYPE in v.ALLOWED_TYPES


def test_structure_accepts_code_authorship() -> None:
    """The structure axis PASSes on a real code-authorship receipt; the type is
    inside ALLOWED_TYPES so it is recognised, not rejected."""
    res, note = v.check_structure(_code_payload())
    assert res == "PASS"
    assert CODE_TYPE in note


def test_run_structure_axis_passes_on_code_authorship(capsys) -> None:
    """A full run over a code-authorship envelope prints `[  ok] structure`:
    the type is recognised, so the structure axis contributes a PASS and the
    code-authorship type never appears in a FAIL line."""
    envelope = {
        "payload": _code_payload(),
        "signature": {"alg": "ML-DSA-65", "kid": "c39probe-org-00001", "sig": "AAAA"},
        "anchors": [],
    }
    v.run(envelope, {"keys": []}, predecessor_payload=None)
    out = capsys.readouterr().out
    assert "[  ok] structure" in out
    # The code-authorship type must never appear in a FAIL line.
    for line in out.splitlines():
        if "[FAIL]" in line:
            assert "structure" not in line


# === payload: null fail-clean (stranger-funnel B2) ===


def test_run_payload_null_fails_clean(capsys) -> None:
    """A hosted /verify response with ``payload: null`` must not traceback.

    The published verifier crashed with a raw AttributeError here; it now
    prints a readable message and returns the INCOMPLETE exit code (2).
    """
    envelope = {
        "signature_id": "sig_abc123",
        "payload": None,
        "signature": "AAAA",
        "algorithm": "ML-DSA-65",
    }
    code = v.run(envelope, {"keys": []}, predecessor_payload=None)
    out = capsys.readouterr().out
    assert code == 2
    assert "receipt payload not available from this surface" in out
    assert "INCOMPLETE" in out
    assert "PASS" not in out.replace("never a PASS", "")


def test_run_payload_null_without_signature_keys(capsys) -> None:
    """Same guard when the response carries only nulls."""
    code = v.run({"payload": None}, {"keys": []}, predecessor_payload=None)
    out = capsys.readouterr().out
    assert code == 2
    assert "INCOMPLETE" in out


def test_verify_signature_nonstring_alg_skips_clean() -> None:
    """A non-string alg SKIPs cleanly instead of crashing on (alg or '').upper()."""
    z = b""
    for bad_alg in (123, None, ["x"], {}):
        result, _ = v.verify_signature(z, z, z, bad_alg)
        assert result == "SKIPPED"


def test_check_key_status_tz_mismatch_fails_clean() -> None:
    """Revoked-before-issuance with a tz-aware revoked_at and tz-naive issued_at FAILs
    the axis, never raises a naive-vs-aware TypeError."""
    result, _ = v.check_key_status(
        "revoked", "2026-06-01T00:00:00", "2026-05-04T12:00:00+00:00"
    )
    assert result == "FAIL"


def test_check_skew_tz_naive_issued_at_no_crash() -> None:
    """A tz-naive issued_at must produce a verdict, never a naive-vs-aware TypeError
    in the wall-clock subtraction."""
    result, _ = v.check_skew("2026-05-04T00:00:00")
    assert result in ("PASS", "FAIL")
