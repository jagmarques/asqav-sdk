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


def test_resolve_key_by_agent_id_matches_agent_id() -> None:
    """The signing key is resolvable by the receipt's agent_id (cloud receipts set
    kid to the issuer/org id but sign with the agent's own key), but only when the
    key's issuer_id equals the receipt's claimed issuer."""
    jwks = {
        "keys": [
            {
                "kid": "agent-key-1",
                "agent_id": "agt_probe",
                "issuer_id": "org-1",
                "alg": "Ed25519",
                "public_key": "AAAA",
                "status": "active",
            }
        ]
    }
    pk, status, alg, kid = v.resolve_key_by_agent_id(jwks, "agt_probe", "org-1")
    assert pk == v._b64decode("AAAA")
    assert (status, alg, kid) == ("active", "Ed25519", "agent-key-1")
    # Unknown agent_id resolves nothing.
    assert v.resolve_key_by_agent_id(jwks, "agt_other", "org-1") == (None, None, None, None)
    # Right agent_id but a different claimed issuer resolves nothing (the bind).
    assert v.resolve_key_by_agent_id(jwks, "agt_probe", "org-evil") == (None, None, None, None)


def test_agent_id_fallback_never_trusts_a_non_verifying_key() -> None:
    """The agent_id fallback only trusts an agent key that actually verifies the
    signature; a signature made by a different key still FAILs (no false pass)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )

    import base64

    signing_key = Ed25519PrivateKey.generate()
    other_key = Ed25519PrivateKey.generate()
    msg = v.canonical_json({"type": "protectmcp:decision", "agent_id": "agt_probe"})
    # Signature made by `other_key`, but the JWKS ties agent_id to `signing_key`.
    wrong_sig = base64.b64encode(other_key.sign(msg)).decode()
    jwks = {
        "keys": [
            {
                "kid": "agent-key-1",
                "agent_id": "agt_probe",
                "issuer_id": "org-1",
                "alg": "Ed25519",
                "public_key": base64.b64encode(
                    signing_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
                ).decode(),
                "status": "active",
            }
        ]
    }
    envelope = {
        "payload": {"type": "protectmcp:decision", "agent_id": "agt_probe", "issuer_id": "org-1"},
        "signature": {"alg": "Ed25519", "kid": "org-1", "sig": wrong_sig},
        "anchors": [],
    }
    code = v.run(envelope, jwks, None)
    assert code == 1  # FAIL: the agent key does not verify the forged signature


# === agent_id fallback must bind the resolved key to the claimed issuer ===


def _ml_dsa_65():
    import pytest

    return pytest.importorskip("dilithium_py.ml_dsa").ML_DSA_65


def _valid_payload(issuer_id: str, agent_id: str) -> dict:
    from datetime import datetime, timezone

    return {
        "type": "protectmcp:decision",
        "issued_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "issuer_id": issuer_id,
        "agent_id": agent_id,
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
    }


def _jwks_key(kid: str, agent_id: str, issuer_id: str, pub: bytes) -> dict:
    import base64

    return {
        "kid": kid,
        "agent_id": agent_id,
        "issuer_id": issuer_id,
        "alg": "ML-DSA-65",
        "public_key": base64.b64encode(pub).decode(),
        "status": "active",
    }


def _envelope(payload: dict, sig: bytes) -> dict:
    import base64

    return {
        "payload": payload,
        "signature": {
            "alg": "ML-DSA-65",
            "kid": payload["issuer_id"],
            "sig": base64.b64encode(sig).decode(),
        },
        "anchors": [{"type": "test", "value": "AAAA"}],
    }


def test_run_agent_id_fallback_rejects_forged_issuer() -> None:
    """FORGERY: a receipt signed by org-attacker's real ML-DSA-65 agent key but
    claiming issuer_id=org-victim must FAIL. The fallback only trusts a key whose
    issuer_id equals the claimed issuer, so the cross-issuer key never PASSes."""
    ml = _ml_dsa_65()
    attacker_pk, attacker_sk = ml.keygen()
    victim_pk, _victim_sk = ml.keygen()
    payload = _valid_payload("org-victim", "agt_attacker")
    sig = ml.sign(attacker_sk, v.canonical_json(payload))  # signed with attacker key
    jwks = {
        "keys": [
            _jwks_key("victim-key", "agt_victim", "org-victim", victim_pk),
            _jwks_key("attacker-key", "agt_attacker", "org-attacker", attacker_pk),
        ]
    }
    assert v.run(_envelope(payload, sig), jwks, None) == 1  # bound: attacker key rejected


def test_run_agent_id_fallback_passes_legit_multi_agent() -> None:
    """LEGIT offline PASS: an org with two agents. The receipt is signed by the
    second agent; the issuer-kid resolves the first (wrong) sibling, and the
    agent_id fallback resolves the real signer whose issuer_id matches -> PASS."""
    ml = _ml_dsa_65()
    a1_pk, _a1_sk = ml.keygen()
    a2_pk, a2_sk = ml.keygen()
    payload = _valid_payload("org-legit", "agt_two")
    sig = ml.sign(a2_sk, v.canonical_json(payload))  # signed by agent two
    jwks = {
        "keys": [
            _jwks_key("agent-one", "agt_one", "org-legit", a1_pk),  # issuer-kid hits this
            _jwks_key("agent-two", "agt_two", "org-legit", a2_pk),  # real signer via agent_id
        ]
    }
    assert v.run(_envelope(payload, sig), jwks, None) == 0  # fallback resolves the bound signer


def test_run_tampered_signature_fails() -> None:
    """A flipped signature byte on the legit multi-agent receipt FAILs."""
    ml = _ml_dsa_65()
    a1_pk, _a1_sk = ml.keygen()
    a2_pk, a2_sk = ml.keygen()
    payload = _valid_payload("org-legit", "agt_two")
    sig = bytearray(ml.sign(a2_sk, v.canonical_json(payload)))
    sig[0] ^= 0xFF  # tamper one byte
    jwks = {
        "keys": [
            _jwks_key("agent-one", "agt_one", "org-legit", a1_pk),
            _jwks_key("agent-two", "agt_two", "org-legit", a2_pk),
        ]
    }
    assert v.run(_envelope(payload, bytes(sig)), jwks, None) == 1


def test_run_tampered_payload_fails() -> None:
    """Mutating a payload field after signing FAILs; the signature no longer binds."""
    ml = _ml_dsa_65()
    a1_pk, _a1_sk = ml.keygen()
    a2_pk, a2_sk = ml.keygen()
    payload = _valid_payload("org-legit", "agt_two")
    sig = ml.sign(a2_sk, v.canonical_json(payload))
    payload["decision"] = "deny"  # tamper after signing
    jwks = {
        "keys": [
            _jwks_key("agent-one", "agt_one", "org-legit", a1_pk),
            _jwks_key("agent-two", "agt_two", "org-legit", a2_pk),
        ]
    }
    assert v.run(_envelope(payload, sig), jwks, None) == 1
