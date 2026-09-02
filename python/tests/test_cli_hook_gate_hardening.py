"""The PreToolUse gate's cache predicate, verification budget and decision vocabulary.

Offline throughout: the signer is a stub, the JWK Set is minted here and every fetch is
monkeypatched, so a green run says nothing about api.asqav.com and everything about the gate.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import pytest
from typer.testing import CliRunner

from asqav import cli_hook
from asqav.cli import app
from asqav.verifier import verify_receipt as vr

runner = CliRunner()

_PRETOOL_EVENT = {
    "session_id": "s9",
    "tool_name": "Write",
    "tool_input": {"file_path": "/tmp/f", "content": "hi"},
}
ORG = "org-hook"
AGENT = "agt_hook"


@pytest.fixture(autouse=True)
def _identity(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("ASQAV_API_KEY", "sk_test_hook")
    monkeypatch.setenv("ASQAV_AGENT_ID", AGENT)
    monkeypatch.setenv(cli_hook._JWKS_CACHE_ENV, str(tmp_path / "jwks-cache.json"))
    monkeypatch.delenv(cli_hook._DEADLINE_ENV, raising=False)


class _Sig:
    def __init__(self, signature_id: str, payload: Any = None, signature: Any = None, anchors=None):
        self.signature_id = signature_id
        self.payload = payload
        self.signature = signature
        self.anchors = anchors


def _ml_dsa_65():
    pytest.importorskip("dilithium_py")
    from dilithium_py.ml_dsa import ML_DSA_65

    return ML_DSA_65


def _payload(decision: str = "allow", **extra: Any) -> dict:
    payload = {
        "type": "protectmcp:decision",
        "issued_at": "2026-06-19T00:00:00+00:00",
        "issuer_id": ORG,
        "agent_id": AGENT,
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": decision,
    }
    payload.update(extra)
    return payload


def _signed(payload: dict, sk: bytes, ml) -> dict:
    sig = ml.sign(sk, vr.canonical_json(payload))
    return {"alg": "ML-DSA-65", "kid": ORG, "sig": base64.b64encode(sig).decode()}


def _row(pk: bytes, kid: str, *, thumbprint: str | None = None) -> dict:
    row = {
        "kid": kid,
        "agent_id": AGENT,
        "issuer_id": ORG,
        "org_id": ORG,
        "alg": "ML-DSA-65",
        "kty": "AKP",
        "public_key": base64.b64encode(pk).decode(),
        "status": "active",
        "revoked_at": None,
    }
    if thumbprint is not None:
        row["key_thumbprint"] = thumbprint
    return row


def _gate_with(monkeypatch, sig: _Sig, jwks: dict, fetch_calls: list | None = None):
    monkeypatch.setattr(cli_hook, "_sign_event", lambda **_k: sig)

    def fake_fetch(_timeout: float) -> dict:
        if fetch_calls is not None:
            fetch_calls.append(_timeout)
        return jwks

    monkeypatch.setattr(cli_hook, "_fetch_jwks", fake_fetch)
    return runner.invoke(app, ["hook", "pretool"], input=json.dumps(_PRETOOL_EVENT))


# === the cache predicate reads the signed thumbprint first (447c) ===


def test_a_rotated_key_refreshes_the_cache_instead_of_blocking_for_a_day(monkeypatch) -> None:
    ml = _ml_dsa_65()
    old_pk, _old_sk = ml.keygen()
    new_pk, new_sk = ml.keygen()
    new_thumb = vr.thumbprint_for_key(alg="ML-DSA-65", public_key=new_pk)
    old_thumb = vr.thumbprint_for_key(alg="ML-DSA-65", public_key=old_pk)
    payload = _payload("allow", key_thumbprint=new_thumb)
    sig = _Sig("sig_rotated", payload, _signed(payload, new_sk, ml), [])
    # The cache predates the rotation: same agent_id, the key the receipt does not bind.
    cli_hook._write_jwks_cache({"keys": [_row(old_pk, "k-old", thumbprint=old_thumb)]})
    calls: list = []
    live = {"keys": [_row(old_pk, "k-old", thumbprint=old_thumb),
                     _row(new_pk, "k-new", thumbprint=new_thumb)]}
    result = _gate_with(monkeypatch, sig, live, calls)
    assert result.exit_code == 0, result.output
    assert len(calls) == 1, "the stale cache answered 'present' and the refresh never fired"
    assert "permit sig_rotated" in result.output


def test_the_agent_bind_still_answers_when_the_directory_publishes_no_thumbprint(monkeypatch) -> None:
    ml = _ml_dsa_65()
    pk, sk = ml.keygen()
    payload = _payload("allow", key_thumbprint=vr.thumbprint_for_key(alg="ML-DSA-65", public_key=pk))
    sig = _Sig("sig_nothumb", payload, _signed(payload, sk, ml), [])
    live = {"keys": [_row(pk, "k-live")]}
    calls: list = []
    first = _gate_with(monkeypatch, sig, live, calls)
    assert first.exit_code == 0, first.output
    assert len(calls) == 1
    second = _gate_with(monkeypatch, sig, live, calls)
    assert second.exit_code == 0, second.output
    assert len(calls) == 1, "a thumbprint-free directory forced a re-fetch on every call"


def test_a_receipt_naming_no_key_is_refreshed_once_then_blocked_with_a_reason(monkeypatch) -> None:
    ml = _ml_dsa_65()
    pk, sk = ml.keygen()
    payload = _payload("allow")
    payload.pop("agent_id")
    envelope_sig = _signed(payload, sk, ml)
    envelope_sig["kid"] = ""
    sig = _Sig("sig_anon", payload, envelope_sig, [])
    cli_hook._write_jwks_cache({"keys": [_row(pk, "k-live")]})
    calls: list = []
    result = _gate_with(monkeypatch, sig, {"keys": [_row(pk, "k-live")]}, calls)
    assert result.exit_code == 2, result.output
    assert len(calls) == 1, f"expected exactly one refresh, got {len(calls)}"
    assert "names no key" in result.output


def test_a_key_the_refreshed_set_does_not_publish_blocks_with_that_reason(monkeypatch) -> None:
    ml = _ml_dsa_65()
    pk, sk = ml.keygen()
    other_pk, _other_sk = ml.keygen()
    payload = _payload("allow", key_thumbprint=vr.thumbprint_for_key(alg="ML-DSA-65", public_key=pk))
    sig = _Sig("sig_absent", payload, _signed(payload, sk, ml), [])
    live = {"keys": [_row(other_pk, "k-other",
                          thumbprint=vr.thumbprint_for_key(alg="ML-DSA-65", public_key=other_pk))]}
    cli_hook._write_jwks_cache(live)
    calls: list = []
    result = _gate_with(monkeypatch, sig, live, calls)
    assert result.exit_code == 2, result.output
    assert len(calls) == 1
    assert "thumbprint" in result.output and "blocking tool call" in result.output


def test_the_predicate_prefers_the_thumbprint_over_the_agent_id() -> None:
    ml = _ml_dsa_65()
    old_pk, _ = ml.keygen()
    new_pk, _ = ml.keygen()
    stale = {"keys": [_row(old_pk, "k-old",
                           thumbprint=vr.thumbprint_for_key(alg="ML-DSA-65", public_key=old_pk))]}
    payload = _payload("allow", key_thumbprint=vr.thumbprint_for_key(alg="ML-DSA-65", public_key=new_pk))
    assert cli_hook._key_present(stale, payload) is False


# === the wire decision vocabulary (447d) ===


def _gate_for_decision(monkeypatch, decision: str):
    ml = _ml_dsa_65()
    pk, sk = ml.keygen()
    payload = _payload(decision)
    if decision != "allow":
        payload["type"] = "protectmcp:lifecycle"
    sig = _Sig(f"sig_{decision}", payload, _signed(payload, sk, ml), [])
    return _gate_with(monkeypatch, sig, {"keys": [_row(pk, "k-live")]})


def test_an_allow_receipt_prints_permit_and_exits_zero(monkeypatch) -> None:
    result = _gate_for_decision(monkeypatch, "allow")
    assert result.exit_code == 0, result.output
    assert "permit sig_allow" in result.output


def test_an_observation_receipt_prints_signed_and_exits_zero(monkeypatch) -> None:
    result = _gate_for_decision(monkeypatch, "observation")
    assert result.exit_code == 0, result.output
    assert "signed sig_observation" in result.output
    assert "permit" not in result.output


def test_a_deny_receipt_is_announced_as_blocked_and_exits_two(monkeypatch) -> None:
    result = _gate_for_decision(monkeypatch, "deny")
    assert result.exit_code == 2, result.output
    assert "blocked sig_deny: deny" in result.output
    assert "signed" not in result.output


def test_a_rate_limit_receipt_is_announced_as_blocked_and_exits_two(monkeypatch) -> None:
    result = _gate_for_decision(monkeypatch, "rate_limit")
    assert result.exit_code == 2, result.output
    assert "blocked sig_rate_limit: rate_limit" in result.output
    assert "signed" not in result.output
