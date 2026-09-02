"""The PreToolUse gate's deadline, hash-only default and receipt verification.

Criteria 575, 578 and 579. Every path here is offline: the signer is a stub and the JWK Set is
minted in the test, so a green run says nothing about api.asqav.com and everything about the gate.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from typing import Any

import pytest
from typer.testing import CliRunner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav import cli_hook  # noqa: E402
from asqav.cli import app  # noqa: E402
from asqav.verifier import verify_receipt as vr  # noqa: E402

runner = CliRunner()

_PRETOOL_EVENT = {
    "session_id": "s2",
    "tool_name": "Write",
    "tool_input": {"file_path": "/tmp/f", "content": "hi"},
}
_POSTTOOL_EVENT = {
    "session_id": "s1",
    "tool_name": "Bash",
    "tool_input": {"command": "ls"},
    "tool_response": "x",
}
ORG = "org-hook"


@pytest.fixture(autouse=True)
def _identity(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("ASQAV_API_KEY", "sk_test_hook")
    monkeypatch.setenv("ASQAV_AGENT_ID", "agt_hook")
    monkeypatch.setenv(cli_hook._JWKS_CACHE_ENV, str(tmp_path / "jwks-cache.json"))
    monkeypatch.delenv(cli_hook._DEADLINE_ENV, raising=False)


class _Sig:
    def __init__(self, signature_id: str, payload: Any = None, signature: Any = None, anchors=None):
        self.signature_id = signature_id
        self.payload = payload
        self.signature = signature
        self.anchors = anchors


def _sleepy_signer(seconds: float):
    def _sign(**_kwargs: Any) -> _Sig:
        time.sleep(seconds)
        return _Sig("sig_late")

    return _sign


# === hash-only default (579) ===


def test_pretool_dry_run_sends_a_digest_not_the_arguments() -> None:
    result = runner.invoke(app, ["hook", "pretool", "--dry-run"], input=json.dumps(_PRETOOL_EVENT))
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert "context" not in body
    assert body["hash"].startswith("sha256:")
    assert body["hash_algo"] == "sha256"
    assert "/tmp/f" not in result.output and "hi" not in json.dumps(body.get("metadata", {}))


def test_pretool_dry_run_clear_context_sends_the_arguments() -> None:
    result = runner.invoke(
        app, ["hook", "pretool", "--dry-run", "--clear-context"], input=json.dumps(_PRETOOL_EVENT)
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["context"] == {"tool_input": {"file_path": "/tmp/f", "content": "hi"}}


def test_sign_event_inits_hash_only_by_default_and_carries_the_deadline(monkeypatch) -> None:
    import asqav

    seen: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> None:
        seen.update(kwargs)

    class _Agent:
        agent_id = "agt_hook"

        def sign(self, *_a: Any, **_k: Any) -> _Sig:
            return _Sig("sig_ok")

    monkeypatch.setattr(asqav, "init", fake_init)
    monkeypatch.setattr(asqav.Agent, "get", staticmethod(lambda _id: _Agent()))
    cli_hook._sign_event(
        action_type="tool:Write", context={"tool_input": {}}, session_id="s",
        compliance_fields={"compliance_mode": True}, api_key="k", agent_id="agt_hook", timeout=2.5,
    )
    assert seen["mode"] == "hash-only" and seen["timeout"] == 2.5
    cli_hook._sign_event(
        action_type="tool:Write", context={"tool_input": {}}, session_id="s",
        compliance_fields={"compliance_mode": True}, api_key="k", agent_id="agt_hook",
        clear_context=True,
    )
    assert seen["mode"] == "full-payload"


# === deadline (575) ===


def test_pretool_blocks_when_the_signer_outlives_the_deadline(monkeypatch) -> None:
    monkeypatch.setenv(cli_hook._DEADLINE_ENV, "0.2")
    monkeypatch.setattr(cli_hook, "_sign_event", _sleepy_signer(5.0))
    started = time.monotonic()
    result = runner.invoke(app, ["hook", "pretool"], input=json.dumps(_PRETOOL_EVENT))
    elapsed = time.monotonic() - started
    assert result.exit_code == 2, result.output
    assert "did not answer within 0.2s" in result.output
    assert elapsed < 2.0, f"gate took {elapsed:.2f}s; the deadline did not fire"


def test_posttool_proceeds_unsigned_when_the_signer_outlives_the_deadline(monkeypatch) -> None:
    monkeypatch.setenv(cli_hook._DEADLINE_ENV, "0.2")
    monkeypatch.setattr(cli_hook, "_sign_event", _sleepy_signer(5.0))
    started = time.monotonic()
    result = runner.invoke(app, ["hook", "posttool"], input=json.dumps(_POSTTOOL_EVENT))
    assert result.exit_code == 0, result.output
    assert "proceeding unsigned" in result.output
    assert time.monotonic() - started < 2.0


def test_deadline_env_falls_back_to_the_default_on_garbage(monkeypatch) -> None:
    monkeypatch.setenv(cli_hook._DEADLINE_ENV, "soon")
    assert cli_hook._deadline_seconds() == cli_hook._DEFAULT_DEADLINE_SECONDS
    monkeypatch.setenv(cli_hook._DEADLINE_ENV, "-3")
    assert cli_hook._deadline_seconds() == cli_hook._DEFAULT_DEADLINE_SECONDS
    monkeypatch.setenv(cli_hook._DEADLINE_ENV, "7.5")
    assert cli_hook._deadline_seconds() == 7.5


# === receipt verification (578) ===


def _ml_dsa_65():
    pytest.importorskip("dilithium_py")
    from dilithium_py.ml_dsa import ML_DSA_65

    return ML_DSA_65


def _payload(decision: str = "allow") -> dict:
    return {
        "type": "protectmcp:decision",
        "issued_at": "2026-06-19T00:00:00+00:00",
        "issuer_id": ORG,
        "agent_id": "agt_hook",
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": decision,
    }


def _signed(payload: dict, sk: bytes, ml) -> dict:
    sig = ml.sign(sk, vr.canonical_json(payload))
    return {"alg": "ML-DSA-65", "kid": ORG, "sig": base64.b64encode(sig).decode()}


def _jwks(pk: bytes) -> dict:
    return {
        "keys": [
            {
                "kid": "k-hook",
                "agent_id": "agt_hook",
                "issuer_id": ORG,
                "org_id": ORG,
                "alg": "ML-DSA-65",
                "kty": "AKP",
                "public_key": base64.b64encode(pk).decode(),
                "status": "active",
                "revoked_at": None,
            }
        ]
    }


def _gate_with(monkeypatch, sig: _Sig, jwks: dict, fetch_calls: list | None = None):
    monkeypatch.setattr(cli_hook, "_sign_event", lambda **_k: sig)

    def fake_fetch(_timeout: float) -> dict:
        if fetch_calls is not None:
            fetch_calls.append(_timeout)
        return jwks

    monkeypatch.setattr(cli_hook, "_fetch_jwks", fake_fetch)
    return runner.invoke(app, ["hook", "pretool"], input=json.dumps(_PRETOOL_EVENT))


def test_pretool_accepts_a_receipt_that_verifies_and_labels_a_permit(monkeypatch) -> None:
    ml = _ml_dsa_65()
    pk, sk = ml.keygen()
    payload = _payload("allow")
    sig = _Sig("sig_ok", payload, _signed(payload, sk, ml), [])
    result = _gate_with(monkeypatch, sig, _jwks(pk))
    assert result.exit_code == 0, result.output
    assert "permit sig_ok" in result.output


def test_pretool_labels_an_observation_as_signed_not_permit(monkeypatch) -> None:
    ml = _ml_dsa_65()
    pk, sk = ml.keygen()
    payload = _payload("observation")
    payload["type"] = "protectmcp:lifecycle"
    sig = _Sig("sig_obs", payload, _signed(payload, sk, ml), [])
    result = _gate_with(monkeypatch, sig, _jwks(pk))
    assert result.exit_code == 0, result.output
    assert "signed sig_obs" in result.output
    assert "permit" not in result.output


def test_pretool_blocks_a_forged_receipt(monkeypatch) -> None:
    ml = _ml_dsa_65()
    pk, _sk = ml.keygen()
    _pk2, forger_sk = ml.keygen()
    payload = _payload("allow")
    sig = _Sig("sig_forged", payload, _signed(payload, forger_sk, ml), [])
    result = _gate_with(monkeypatch, sig, _jwks(pk))
    assert result.exit_code == 2, result.output
    assert "did not verify" in result.output


def test_pretool_blocks_when_the_signer_returns_no_receipt_bytes(monkeypatch) -> None:
    result = _gate_with(monkeypatch, _Sig("sig_bare"), {"keys": []})
    assert result.exit_code == 2, result.output
    assert "no verifiable receipt" in result.output


def test_pretool_blocks_when_the_jwks_cannot_be_fetched(monkeypatch) -> None:
    ml = _ml_dsa_65()
    _pk, sk = ml.keygen()
    payload = _payload("allow")
    monkeypatch.setattr(cli_hook, "_sign_event", lambda **_k: _Sig("sig_x", payload, _signed(payload, sk, ml), []))

    def broken(_timeout: float) -> dict:
        raise OSError("network down")

    monkeypatch.setattr(cli_hook, "_fetch_jwks", broken)
    result = runner.invoke(app, ["hook", "pretool"], input=json.dumps(_PRETOOL_EVENT))
    assert result.exit_code == 2, result.output
    assert "could not verify" in result.output


def test_jwks_cache_is_reused_and_refreshed_once_on_a_key_miss(monkeypatch) -> None:
    ml = _ml_dsa_65()
    pk, sk = ml.keygen()
    payload = _payload("allow")
    sig = _Sig("sig_c", payload, _signed(payload, sk, ml), [])
    calls: list = []
    first = _gate_with(monkeypatch, sig, _jwks(pk), calls)
    assert first.exit_code == 0, first.output
    assert len(calls) == 1
    # Second run: the cache holds the key, so no fetch happens.
    second = _gate_with(monkeypatch, sig, _jwks(pk), calls)
    assert second.exit_code == 0, second.output
    assert len(calls) == 1
    # A stale cache that lacks the key triggers exactly one refresh.
    cli_hook._write_jwks_cache({"keys": []})
    third = _gate_with(monkeypatch, sig, _jwks(pk), calls)
    assert third.exit_code == 0, third.output
    assert len(calls) == 2


def test_jwks_url_derives_from_the_api_base(monkeypatch) -> None:
    from asqav import client as c

    monkeypatch.setattr(c, "_api_base", "https://api.example.test/api/v1")
    assert cli_hook._jwks_url() == "https://api.example.test/.well-known/jwks.json"
