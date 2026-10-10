"""The PreToolUse gate against the body the real Agent.sign POSTs (criterion 578).

Only the client's network edge is stubbed (_get, _post) plus the JWK Set fetch. The signer
stub signs over the body it actually received, so the receipt is bound to the sent bytes and
never to a digest the gate rebuilds. Offline: a green run says nothing about api.asqav.com.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any

import pytest
from typer.testing import CliRunner

from asqav import cli_hook
from asqav import client as _client
from asqav.cli import app
from asqav.verifier import verify_receipt as vr

runner = CliRunner()

ORG = "org-e2e"
AGENT = "agt_e2e"


@pytest.fixture(autouse=True)
def _identity(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("ASQAV_API_KEY", "sk_test_e2e")
    monkeypatch.setenv("ASQAV_AGENT_ID", AGENT)
    monkeypatch.setenv(cli_hook._JWKS_CACHE_ENV, str(tmp_path / "jwks-cache.json"))
    monkeypatch.delenv(cli_hook._DEADLINE_ENV, raising=False)
    # asqav.init() runs inside the gate and rewrites these module globals.
    for name in ("_api_key", "_api_base", "_client", "_mode", "_org_salt", "_timeout"):
        monkeypatch.setattr(_client, name, getattr(_client, name))


def _ml_dsa_65():
    pytest.importorskip("dilithium_py")
    from dilithium_py.ml_dsa import ML_DSA_65

    return ML_DSA_65


def _digest_as_spelled(monkeypatch, session_id: str) -> str:
    """The hash-only digest over the event exactly as the host spells it (no normalising)."""
    monkeypatch.setattr(_client, "_mode", "hash-only")
    return _client._build_sign_body(
        action_type="tool:Bash",
        context={"tool_input": {"command": "ls"}},
        session_id=session_id,
        agent_id=AGENT,
    )["hash"]


def _flip_last_char(digest: str) -> str:
    return digest[:-1] + ("0" if digest[-1] != "0" else "1")


class _Signer:
    """Stands in for the platform: signs an allow receipt over the body it received."""

    def __init__(
        self, *, alter_digest: bool = False, omit_invocation_ref: bool = False
    ) -> None:
        self.ml = _ml_dsa_65()
        self.pk, self.sk = self.ml.keygen()
        self.thumb = vr.thumbprint_for_key(alg="ML-DSA-65", public_key=self.pk)
        self.alter_digest = alter_digest
        self.omit_invocation_ref = omit_invocation_ref
        self.posted: list[dict[str, Any]] = []

    def jwks(self) -> dict[str, Any]:
        return {
            "keys": [
                {
                    "kid": "k-e2e",
                    "agent_id": AGENT,
                    "issuer_id": ORG,
                    "org_id": ORG,
                    "alg": "ML-DSA-65",
                    "kty": "AKP",
                    "public_key": base64.b64encode(self.pk).decode(),
                    "status": "active",
                    "revoked_at": None,
                    "key_thumbprint": self.thumb,
                }
            ]
        }

    def get(self, path: str, *_a: Any, **_k: Any) -> dict[str, Any]:
        assert path == f"/agents/{AGENT}", path
        return {
            "agent_id": AGENT,
            "name": "e2e",
            "public_key": "pk",
            "key_id": "k-e2e",
            "algorithm": "ml-dsa-65",
            "capabilities": [],
            "created_at": "2026-10-10T00:00:00+00:00",
        }

    def post(self, path: str, body: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
        assert path == f"/agents/{AGENT}/sign", path
        self.posted.append(json.loads(json.dumps(body)))
        digest = body.get("hash", "sha256:" + "5" * 64)
        if self.alter_digest:
            digest = _flip_last_char(digest)
        payload = {
            "type": "protectmcp:decision",
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "issuer_id": ORG,
            "agent_id": AGENT,
            "action_ref": digest,
            "hash": digest,
            "key_thumbprint": self.thumb,
            "policy_digest": "sha256:" + "3" * 64,
            "previousReceiptHash": "0" * 64,
            "decision": "allow",
        }
        # The cloud signs the caller's invocation_ref into the payload, top level.
        if body.get("invocation_ref") is not None and not self.omit_invocation_ref:
            payload["invocation_ref"] = body["invocation_ref"]
        envelope = {
            "alg": "ML-DSA-65",
            "kid": ORG,
            "sig": base64.b64encode(self.ml.sign(self.sk, vr.canonical_json(payload))).decode(),
        }
        self.last_response = {
            "signature": envelope,
            "signature_id": "sig_e2e",
            "action_id": "act_e2e",
            "timestamp": payload["issued_at"],
            "verification_url": "https://example.invalid/verify/sig_e2e",
            "payload": payload,
            "anchors": [],
            "decision": "allow",
            "compliance_mode": True,
        }
        return self.last_response


def _run_gate(monkeypatch, signer: _Signer, event: dict[str, Any], *extra: str):
    monkeypatch.setattr(_client, "_get", signer.get)
    monkeypatch.setattr(_client, "_post", signer.post)
    monkeypatch.setattr(cli_hook, "_fetch_jwks", lambda _t: signer.jwks())
    return runner.invoke(app, ["hook", "pretool", *extra], input=json.dumps(event))


_EVENTS = [
    pytest.param(
        {
            "session_id": "s-trace-1",
            "tool_use_id": "toolu_01",
            "tool_name": "Bash",
            "tool_input": {"command": "ls -la"},
        },
        id="mixed-case-tool-with-trace-id",
    ),
    pytest.param(
        {"tool_name": "Bash", "tool_input": {"command": "ls -la"}},
        id="mixed-case-tool-without-trace-id",
    ),
    pytest.param(
        {
            "session_id": "s-trace-3",
            "tool_use_id": "toolu_03",
            "tool_name": "bash",
            "tool_input": {"command": "ls -la"},
        },
        id="lowercase-tool-with-trace-id",
    ),
    pytest.param(
        {
            "session_id": "s-trace-2",
            "tool_use_id": "toolu_02",
            "tool_name": "Write",
            "tool_input": {"file_path": "/f"},
        },
        id="capitalised-tool-with-trace-id",
    ),
]


@pytest.mark.parametrize("event", _EVENTS)
def test_an_allow_receipt_signed_over_the_posted_body_opens_the_gate(monkeypatch, event) -> None:
    signer = _Signer()
    result = _run_gate(monkeypatch, signer, event)
    assert result.exit_code == 0, result.output
    assert "permit sig_e2e" in result.output
    assert len(signer.posted) == 1, "the stub saw no POST, so nothing real was exercised"


@pytest.mark.parametrize("event", _EVENTS)
def test_the_same_receipt_with_one_byte_of_the_digest_altered_blocks(monkeypatch, event) -> None:
    signer = _Signer(alter_digest=True)
    result = _run_gate(monkeypatch, signer, event)
    assert result.exit_code == 2, result.output
    assert "does not match the digest the gate sent" in result.output
    assert len(signer.posted) == 1


def test_the_posted_body_is_the_lowercased_action_with_the_trace_id_in_the_context(
    monkeypatch,
) -> None:
    signer = _Signer()
    event = {"session_id": "s-trace-1", "tool_name": "Bash", "tool_input": {"command": "ls"}}
    assert _run_gate(monkeypatch, signer, event).exit_code == 0
    sent = signer.posted[0]
    assert sent["action_type"] == "tool:bash"
    assert "trace_id" not in sent
    # The digest is over the normalised action, not the event's own spelling.
    assert sent["hash"] != _digest_as_spelled(monkeypatch, "s-trace-1")


def test_a_receipt_bound_to_the_unnormalised_digest_is_refused(monkeypatch) -> None:
    """A digest over the event as spelled (not as Agent.sign sends it) must not open the gate."""
    event = {"session_id": "s-trace-1", "tool_name": "Bash", "tool_input": {"command": "ls"}}
    signer = _Signer()
    unnormalised = _digest_as_spelled(monkeypatch, "s-trace-1")
    original_post = signer.post

    def post_bound_to_unnormalised(path, body, *a, **k):
        return original_post(path, {**body, "hash": unnormalised}, *a, **k)

    signer.post = post_bound_to_unnormalised  # type: ignore[method-assign]
    result = _run_gate(monkeypatch, signer, event)
    assert result.exit_code == 2, result.output
    assert "does not match the digest the gate sent" in result.output


def test_the_dry_run_prints_the_digest_the_live_hook_posts(monkeypatch) -> None:
    signer = _Signer()
    event = {"session_id": "s-trace-1", "tool_name": "Bash", "tool_input": {"command": "ls"}}
    assert _run_gate(monkeypatch, signer, event).exit_code == 0
    dry = runner.invoke(app, ["hook", "pretool", "--dry-run"], input=json.dumps(event))
    assert dry.exit_code == 0, dry.output
    printed = json.loads(dry.output)
    assert printed["hash"] == signer.posted[0]["hash"]
    assert printed["action_type"] == signer.posted[0]["action_type"]
    assert "trace_id" not in printed


def test_a_receipt_for_another_agent_still_blocks_end_to_end(monkeypatch) -> None:
    signer = _Signer()
    real_post = signer.post

    def post_other_agent(path, body, *a, **k):
        data = real_post(path, body, *a, **k)
        payload = {**data["payload"], "agent_id": "agt_other"}
        data["payload"] = payload
        data["signature"] = {
            **data["signature"],
            "sig": base64.b64encode(
                signer.ml.sign(signer.sk, vr.canonical_json(payload))
            ).decode(),
        }
        return data

    signer.post = post_other_agent  # type: ignore[method-assign]
    event = {"session_id": "s-trace-1", "tool_name": "Bash", "tool_input": {"command": "ls"}}
    result = _run_gate(monkeypatch, signer, event)
    assert result.exit_code == 2, result.output
    assert "agent mismatch" in result.output


# === the receipt must name THIS invocation, and nothing after signing may exit open ===

_CALL = {
    "session_id": "s-replay",
    "tool_use_id": "toolu_A",
    "tool_name": "Bash",
    "tool_input": {"command": "rm -rf build"},
}


def test_a_receipt_minted_for_call_one_does_not_open_an_identical_call_two(monkeypatch) -> None:
    signer = _Signer()
    first = _run_gate(monkeypatch, signer, _CALL)
    assert first.exit_code == 0, first.output
    call_one = json.loads(json.dumps(signer.last_response))
    # Same tool, input and session inside the freshness window, a different invocation:
    # the platform (or whoever holds the response channel) hands back call one's receipt.
    monkeypatch.setattr(signer, "post", lambda *_a, **_k: json.loads(json.dumps(call_one)))
    second = _run_gate(monkeypatch, signer, {**_CALL, "tool_use_id": "toolu_B"})
    assert second.exit_code == 2, second.output
    assert "invocation_ref does not match this tool call" in second.output


def test_a_receipt_without_an_invocation_ref_blocks_when_the_host_named_one(monkeypatch) -> None:
    signer = _Signer(omit_invocation_ref=True)
    result = _run_gate(monkeypatch, signer, _CALL)
    assert result.exit_code == 2, result.output
    assert "invocation_ref does not match this tool call (missing)" in result.output


def test_a_host_that_names_no_invocation_leaves_the_digest_binding_alone(monkeypatch) -> None:
    event = {k: v for k, v in _CALL.items() if k != "tool_use_id"}
    result = _run_gate(monkeypatch, _Signer(), event)
    assert result.exit_code == 0, result.output


class _FlappingDetector:
    """Allows the first inspect() and raises on every later one."""

    name = "flapping"

    def __init__(self) -> None:
        self.calls = 0

    def inspect(self, action_type: str, context: dict[str, Any]):
        from asqav._detectors import DetectorResult

        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("backend flapped on the second run")
        return DetectorResult(allow=True)


@pytest.fixture
def _clean_registries():
    from asqav._detectors import clear_detectors
    from asqav.hooks import clear_hooks

    clear_detectors()
    clear_hooks()
    yield
    clear_detectors()
    clear_hooks()


def test_detectors_and_before_hooks_run_once_and_the_gate_does_not_rebuild(
    monkeypatch, _clean_registries
) -> None:
    from asqav._detectors import register_detector
    from asqav.hooks import register_before

    detector = _FlappingDetector()
    register_detector(detector)
    seen: list[str] = []
    register_before("tool:*", lambda action, ctx: seen.append(action))
    signer = _Signer()
    result = _run_gate(monkeypatch, signer, _CALL)
    assert result.exit_code == 0, result.output
    assert detector.calls == 1, "the gate re-ran the detector after signing"
    assert seen == ["tool:bash"], "the gate re-ran the before-hook after signing"
    assert len(signer.posted) == 1


def test_a_detector_that_raises_on_the_signing_run_blocks_with_exit_two(
    monkeypatch, _clean_registries
) -> None:
    from asqav._detectors import register_detector

    detector = _FlappingDetector()
    detector.calls = 1
    register_detector(detector)
    result = _run_gate(monkeypatch, _Signer(), _CALL)
    assert result.exit_code == 2, result.output


def test_any_error_after_signing_exits_two_never_one(monkeypatch) -> None:
    def broken(_sig):
        raise RuntimeError("receipt shape surprised the gate")

    monkeypatch.setattr(cli_hook, "_wire_decision", broken)
    result = _run_gate(monkeypatch, _Signer(), _CALL)
    assert result.exit_code == 2, result.output
    assert "blocking tool call" in result.output


def test_clear_context_blocks_because_the_real_body_carries_no_digest(monkeypatch) -> None:
    result = _run_gate(monkeypatch, _Signer(), _CALL, "--clear-context")
    assert result.exit_code == 2, result.output
    assert "does not match the digest the gate sent" in result.output
