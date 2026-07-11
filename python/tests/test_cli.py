"""Tests for the Asqav CLI."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typer.testing import CliRunner

from asqav import __version__
from asqav.cli import app

runner = CliRunner()


# === Version ===


def test_version() -> None:
    """--version prints the SDK version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "asqav" in result.output
    assert __version__ in result.output


# === verify command ===


@patch("asqav.verify_signature")
def test_verify_valid(mock_verify: MagicMock) -> None:
    """verify prints details for a valid signature."""
    mock_verify.return_value = MagicMock(
        verified=True,
        agent_name="test-agent",
        agent_id="agent_123",
        action_type="api:call",
        algorithm="ml-dsa-65",
    )
    result = runner.invoke(app, ["verify", "sig_abc"])
    assert result.exit_code == 0
    assert "valid" in result.output
    assert "test-agent" in result.output
    assert "agent_123" in result.output
    assert "api:call" in result.output
    mock_verify.assert_called_once_with("sig_abc")


@patch("asqav.verify_signature")
def test_verify_invalid(mock_verify: MagicMock) -> None:
    """verify prints 'invalid' when signature is not verified."""
    mock_verify.return_value = MagicMock(
        verified=False,
        agent_name="bad-agent",
        agent_id="agent_999",
        action_type="write:data",
        algorithm="ml-dsa-44",
    )
    result = runner.invoke(app, ["verify", "sig_bad"])
    assert result.exit_code == 0
    assert "invalid" in result.output
    assert "bad-agent" in result.output


@patch("asqav.verify_signature")
def test_verify_not_found(mock_verify: MagicMock) -> None:
    """verify exits with code 1 for non-existent signature."""
    from asqav import APIError

    mock_verify.side_effect = APIError("Signature not found", 404)
    result = runner.invoke(app, ["verify", "sig_missing"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


@patch("asqav.verify_signature")
def test_verify_network_failure_readable(mock_verify: MagicMock) -> None:
    """A transport failure prints one readable line and exits nonzero, never a
    raw httpx traceback (the command caught only APIError before)."""
    import httpx

    mock_verify.side_effect = httpx.ConnectError("Connection refused")
    result = runner.invoke(app, ["verify", "sig_x"])
    assert result.exit_code == 1
    assert "could not reach" in result.output.lower()


# === sign command (IETF Compliance Receipts profile) ===


@patch("asqav.Agent.get")
@patch("asqav.init")
def test_sign_compliance_mode_calls_sdk(
    mock_init: MagicMock, mock_get: MagicMock, tmp_path
) -> None:
    """sign --compliance-mode threads every IETF flag into Agent.sign()."""
    fake_agent = MagicMock()
    fake_agent.agent_id = "agent_x"
    fake_agent.algorithm = "ml-dsa-65"
    fake_agent.sign.return_value = MagicMock(
        signature_id="sig_abc",
        action_id="act_1",
        algorithm="ml-dsa-65",
        timestamp=1717.0,
        verification_url="https://verify.example/sig_abc",
        policy_digest="sha256:cafe",
        policy_decision="permit",
        compliance_mode=True,
        receipt_type="protectmcp:decision",
        action_ref="sha256:beef",
        issuer_id="legal:Acme",
        previous_receipt_hash="0" * 64,
        bitcoin_anchor=None,
        rfc3161_tsa=None,
        rfc3161_serial=None,
    )
    mock_get.return_value = fake_agent

    action_path = tmp_path / "action.json"
    action_path.write_text('{"amount_eur": 850000}')

    result = runner.invoke(
        app,
        [
            "sign",
            "--agent-id", "agent_x",
            "--action-type", "payment.wire_transfer",
            "--action-json", str(action_path),
            "--compliance-mode",
            "--receipt-type", "protectmcp:decision",
            "--risk-class", "high",
            "--issuer-id", "legal:Acme",
            "--iteration-id", "task-2026-Q2",
            "--sandbox-state", "enabled",
        ],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0, result.output
    assert "sig_abc" in result.output
    assert "compliance_mode: True" in result.output
    assert "receipt_type:  protectmcp:decision" in result.output
    fake_agent.sign.assert_called_once()
    call_kwargs = fake_agent.sign.call_args.kwargs
    assert call_kwargs["compliance_mode"] is True
    assert call_kwargs["receipt_type"] == "protectmcp:decision"
    assert call_kwargs["risk_class"] == "high"
    assert call_kwargs["issuer_id"] == "legal:Acme"
    assert call_kwargs["iteration_id"] == "task-2026-Q2"
    assert call_kwargs["sandbox_state"] == "enabled"


@patch("asqav.Agent.get")
@patch("asqav.init")
def test_sign_deny_without_reason_rejects(
    mock_init: MagicMock, mock_get: MagicMock
) -> None:
    """sign --policy-decision deny without --reason exits non-zero."""
    result = runner.invoke(
        app,
        [
            "sign",
            "--agent-id", "agent_x",
            "--action-type", "x.y",
            "--policy-decision", "deny",
        ],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 1
    assert "--reason is required" in result.output


# === sign command: edge-case branches ===


@patch("asqav.init")
def test_sign_action_json_invalid_exits_1(
    mock_init: MagicMock, tmp_path
) -> None:
    """A malformed --action-json file is reported with a clear error."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    result = runner.invoke(
        app,
        [
            "sign",
            "--agent-id", "agent_x",
            "--action-type", "payment.wire_transfer",
            "--action-json", str(bad),
        ],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 1
    assert "Error reading action JSON" in result.output


@patch("asqav.Agent.get")
@patch("asqav.init")
def test_sign_agent_get_api_error_exits_1(
    mock_init: MagicMock, mock_get: MagicMock
) -> None:
    """Agent.get APIError surfaces as 'Error fetching agent' and exit 1."""
    from asqav import APIError

    mock_get.side_effect = APIError("agent revoked", 404)
    result = runner.invoke(
        app,
        [
            "sign",
            "--agent-id", "agent_missing",
            "--action-type", "x.y",
        ],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 1
    assert "Error fetching agent" in result.output


@patch("asqav.Agent.get")
@patch("asqav.init")
def test_sign_json_output_with_policy_artefact_and_algo_mismatch(
    mock_init: MagicMock, mock_get: MagicMock, tmp_path
) -> None:
    """Exercise --output json + --policy-artefact + algorithm mismatch + --session-id.

    Covers: JSON renderer, _load_json_arg file branch, _build_sign_context
    policy_artefact wiring, the algorithm-mismatch warning, and the
    --session-id agent attribute injection.
    """
    import json

    fake_agent = MagicMock()
    fake_agent.agent_id = "agent_x"
    fake_agent.algorithm = "ml-dsa-65"
    fake_agent.sign.return_value = MagicMock(
        signature_id="sig_json",
        action_id="act_json",
        algorithm="ml-dsa-65",
        timestamp=1717.0,
        verification_url="https://verify.example/sig_json",
        policy_digest="sha256:abc",
        policy_decision="permit",
        compliance_mode=False,
        receipt_type="protectmcp:decision",
        action_ref=None,
        issuer_id=None,
        previous_receipt_hash=None,
        bitcoin_anchor=MagicMock(status="pending"),
        rfc3161_tsa="tsa.example",
        rfc3161_serial="123",
    )
    mock_get.return_value = fake_agent

    policy_path = tmp_path / "policy.json"
    policy_path.write_text('{"rule": "deny_over_500k"}')

    result = runner.invoke(
        app,
        [
            "sign",
            "--agent-id", "agent_x",
            "--action-type", "payment.wire_transfer",
            "--policy-artefact", str(policy_path),
            "--session-id", "sess_bound",
            "--algorithm", "ed25519",  # mismatch
            "--output", "json",
        ],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0, result.output
    assert "Warning: --algorithm=ed25519" in result.output
    # session_id was threaded onto the agent before sign()
    assert fake_agent._session_id == "sess_bound"
    # context wraps the policy artefact under _policy_artefact
    ctx = fake_agent.sign.call_args.kwargs["context"]
    assert ctx["_policy_artefact"] == {"rule": "deny_over_500k"}
    # JSON output is parseable and carries anchor + RFC 3161 fields
    json_start = result.output.find("{")
    payload = json.loads(result.output[json_start:])
    assert payload["signature_id"] == "sig_json"
    assert payload["anchor_status"] == "pending"
    assert payload["rfc3161_tsa"] == "tsa.example"


# === sign command helpers (unit tests, no CLI runner) ===


def test_load_json_arg_empty_returns_none() -> None:
    from asqav.cli import _load_json_arg

    assert _load_json_arg("", "any") is None


def test_build_sign_kwargs_drops_empty_strings() -> None:
    from asqav.cli import _build_sign_kwargs

    out = _build_sign_kwargs(
        compliance_mode=True,
        policy_decision="permit",
        action_ref="",
        sandbox_state="enabled",
        iteration_id="",
        risk_class="high",
        incident_class="",
        issuer_id="",
        receipt_type="protectmcp:decision",
        reason="",
        valid_seconds=0,
    )
    assert out == {
        "compliance_mode": True,
        "policy_decision": "permit",
        "sandbox_state": "enabled",
        "risk_class": "high",
        "receipt_type": "protectmcp:decision",
    }


def test_build_sign_kwargs_includes_valid_seconds_when_positive() -> None:
    from asqav.cli import _build_sign_kwargs

    out = _build_sign_kwargs(
        compliance_mode=False,
        policy_decision="permit",
        action_ref="",
        sandbox_state="",
        iteration_id="",
        risk_class="",
        incident_class="",
        issuer_id="",
        receipt_type="",
        reason="",
        valid_seconds=60,
    )
    assert out["valid_seconds"] == 60


def test_build_sign_context_merges_action_id_and_policy() -> None:
    from asqav.cli import _build_sign_context

    out = _build_sign_context(
        {"amount": 1000},
        "act_123",
        {"rule": "deny"},
    )
    assert out == {
        "amount": 1000,
        "_action_id": "act_123",
        "_policy_artefact": {"rule": "deny"},
    }


def test_build_sign_context_returns_none_when_empty() -> None:
    from asqav.cli import _build_sign_context

    assert _build_sign_context({}, "", None) is None


def test_build_sign_context_only_action_id() -> None:
    """`--action-id` without --action-json produces a context with just the id."""
    from asqav.cli import _build_sign_context

    assert _build_sign_context({}, "act_only", None) == {"_action_id": "act_only"}


@patch("asqav.verify_signature")
def test_verify_text_shows_ietf_axes(mock_verify: MagicMock) -> None:
    """verify --output text surfaces IETF profile sub-axes when present."""
    from asqav import VerificationDetail

    mock_verify.return_value = MagicMock(
        verified=True,
        agent_name="ietf-agent",
        agent_id="agent_ietf",
        action_type="payment.wire_transfer",
        algorithm="ml-dsa-65",
        verification_detail=VerificationDetail(
            signer_key_match=True,
            signature_valid=True,
            algorithm_match=True,
            agent_active=True,
            validation_label="valid",
            signed_at_skew_seconds=1.234,
            chain_valid=True,
            anchor_status_ots="pending",
            anchor_status_rfc3161="valid",
            missing_fields=None,
        ),
    )
    result = runner.invoke(app, ["verify", "sig_ietf"])
    assert result.exit_code == 0
    assert "Chain valid: True" in result.output
    assert "Anchor (OTS): pending" in result.output
    assert "Anchor (RFC 3161): valid" in result.output
    assert "skew: 1.234" in result.output


@patch("asqav.verify_signature")
def test_verify_json_emits_full_detail(mock_verify: MagicMock) -> None:
    """verify --output json returns the full VerificationResponse."""
    import json

    from asqav import VerificationDetail

    mock_verify.return_value = MagicMock(
        signature_id="sig_xyz",
        agent_id="agent_x",
        agent_name="agent-x",
        action_id="act_1",
        action_type="api:call",
        algorithm="ml-dsa-65",
        signed_at=1717171717.0,
        verified=True,
        verification_url="https://verify.example/sig_xyz",
        verification_detail=VerificationDetail(
            signer_key_match=True,
            signature_valid=True,
            algorithm_match=True,
            agent_active=True,
            validation_label="valid",
            signed_at_skew_seconds=0.5,
            chain_valid=True,
            anchor_status_ots="valid",
            anchor_status_rfc3161="valid",
            missing_fields=[],
        ),
    )
    result = runner.invoke(app, ["verify", "sig_xyz", "--output", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["signature_id"] == "sig_xyz"
    assert payload["verification_detail"]["chain_valid"] is True
    assert payload["verification_detail"]["anchor_status_ots"] == "valid"
    assert payload["verification_detail"]["signed_at_skew_seconds"] == 0.5


# === agents list command ===


@patch("asqav.client._get")
@patch("asqav.init")
def test_agents_list(mock_init: MagicMock, mock_get: MagicMock) -> None:
    """agents list prints agents when present."""
    mock_get.return_value = {
        "agents": [
            {"name": "alpha", "agent_id": "agent_001", "algorithm": "ml-dsa-65"},
            {"name": "beta", "agent_id": "agent_002", "algorithm": "ml-dsa-87"},
        ]
    }
    result = runner.invoke(app, ["agents", "list"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "agent_001" in result.output
    assert "beta" in result.output
    mock_init.assert_called_once_with(api_key="sk_test")


@patch("asqav.client._get")
@patch("asqav.init")
def test_agents_list_empty(mock_init: MagicMock, mock_get: MagicMock) -> None:
    """agents list prints message when no agents exist."""
    mock_get.return_value = {"agents": []}
    result = runner.invoke(app, ["agents", "list"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 0
    assert "No agents found" in result.output


@patch("asqav.client._get")
@patch("asqav.init")
def test_agents_list_handles_array_response(
    mock_init: MagicMock, mock_get: MagicMock
) -> None:
    """The cloud returns a bare JSON array; the CLI must render it directly."""
    mock_get.return_value = [
        {"name": "alpha", "agent_id": "agent_001", "algorithm": "ml-dsa-65"},
        {"name": "beta", "agent_id": "agent_002", "algorithm": "ed25519"},
    ]
    result = runner.invoke(app, ["agents", "list"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "agent_001" in result.output
    assert "beta" in result.output
    assert "agent_002" in result.output


@patch("asqav.client._get")
@patch("asqav.init")
def test_agents_list_empty_array(mock_init: MagicMock, mock_get: MagicMock) -> None:
    """An empty array response must produce the friendly empty message."""
    mock_get.return_value = []
    result = runner.invoke(app, ["agents", "list"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 0
    assert "No agents found" in result.output


def test_agents_list_no_api_key() -> None:
    """agents list exits with error when ASQAV_API_KEY is not set."""
    result = runner.invoke(app, ["agents", "list"], env={"ASQAV_API_KEY": ""})
    assert result.exit_code == 1
    assert "ASQAV_API_KEY" in result.output


# === agents create command ===


@patch("asqav.Agent.create")
@patch("asqav.init")
def test_agents_create(mock_init: MagicMock, mock_create: MagicMock) -> None:
    """agents create prints the new agent details."""
    mock_agent = MagicMock(
        name="my-agent",
        agent_id="agent_new",
        algorithm="ml-dsa-65",
        key_id="key_abc",
    )
    mock_create.return_value = mock_agent
    result = runner.invoke(
        app, ["agents", "create", "my-agent"], env={"ASQAV_API_KEY": "sk_test"}
    )
    assert result.exit_code == 0
    assert "my-agent" in result.output
    assert "agent_new" in result.output
    assert "ml-dsa-65" in result.output
    assert "key_abc" in result.output
    mock_create.assert_called_once_with("my-agent")


# === quickstart command ===


def test_quickstart_with_api_key() -> None:
    """quickstart shows success when API key is set."""
    result = runner.invoke(app, ["quickstart"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 0
    assert "API key detected" in result.output
    assert "asqav hook" in result.output
    assert "Policy" in result.output
    assert "Next steps" in result.output


def test_quickstart_without_api_key() -> None:
    """quickstart warns when API key is missing."""
    result = runner.invoke(app, ["quickstart"], env={"ASQAV_API_KEY": ""})
    assert result.exit_code == 0
    assert "ASQAV_API_KEY not set" in result.output
    assert "asqav hook" in result.output


# === doctor command ===


@patch("urllib.request.urlopen")
def test_doctor_no_api_key(mock_urlopen: MagicMock) -> None:
    """doctor reports failure when API key is missing."""
    result = runner.invoke(app, ["doctor"], env={"ASQAV_API_KEY": ""})
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "ASQAV_API_KEY not set" in result.output
    assert "SKIP" in result.output


@patch("asqav.client._urllib_request")
@patch("urllib.request.urlopen")
@patch("asqav.client.health_check")
@patch("asqav.init")
def test_doctor_all_pass(
    mock_init: MagicMock,
    mock_health: MagicMock,
    mock_urlopen: MagicMock,
    mock_urllib_req: MagicMock,
) -> None:
    """doctor passes all checks when API key is set and API is reachable."""
    mock_health.return_value = {"status": "ok"}
    mock_urllib_req.return_value = []
    result = runner.invoke(app, ["doctor"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 0
    assert "PASS" in result.output
    assert "API key set" in result.output
    assert "API reachable" in result.output
    assert "URL join preserves API base path" in result.output
    assert "API edge accepts this client" in result.output
    assert "All checks passed" in result.output
    mock_urllib_req.assert_called_once_with("GET", "/agents")


@patch("asqav.client._urllib_request")
@patch("urllib.request.urlopen")
@patch("asqav.client.health_check")
@patch("asqav.init")
def test_doctor_url_join_broken_fails(
    mock_init: MagicMock,
    mock_health: MagicMock,
    mock_urlopen: MagicMock,
    mock_urllib_req: MagicMock,
) -> None:
    """doctor FAILS when the stdlib fallback join 404s a real endpoint.

    Pins the stranger-funnel regression where /health passed (it exists at
    the host root too) while every prefixed call died on 404.
    """
    from asqav.client import APIError

    mock_health.return_value = {"status": "ok"}
    mock_urllib_req.side_effect = APIError("HTTP Error 404: Not Found", 404)
    result = runner.invoke(app, ["doctor"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 1
    assert "URL join drops the API base path" in result.output
    assert "All checks passed" not in result.output


@patch("asqav.client._urllib_request", return_value=[])
@patch("urllib.request.urlopen")
@patch("asqav.client.health_check")
@patch("asqav.init")
def test_doctor_api_unreachable(
    mock_init: MagicMock,
    mock_health: MagicMock,
    mock_urlopen: MagicMock,
    mock_urllib_req: MagicMock,
) -> None:
    """doctor reports failure when API is unreachable."""
    mock_health.side_effect = Exception("Connection refused")
    result = runner.invoke(app, ["doctor"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 1
    assert "API unreachable" in result.output
    assert "Connection refused" in result.output


@patch("asqav.client._urllib_request", return_value=[])
@patch("urllib.request.urlopen")
@patch("asqav.client.health_check")
@patch("asqav.init")
def test_doctor_edge_blocked_403(
    mock_init: MagicMock,
    mock_health: MagicMock,
    mock_urlopen: MagicMock,
    mock_urllib_req: MagicMock,
) -> None:
    """doctor FAILS loudly when the API edge 403-blocks this client.

    Pins the stranger-funnel regression: the key-authenticated health check
    can succeed while the edge firewall (Cloudflare error 1010) rejects the
    client and every sign call dies. doctor must not print all-pass then.
    """
    import urllib.error

    mock_health.return_value = {"status": "ok"}
    mock_urlopen.side_effect = urllib.error.HTTPError(
        "https://api.asqav.com/api/v1/health", 403, "Forbidden", None, None
    )
    result = runner.invoke(app, ["doctor"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 1
    assert "BLOCKED" in result.output
    assert "Sign calls will fail" in result.output
    assert "All checks passed" not in result.output


@patch("asqav.client._urllib_request", return_value=[])
@patch("urllib.request.urlopen")
@patch("asqav.client.health_check")
@patch("asqav.init")
def test_doctor_edge_unreachable(
    mock_init: MagicMock,
    mock_health: MagicMock,
    mock_urlopen: MagicMock,
    mock_urllib_req: MagicMock,
) -> None:
    """doctor reports a clear failure when the edge probe cannot connect."""
    import urllib.error

    mock_health.return_value = {"status": "ok"}
    mock_urlopen.side_effect = urllib.error.URLError("no route to host")
    result = runner.invoke(app, ["doctor"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 1
    assert "API edge unreachable" in result.output
    assert "All checks passed" not in result.output


@patch("urllib.request.urlopen")
def test_doctor_edge_probe_sends_user_agent(mock_urlopen: MagicMock) -> None:
    """The edge probe identifies itself with the SDK User-Agent."""
    from asqav._useragent import USER_AGENT

    runner.invoke(app, ["doctor"], env={"ASQAV_API_KEY": ""})
    assert mock_urlopen.called
    req = mock_urlopen.call_args[0][0]
    assert req.get_header("User-agent") == USER_AGENT


# === replay command ===


def _fake_timeline(*, chain_integrity: bool = True) -> MagicMock:
    """Build a ReplayTimeline-like double for CLI tests."""
    tl = MagicMock()
    tl.agent_id = "agt_test"
    tl.session_id = "sess_test"
    tl.chain_integrity = chain_integrity
    tl.summary.return_value = "  Steps:\n    [0] api:call (chain: ok)"
    tl.to_json.return_value = '{"agent_id":"agt_test","session_id":"sess_test"}'
    return tl


def test_replay_requires_agent_and_session() -> None:
    """replay without agent_id+session_id and without --bundle exits 1."""
    result = runner.invoke(app, ["replay"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 1
    assert "agent_id and session_id are required" in result.output


@patch("asqav.replay")
@patch("asqav.init")
def test_replay_online_passes_chain(mock_init: MagicMock, mock_replay: MagicMock) -> None:
    """replay <agent> <session> calls asqav.replay and prints the summary."""
    mock_replay.return_value = _fake_timeline(chain_integrity=True)
    result = runner.invoke(
        app,
        ["replay", "agt_test", "sess_test"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "Steps:" in result.output
    mock_replay.assert_called_once_with("agt_test", "sess_test")


@patch("asqav.replay")
@patch("asqav.init")
def test_replay_chain_broken_exits_1(mock_init: MagicMock, mock_replay: MagicMock) -> None:
    """replay exits non-zero when the chain failed verification."""
    mock_replay.return_value = _fake_timeline(chain_integrity=False)
    result = runner.invoke(
        app,
        ["replay", "agt_test", "sess_test"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 1


@patch("asqav.replay")
@patch("asqav.init")
def test_replay_json_flag(mock_init: MagicMock, mock_replay: MagicMock) -> None:
    """--json prints to_json output."""
    mock_replay.return_value = _fake_timeline()
    result = runner.invoke(
        app,
        ["replay", "agt_test", "sess_test", "--json"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "agt_test" in result.output


def test_replay_bundle_missing_file() -> None:
    """--bundle pointing to a nonexistent file exits 1."""
    result = runner.invoke(app, ["replay", "--bundle", "/nonexistent/path.json"])
    assert result.exit_code == 1
    assert "Error reading bundle" in result.output


# === preflight command ===


@patch("asqav.Agent.get")
@patch("asqav.init")
def test_preflight_cleared(mock_init: MagicMock, mock_get: MagicMock) -> None:
    """preflight prints CLEARED and exits 0 when the action is allowed."""
    mock_agent = MagicMock()
    mock_agent.preflight.return_value = MagicMock(
        cleared=True, agent_active=True, policy_allowed=True,
        reasons=[], explanation="ok",
    )
    mock_get.return_value = mock_agent
    result = runner.invoke(
        app, ["preflight", "agt_x", "data:read"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "CLEARED" in result.output


@patch("asqav.Agent.get")
@patch("asqav.init")
def test_preflight_blocked(mock_init: MagicMock, mock_get: MagicMock) -> None:
    """preflight exits 1 and prints reasons when blocked."""
    mock_agent = MagicMock()
    mock_agent.preflight.return_value = MagicMock(
        cleared=False, agent_active=False, policy_allowed=True,
        reasons=["agent is suspended"], explanation="blocked",
    )
    mock_get.return_value = mock_agent
    result = runner.invoke(
        app, ["preflight", "agt_x", "write:data"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 1
    assert "BLOCKED" in result.output
    assert "agent is suspended" in result.output


# === budget commands ===


@patch("asqav.Agent.get")
@patch("asqav.init")
def test_budget_check_allowed(mock_init: MagicMock, mock_get: MagicMock) -> None:
    """budget check exits 0 when within limit."""
    mock_get.return_value = MagicMock()
    result = runner.invoke(
        app,
        ["budget", "check", "--agent-id", "agt_x", "--limit", "10",
         "--estimated-cost", "1.5", "--current-spend", "2.0"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "ALLOWED" in result.output


@patch("asqav.Agent.get")
@patch("asqav.init")
def test_budget_check_exhausted(mock_init: MagicMock, mock_get: MagicMock) -> None:
    """budget check exits 1 when estimated_cost would exceed limit."""
    mock_get.return_value = MagicMock()
    result = runner.invoke(
        app,
        ["budget", "check", "--agent-id", "agt_x", "--limit", "10",
         "--estimated-cost", "20", "--current-spend", "0"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 1
    assert "DENIED" in result.output
    assert "budget_exhausted" in result.output


@patch("asqav.BudgetTracker.record")
@patch("asqav.Agent.get")
@patch("asqav.init")
def test_budget_record(
    mock_init: MagicMock, mock_get: MagicMock, mock_record: MagicMock,
) -> None:
    """budget record signs a spend record and prints the signature_id."""
    mock_get.return_value = MagicMock()
    mock_record.return_value = MagicMock(
        signature_id="sig_b1", action_id="act_b1",
        verification_url="https://asqav.com/verify/sig_b1",
    )
    result = runner.invoke(
        app,
        ["budget", "record", "--agent-id", "agt_x", "--action", "api:openai",
         "--actual-cost", "0.05", "--limit", "10"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "sig_b1" in result.output
    mock_record.assert_called_once()


# === approve command ===


@patch("asqav.approve_action")
@patch("asqav.init")
def test_approve_approved(mock_init: MagicMock, mock_approve: MagicMock) -> None:
    """approve prints APPROVED when the session is fully approved."""
    mock_approve.return_value = MagicMock(
        session_id="thr_a", entity_id="ent_a",
        signatures_collected=2, approvals_required=2,
        status="approved", approved=True,
    )
    result = runner.invoke(
        app, ["approve", "thr_a", "ent_a"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "APPROVED" in result.output


# === compliance commands ===


def test_compliance_frameworks_lists_known() -> None:
    """compliance frameworks lists known framework keys."""
    result = runner.invoke(app, ["compliance", "frameworks"])
    assert result.exit_code == 0
    # FRAMEWORKS dict from compliance.py contains at least these.
    assert "eu_ai_act" in result.output
    assert "dora" in result.output


@patch("asqav.compliance.export_bundle")
@patch("asqav.client.get_session_signatures")
@patch("asqav.init")
def test_compliance_export_writes_bundle(
    mock_init: MagicMock, mock_get_sigs: MagicMock, mock_export: MagicMock,
    tmp_path,
) -> None:
    """compliance export pulls signatures and writes a bundle file."""
    mock_get_sigs.return_value = [{"signature_id": "s1"}]
    mock_bundle = MagicMock(receipt_count=1, merkle_root="abc123")
    mock_export.return_value = mock_bundle
    out = tmp_path / "bundle.json"

    result = runner.invoke(
        app,
        ["compliance", "export", "--session", "sess_a", "--output", str(out)],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "1 receipts" in result.output
    assert "abc123" in result.output
    mock_bundle.to_file.assert_called_once_with(str(out))


@patch("asqav.client.get_session_signatures")
@patch("asqav.init")
def test_compliance_export_unknown_framework(
    mock_init: MagicMock, mock_get_sigs: MagicMock,
) -> None:
    """compliance export exits 1 when the framework key is not known."""
    mock_get_sigs.return_value = []
    result = runner.invoke(
        app,
        ["compliance", "export", "--session", "sess_a",
         "--framework", "nonexistent", "--output", "/tmp/x.json"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 1
    assert "nonexistent" in result.output


# === Free-tier access (every CLI command is free) ===


@patch("asqav.Agent.get")
@patch("asqav.client._get")
@patch("asqav.init")
def test_preflight_runs_on_free_tier_without_gate(
    mock_init: MagicMock, mock_get_acc: MagicMock, mock_agent_get: MagicMock,
) -> None:
    """A free-tier key runs preflight to completion with no upgrade block.

    The CLI does not gate commands by tier, so a free-tier org reaches the
    command body. The /account endpoint must not be consulted as a gate.
    """
    mock_get_acc.return_value = {"tier": "free", "organization_id": "o", "organization_name": "n"}
    mock_agent = MagicMock()
    mock_agent.preflight.return_value = MagicMock(
        cleared=True, agent_active=True, policy_allowed=True,
        reasons=[], explanation="ok",
    )
    mock_agent_get.return_value = mock_agent
    result = runner.invoke(
        app,
        ["preflight", "agt_x", "data:read"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "Upgrade" not in result.output
    # The command must not call /account to decide access.
    for call in mock_get_acc.call_args_list:
        assert call.args[:1] != ("/account",)


@patch("asqav.client._get")
@patch("asqav.init")
def test_policies_list_runs_on_free_tier_without_gate(
    mock_init: MagicMock, mock_get: MagicMock,
) -> None:
    """policies list (a former Pro command) is reachable on the free tier."""
    mock_get.return_value = []
    result = runner.invoke(
        app,
        ["policies", "list"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "Upgrade" not in result.output
    mock_get.assert_called_once_with("/policies")


# === CLI gap-fill (agents revoke, sessions, policies, webhooks) ===


@patch("asqav.Agent.get")
@patch("asqav.init")
def test_agents_revoke_calls_agent_revoke(
    mock_init: MagicMock, mock_get: MagicMock
) -> None:
    mock_agent = MagicMock()
    mock_get.return_value = mock_agent
    result = runner.invoke(
        app,
        ["agents", "revoke", "agt_z", "--reason", "leak"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    mock_agent.revoke.assert_called_once_with(reason="leak")
    assert "Revoked agt_z" in result.output


@patch("asqav.list_sessions")
@patch("asqav.init")
def test_sessions_list_prints_rows(
    mock_init: MagicMock, mock_list: MagicMock
) -> None:
    mock_list.return_value = [
        MagicMock(session_id="sess_1", agent_id="agt_a", status="active", started_at="2026-05-02T00:00:00Z"),
    ]
    result = runner.invoke(
        app,
        ["sessions", "list", "--limit", "5"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "sess_1" in result.output
    assert "active" in result.output


@patch("asqav.client._patch")
@patch("asqav.init")
def test_sessions_end_calls_patch(
    mock_init: MagicMock, mock_patch: MagicMock
) -> None:
    mock_patch.return_value = {}
    result = runner.invoke(
        app,
        ["sessions", "end", "sess_x", "--status", "completed"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "sess_x" in result.output
    mock_patch.assert_called_once_with("/sessions/sess_x", {"status": "completed"})


@patch("asqav.client._get")
@patch("asqav.init")
def test_policies_list_prints_active_flag(
    mock_init: MagicMock, mock_get: MagicMock
) -> None:
    def side_effect(path: str):
        if path == "/account":
            return {"tier": "pro", "organization_id": "o", "organization_name": "n"}
        if path == "/policies":
            return [{"id": "pol_1", "name": "no-data", "action_pattern": "data:*", "action": "block", "is_active": True}]
        return None

    mock_get.side_effect = side_effect
    result = runner.invoke(app, ["policies", "list"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 0
    assert "pol_1" in result.output
    assert "active" in result.output


@patch("asqav.client._post")
@patch("asqav.client._get")
@patch("asqav.init")
def test_policies_create_posts_payload(
    mock_init: MagicMock, mock_get: MagicMock, mock_post: MagicMock
) -> None:
    mock_get.return_value = {"tier": "pro", "organization_id": "o", "organization_name": "n"}
    mock_post.return_value = {"id": "pol_new"}
    result = runner.invoke(
        app,
        ["policies", "create", "--name", "test-rule", "--pattern", "data:*", "--action", "block"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    mock_post.assert_called_once_with(
        "/policies",
        {"name": "test-rule", "action_pattern": "data:*", "action": "block", "is_active": True},
    )


@patch("asqav.client._delete")
@patch("asqav.client._get")
@patch("asqav.init")
def test_webhooks_delete_calls_delete(
    mock_init: MagicMock, mock_get: MagicMock, mock_delete: MagicMock
) -> None:
    mock_get.return_value = {"tier": "pro", "organization_id": "o", "organization_name": "n"}
    mock_delete.return_value = {}
    result = runner.invoke(
        app, ["webhooks", "delete", "wh_x"], env={"ASQAV_API_KEY": "sk_test"}
    )
    assert result.exit_code == 0
    mock_delete.assert_called_once_with("/webhooks/wh_x")


# === audit-pack export / policy ===


@patch("asqav.client._post")
@patch("asqav.init")
def test_audit_pack_export_writes_bundle(
    mock_init: MagicMock, mock_post: MagicMock, tmp_path
) -> None:
    """audit-pack export writes the cloud-signed bundle to disk."""
    mock_post.return_value = {
        "bundle_digest": "sha256:abc",
        "bundle_signature": "BASE64SIG==",
        "bundle_signature_algorithm": "ml-dsa-65",
        "bundle_public_key": "BASE64PUB==",
        "algorithm_registry_version": "2026-05-04.v1",
        "receipt_count": 3,
        "start": "2026-04-01T00:00:00Z",
        "end": "2026-05-01T00:00:00Z",
        "receipts": [{"signature_id": "sig_1"}],
    }
    out = tmp_path / "bundle.json"
    result = runner.invoke(
        app,
        [
            "audit-pack", "export",
            "--start", "2026-04-01T00:00:00Z",
            "--end", "2026-05-01T00:00:00Z",
            "--output-file", str(out),
        ],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "wrote 3 receipts" in result.output
    body = mock_post.call_args.args[1]
    assert body["start"] == "2026-04-01T00:00:00Z"
    assert body["only_compliance"] is True


@patch("asqav.client._get")
@patch("asqav.init")
def test_audit_pack_policy_resolves_digest(
    mock_init: MagicMock, mock_get: MagicMock
) -> None:
    """audit-pack policy returns the artefact JSON. 64-hex prefixes sha256:."""
    mock_get.return_value = {
        "digest": "sha256:" + "a" * 64,
        "organization_id": "org_x",
        "agent_id": "agent_x",
        "action_type": "payment.wire_transfer",
        "artefact": {"rule": "deny_over_500k"},
        "created_at": "2026-04-01T00:00:00Z",
    }
    result = runner.invoke(
        app,
        ["audit-pack", "policy", "a" * 64, "--output", "json"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "deny_over_500k" in result.output
    called_path = mock_get.call_args.args[0]
    assert called_path == f"/audit-pack/policy/sha256:{'a' * 64}"


# === payloads erase ===


@patch("asqav.client._delete")
@patch("asqav.init")
def test_payloads_erase_calls_delete(
    mock_init: MagicMock, mock_delete: MagicMock
) -> None:
    """payloads erase --yes calls DELETE /signatures/payloads/{id}."""
    mock_delete.return_value = {}
    result = runner.invoke(
        app,
        ["payloads", "erase", "sig_abc", "--yes"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "Payload erased" in result.output
    mock_delete.assert_called_once_with("/signatures/payloads/sig_abc")


# === org set-compliance-strict ===


@patch("asqav.client._patch")
@patch("asqav.init")
def test_org_set_compliance_strict_enable(
    mock_init: MagicMock, mock_patch: MagicMock
) -> None:
    mock_patch.return_value = {"id": "org_x", "compliance_mode_strict": True}
    result = runner.invoke(
        app,
        ["org", "set-compliance-strict", "org_x", "--enable"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0
    assert "compliance_mode_strict enabled" in result.output
    body = mock_patch.call_args.args[1]
    assert body == {"compliance_mode_strict": True}


def test_org_set_compliance_strict_requires_one_flag() -> None:
    result = runner.invoke(
        app,
        ["org", "set-compliance-strict", "org_x"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 1


# === keys generate ===


def test_keys_generate_ed25519(tmp_path) -> None:
    """keys generate --algorithm ed25519 emits a PKCS#8 PEM private key."""
    pytest_skipif_no_crypto = False
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: F401
    except ImportError:
        pytest_skipif_no_crypto = True
    if pytest_skipif_no_crypto:
        return
    out = tmp_path / "ed25519.pem"
    result = runner.invoke(
        app,
        ["keys", "generate", "--algorithm", "ed25519", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "BEGIN PUBLIC KEY" in result.output
    assert out.exists()
    assert b"BEGIN PRIVATE KEY" in out.read_bytes()


def test_keys_generate_ml_dsa_errors() -> None:
    """keys generate --algorithm ml-dsa-65 rejects: ML-DSA-65 keypair
    generation is server-side only (cloud KMS), not in the SDK's local
    SUPPORTED_ALGORITHMS set."""
    result = runner.invoke(app, ["keys", "generate", "--algorithm", "ml-dsa-65"])
    assert result.exit_code == 1
    assert "unsupported_algorithm" in result.output.lower()


# === audit-pack verify ===


@patch("asqav.client._post")
@patch("asqav.init")
def test_audit_pack_verify_valid(
    mock_init: MagicMock, mock_post: MagicMock, tmp_path
) -> None:
    """audit-pack verify posts the bundle wrapped in {bundle: ...} and prints VALID."""
    mock_post.return_value = {
        "bundle_signature_valid": True,
        "bundle_digest": "sha256:abc",
        "receipt_count": 2,
        "report_signature_algorithm": "ml-dsa-65",
        "regimes_summary": {"eu_ai_act": 2},
        "per_receipt": [
            {"record_id": "r1", "signature_valid": True},
            {"record_id": "r2", "signature_valid": True},
        ],
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text('{"receipt_count": 2, "receipts": []}')
    result = runner.invoke(
        app,
        ["audit-pack", "verify", str(bundle_path)],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0, result.output
    assert "VALID" in result.output
    assert "sha256:abc" in result.output
    called_path = mock_post.call_args.args[0]
    body = mock_post.call_args.args[1]
    assert called_path == "/audit-pack/verify"
    assert body["bundle"] == {"receipt_count": 2, "receipts": []}


@patch("asqav.client._post")
@patch("asqav.init")
def test_audit_pack_verify_invalid_exits_1(
    mock_init: MagicMock, mock_post: MagicMock, tmp_path
) -> None:
    """audit-pack verify exits 1 and lists failed receipts on an invalid bundle."""
    mock_post.return_value = {
        "bundle_signature_valid": False,
        "bundle_digest": "sha256:def",
        "receipt_count": 1,
        "report_signature_algorithm": "ml-dsa-65",
        "regimes_summary": {},
        "per_receipt": [{"record_id": "r_bad", "signature_valid": False}],
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("{}")
    result = runner.invoke(
        app,
        ["audit-pack", "verify", str(bundle_path)],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 1
    assert "INVALID" in result.output
    assert "r_bad" in result.output


# === org halt / resume (emergency kill-switch, JWT-scoped) ===


@patch("asqav.cli._session_request")
def test_org_halt_calls_emergency_route(mock_session: MagicMock) -> None:
    """org halt --yes POSTs the emergency-halt route with the reason note."""
    mock_session.return_value = {
        "emergency_halt": True,
        "emergency_halt_at": "2026-06-01T00:00:00Z",
        "emergency_halt_reason": "rogue agent",
    }
    result = runner.invoke(
        app,
        ["org", "halt", "org_x", "--reason", "rogue agent", "--yes"],
        env={"ASQAV_SESSION_TOKEN": "jwt_x"},
    )
    assert result.exit_code == 0, result.output
    assert "RAISED" in result.output
    method, path = mock_session.call_args.args[0], mock_session.call_args.args[1]
    body = mock_session.call_args.args[2]
    assert method == "POST"
    assert path == "/orgs/org_x/emergency-halt"
    assert body == {"reason": "rogue agent"}


@patch("asqav.cli._session_request")
def test_org_resume_calls_deactivate_route(mock_session: MagicMock) -> None:
    """org resume POSTs the deactivate route."""
    mock_session.return_value = {"emergency_halt": False}
    result = runner.invoke(
        app,
        ["org", "resume", "org_x"],
        env={"ASQAV_SESSION_TOKEN": "jwt_x"},
    )
    assert result.exit_code == 0, result.output
    assert "LIFTED" in result.output
    path = mock_session.call_args.args[1]
    assert path == "/orgs/org_x/emergency-halt/deactivate"


def test_org_halt_requires_session_token() -> None:
    """org halt fails closed with a clear message when ASQAV_SESSION_TOKEN is unset."""
    result = runner.invoke(
        app,
        ["org", "halt", "org_x", "--yes"],
        env={"ASQAV_SESSION_TOKEN": "", "ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 1
    assert "ASQAV_SESSION_TOKEN" in result.output


# === keys create / list / revoke (account API keys, JWT-scoped) ===


@patch("asqav.cli._session_request")
def test_keys_create_prints_full_key_once(mock_session: MagicMock) -> None:
    """keys create POSTs /keys and prints the one-time full key."""
    mock_session.return_value = {
        "id": "key_1",
        "name": "ci",
        "scopes": ["agents:read"],
        "key": "sk_live_secret",
    }
    result = runner.invoke(
        app,
        ["keys", "create", "ci", "--scope", "agents:read"],
        env={"ASQAV_SESSION_TOKEN": "jwt_x"},
    )
    assert result.exit_code == 0, result.output
    assert "sk_live_secret" in result.output
    method, path = mock_session.call_args.args[0], mock_session.call_args.args[1]
    body = mock_session.call_args.args[2]
    assert method == "POST"
    assert path == "/keys"
    assert body == {"name": "ci", "scopes": ["agents:read"]}


@patch("asqav.cli._session_request")
def test_keys_list_renders_rows(mock_session: MagicMock) -> None:
    """keys list GETs /keys and renders id + prefix + state."""
    mock_session.return_value = [
        {
            "id": "key_1",
            "name": "ci",
            "key_prefix": "sk_live_ab",
            "scopes": ["*"],
            "revoked": False,
        }
    ]
    result = runner.invoke(
        app, ["keys", "list"], env={"ASQAV_SESSION_TOKEN": "jwt_x"}
    )
    assert result.exit_code == 0, result.output
    assert "key_1" in result.output
    assert "sk_live_ab" in result.output
    assert "active" in result.output
    assert mock_session.call_args.args[1] == "/keys"


@patch("asqav.cli._session_request")
def test_keys_revoke_calls_delete(mock_session: MagicMock) -> None:
    """keys revoke --yes DELETEs /keys/{id}."""
    mock_session.return_value = {"message": "revoked"}
    result = runner.invoke(
        app,
        ["keys", "revoke", "key_1", "--yes"],
        env={"ASQAV_SESSION_TOKEN": "jwt_x"},
    )
    assert result.exit_code == 0, result.output
    assert "Revoked" in result.output
    method, path = mock_session.call_args.args[0], mock_session.call_args.args[1]
    assert method == "DELETE"
    assert path == "/keys/key_1"


# === compliance templates / report / reports ===


@patch("asqav.client._get")
@patch("asqav.init")
def test_compliance_templates_lists(
    mock_init: MagicMock, mock_get: MagicMock
) -> None:
    """compliance templates GETs the templates route and renders ids."""
    mock_get.return_value = [
        {"template_id": "eu_ai_act", "framework": "eu_ai_act", "name": "EU AI Act"}
    ]
    result = runner.invoke(
        app, ["compliance", "templates"], env={"ASQAV_API_KEY": "sk_test"}
    )
    assert result.exit_code == 0, result.output
    assert "eu_ai_act" in result.output
    assert mock_get.call_args.args[0] == "/compliance-reports/templates"


@patch("asqav.client._post")
@patch("asqav.init")
def test_compliance_report_creates(
    mock_init: MagicMock, mock_post: MagicMock
) -> None:
    """compliance report POSTs the create route with the report_type body."""
    mock_post.return_value = {
        "id": "rep_1",
        "name": "Q2",
        "framework": "eu_ai_act",
        "status": "generating",
    }
    result = runner.invoke(
        app,
        ["compliance", "report", "--report-type", "eu_ai_act", "--name", "Q2"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0, result.output
    assert "rep_1" in result.output
    called_path = mock_post.call_args.args[0]
    body = mock_post.call_args.args[1]
    assert called_path == "/compliance-reports"
    assert body["report_type"] == "eu_ai_act"
    assert body["name"] == "Q2"


@patch("asqav.client._get")
@patch("asqav.init")
def test_compliance_reports_lists(
    mock_init: MagicMock, mock_get: MagicMock
) -> None:
    """compliance reports GETs the list route and renders report rows."""
    mock_get.return_value = {
        "reports": [
            {
                "id": "rep_1",
                "name": "Q2",
                "framework": "eu_ai_act",
                "status": "ready",
                "record_count": 5,
            }
        ],
        "total": 1,
    }
    result = runner.invoke(
        app,
        ["compliance", "reports", "--status", "ready"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0, result.output
    assert "rep_1" in result.output
    assert "ready" in result.output
    assert "report_status=ready" in mock_get.call_args.args[0]


# === observability summary / metrics ===


@patch("asqav.client._get")
@patch("asqav.init")
def test_observability_summary(
    mock_init: MagicMock, mock_get: MagicMock
) -> None:
    """observability summary GETs the summary route and renders totals."""
    mock_get.return_value = {
        "total_actions": 12,
        "total_agents_active": 3,
        "total_tokens": 4000,
        "total_cost_usd": 0.42,
        "avg_latency_ms": 18.5,
    }
    result = runner.invoke(
        app, ["observability", "summary"], env={"ASQAV_API_KEY": "sk_test"}
    )
    assert result.exit_code == 0, result.output
    assert "12" in result.output
    assert "18.5" in result.output
    assert mock_get.call_args.args[0] == "/observability/summary"


@patch("asqav.client._get")
@patch("asqav.init")
def test_observability_metrics_passes_window(
    mock_init: MagicMock, mock_get: MagicMock
) -> None:
    """observability metrics threads --hours and --window into the query string."""
    mock_get.return_value = {
        "metrics": [
            {
                "window_start": "2026-06-01T00:00:00Z",
                "agent_id": "agent_x",
                "total_actions": 5,
                "failed_actions": 0,
                "latency": {"p95_ms": 20.0},
            }
        ],
        "window_type": "1hr",
        "hours": 24,
        "count": 1,
    }
    result = runner.invoke(
        app,
        ["observability", "metrics", "--hours", "48", "--window", "1day"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0, result.output
    assert "agent_x" in result.output
    path = mock_get.call_args.args[0]
    assert "hours=48" in path
    assert "window_type=1day" in path


# === replay-verify ===


@patch("asqav.client._get")
@patch("asqav.replay")
@patch("asqav.init")
def test_replay_verify_chain_valid(
    mock_init: MagicMock, mock_replay: MagicMock, mock_account: MagicMock
) -> None:
    """replay-verify wraps replay() and prints the IETF outcome."""
    mock_account.return_value = {"tier": "pro", "organization_id": "o", "organization_name": "n"}
    timeline = MagicMock()
    timeline.compliance_chain_valid = True
    step = MagicMock()
    step.index = 0
    step.signature_id = "sig_1"
    step.chain_valid = True
    step.signed_envelope = {"x": 1}
    timeline.steps = [step]

    def verify_chain_stub() -> bool:
        return True

    timeline.verify_chain = verify_chain_stub
    mock_replay.return_value = timeline

    result = runner.invoke(
        app,
        ["replay-verify", "agent_x", "sess_x"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 0, result.output
    assert "compliance_chain_valid: True" in result.output


@patch("asqav.client._get")
@patch("asqav.replay")
@patch("asqav.init")
def test_replay_verify_strict_rejects_steps_without_envelope(
    mock_init: MagicMock, mock_replay: MagicMock, mock_account: MagicMock
) -> None:
    """replay-verify --strict fails when any step lacks signed_envelope."""
    mock_account.return_value = {"tier": "pro", "organization_id": "o", "organization_name": "n"}
    timeline = MagicMock()
    timeline.compliance_chain_valid = True
    step = MagicMock()
    step.index = 0
    step.signature_id = "sig_1"
    step.chain_valid = True
    step.signed_envelope = None
    timeline.steps = [step]
    timeline.verify_chain = lambda: True
    mock_replay.return_value = timeline

    result = runner.invoke(
        app,
        ["replay-verify", "agent_x", "sess_x", "--strict"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 1
    assert "strict mode FAILED" in result.output


# === migrate run ===


def test_migrate_run_requires_maintenance_key(monkeypatch) -> None:
    """migrate run refuses without ASQAV_MAINTENANCE_KEY."""
    monkeypatch.delenv("ASQAV_MAINTENANCE_KEY", raising=False)
    result = runner.invoke(
        app,
        ["migrate", "run", "v3-20"],
        env={"ASQAV_API_KEY": "sk_test"},
    )
    assert result.exit_code == 1
    assert "ASQAV_MAINTENANCE_KEY" in result.output


def test_migrate_run_rejects_unknown(monkeypatch) -> None:
    """migrate run rejects unsupported migration names."""
    monkeypatch.setenv("ASQAV_MAINTENANCE_KEY", "mk")
    result = runner.invoke(
        app,
        ["migrate", "run", "v9-99"],
        env={"ASQAV_API_KEY": "sk_test", "ASQAV_MAINTENANCE_KEY": "mk"},
    )
    assert result.exit_code == 1
    assert "unsupported migration" in result.output


@patch("urllib.request.urlopen")
@patch("asqav.init")
def test_migrate_run_v3_22_calls_endpoint(
    mock_init: MagicMock, mock_urlopen: MagicMock, monkeypatch
) -> None:
    """migrate run v3-22 POSTs to the correct path with X-Maintenance-Key."""
    response = MagicMock()
    response.read.return_value = b'{"status": "ok", "applied": 1, "migration": "v3-22"}'
    response.__enter__ = lambda self: response
    response.__exit__ = lambda *a: None
    mock_urlopen.return_value = response

    result = runner.invoke(
        app,
        ["migrate", "run", "v3-22"],
        env={"ASQAV_API_KEY": "sk_test", "ASQAV_MAINTENANCE_KEY": "mk_test"},
    )
    assert result.exit_code == 0, result.output
    req = mock_urlopen.call_args.args[0]
    assert "/maintenance/run-migration-v3-22" in req.full_url
    assert req.headers.get("X-maintenance-key") == "mk_test"
    assert "v3-22" in result.output
