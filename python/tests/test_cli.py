"""Tests for asqav CLI."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typer.testing import CliRunner

from asqav import __version__
from asqav.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_version() -> None:
    """--version prints the SDK version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "asqav" in result.output
    assert __version__ in result.output


# ---------------------------------------------------------------------------
# verify command
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# agents list command
# ---------------------------------------------------------------------------


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


def test_agents_list_no_api_key() -> None:
    """agents list exits with error when ASQAV_API_KEY is not set."""
    result = runner.invoke(app, ["agents", "list"], env={"ASQAV_API_KEY": ""})
    assert result.exit_code == 1
    assert "ASQAV_API_KEY" in result.output


# ---------------------------------------------------------------------------
# agents create command
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# quickstart command
# ---------------------------------------------------------------------------


def test_quickstart_with_api_key() -> None:
    """quickstart shows success when API key is set."""
    result = runner.invoke(app, ["quickstart"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 0
    assert "API key detected" in result.output
    assert "mcpServers" in result.output
    assert "asqav-mcp" in result.output
    assert "Policy" in result.output
    assert "Next steps" in result.output


def test_quickstart_without_api_key() -> None:
    """quickstart warns when API key is missing."""
    result = runner.invoke(app, ["quickstart"], env={"ASQAV_API_KEY": ""})
    assert result.exit_code == 0
    assert "ASQAV_API_KEY not set" in result.output
    assert "mcpServers" in result.output


# ---------------------------------------------------------------------------
# doctor command
# ---------------------------------------------------------------------------


def test_doctor_no_api_key() -> None:
    """doctor reports failure when API key is missing."""
    result = runner.invoke(app, ["doctor"], env={"ASQAV_API_KEY": ""})
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "ASQAV_API_KEY not set" in result.output
    assert "SKIP" in result.output


@patch("asqav.client.health_check")
@patch("asqav.init")
def test_doctor_all_pass(mock_init: MagicMock, mock_health: MagicMock) -> None:
    """doctor passes all checks when API key is set and API is reachable."""
    mock_health.return_value = {"status": "ok"}
    result = runner.invoke(app, ["doctor"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 0
    assert "PASS" in result.output
    assert "API key set" in result.output
    assert "API reachable" in result.output
    assert "All checks passed" in result.output


@patch("asqav.client.health_check")
@patch("asqav.init")
def test_doctor_api_unreachable(mock_init: MagicMock, mock_health: MagicMock) -> None:
    """doctor reports failure when API is unreachable."""
    mock_health.side_effect = Exception("Connection refused")
    result = runner.invoke(app, ["doctor"], env={"ASQAV_API_KEY": "sk_test"})
    assert result.exit_code == 1
    assert "API unreachable" in result.output
    assert "Connection refused" in result.output


# ---------------------------------------------------------------------------
# replay command
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# preflight command
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# budget commands
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# approve command
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# compliance commands
# ---------------------------------------------------------------------------


def test_compliance_frameworks_lists_known() -> None:
    """compliance frameworks lists known framework keys."""
    result = runner.invoke(app, ["compliance", "frameworks"])
    assert result.exit_code == 0
    # FRAMEWORKS dict from compliance.py contains at least these.
    assert "eu_ai_act_art12" in result.output
    assert "soc2" in result.output


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
