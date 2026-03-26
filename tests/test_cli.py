"""Tests for asqav CLI."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typer.testing import CliRunner

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
    assert "0.2.6" in result.output


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
