"""Preflight does not silently clear when a check cannot complete.

A transient status/policy fetch error must not leave the agent cleared. The
result reports checks_complete=False and cleared=False so a network blip never
reads as a pass for a revoked or policy-blocked agent.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import Agent, PreflightResult

MOCK_AGENT_DATA = {
    "agent_id": "agent_test123",
    "name": "test-agent",
    "public_key": "pk_test123",
    "key_id": "key_test123",
    "algorithm": "ml-dsa-65",
    "capabilities": [],
    "created_at": 1700000000.0,
}


def test_checks_complete_defaults_true():
    """checks_complete defaults to True so existing constructions stay valid."""
    result = PreflightResult(
        cleared=True,
        agent_active=True,
        policy_allowed=True,
        reasons=[],
    )
    assert result.checks_complete is True


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_status_fetch_error_does_not_clear(mock_post, mock_get):
    """A status fetch error blocks rather than silently clearing the agent."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")

    # Status fetch raises (transient), policies return empty.
    mock_get.side_effect = [
        ConnectionError("network down"),  # status check
        [],  # policies check
    ]

    result = agent.preflight("llm:call")
    assert result.cleared is False
    assert result.checks_complete is False
    assert "could not verify" in result.explanation.lower()


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_policy_fetch_error_does_not_clear(mock_post, mock_get):
    """A policy fetch error blocks rather than silently clearing the agent."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")

    mock_get.side_effect = [
        {"revoked": False, "suspended": False},  # status check
        ConnectionError("network down"),  # policies check
    ]

    result = agent.preflight("llm:call")
    assert result.cleared is False
    assert result.checks_complete is False


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_non_list_policies_does_not_clear(mock_post, mock_get):
    """A non-list /policies response fails closed instead of silently clearing."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")

    mock_get.side_effect = [
        {"revoked": False, "suspended": False},  # status check
        {"policies": []},  # anomalous non-list policies response
    ]

    result = agent.preflight("llm:call")
    assert result.cleared is False
    assert result.checks_complete is False
    assert any("unexpected response" in r for r in result.reasons)


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_happy_path_still_clears(mock_post, mock_get):
    """An active agent with no blocking policy still clears, checks_complete True."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")

    mock_get.side_effect = [
        {"revoked": False, "suspended": False},  # status check
        [],  # policies check
    ]

    result = agent.preflight("llm:call")
    assert result.cleared is True
    assert result.checks_complete is True


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_revoked_agent_is_a_real_verdict(mock_post, mock_get):
    """A revoked agent blocks with checks_complete True, distinct from an error."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")

    mock_get.side_effect = [
        {"revoked": True, "suspended": False},  # status check
        [],  # policies check
    ]

    result = agent.preflight("llm:call")
    assert result.cleared is False
    assert result.checks_complete is True
    assert "revoked" in result.explanation.lower()
