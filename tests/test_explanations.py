"""Tests for plain-English explanations in PreflightResult."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from asqav.client import Agent, PreflightResult


# ---------------------------------------------------------------------------
# Direct PreflightResult construction tests
# ---------------------------------------------------------------------------


def test_explanation_field_exists():
    """PreflightResult has an explanation field."""
    result = PreflightResult(
        cleared=True,
        agent_active=True,
        policy_allowed=True,
        reasons=[],
        explanation="test",
    )
    assert result.explanation == "test"


def test_explanation_defaults_to_empty():
    """explanation defaults to empty string if not provided."""
    result = PreflightResult(
        cleared=True,
        agent_active=True,
        policy_allowed=True,
        reasons=[],
    )
    assert result.explanation == ""


# ---------------------------------------------------------------------------
# Integration tests via Agent.preflight()
# ---------------------------------------------------------------------------


MOCK_AGENT_DATA = {
    "agent_id": "agent_test123",
    "agent_name": "test-agent",
    "public_key_id": "key_test123",
    "status": "active",
    "created_at": 1700000000.0,
}


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_explanation_when_cleared(mock_post, mock_get):
    """When cleared, explanation says action is allowed."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")

    # Status: active, no revocation/suspension
    # Policies: empty list (nothing blocks)
    mock_get.side_effect = [
        {"revoked": False, "suspended": False},  # status check
        [],  # policies check
    ]

    result = agent.preflight("llm:call")
    assert result.cleared is True
    assert "Allowed" in result.explanation
    assert "permitted by policy" in result.explanation


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_explanation_when_revoked(mock_post, mock_get):
    """When agent is revoked, explanation says agent has been revoked."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")

    mock_get.side_effect = [
        {"revoked": True, "suspended": False},  # status check
        [],  # policies check
    ]

    result = agent.preflight("llm:call")
    assert result.cleared is False
    assert "revoked" in result.explanation.lower()


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_explanation_when_suspended(mock_post, mock_get):
    """When agent is suspended, explanation mentions suspension."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")

    mock_get.side_effect = [
        {"revoked": False, "suspended": True},  # status check
        [],  # policies check
    ]

    result = agent.preflight("llm:call")
    assert result.cleared is False
    assert "suspended" in result.explanation.lower()


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_explanation_when_policy_blocked(mock_post, mock_get):
    """When blocked by policy, explanation mentions policy rules."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")

    mock_get.side_effect = [
        {"revoked": False, "suspended": False},  # status check
        [{"is_active": True, "action_pattern": "*", "action": "block", "name": "deny-all"}],
    ]

    result = agent.preflight("llm:call")
    assert result.cleared is False
    assert "policy" in result.explanation.lower()


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_explanation_revoked_takes_priority_over_policy(mock_post, mock_get):
    """When both revoked and policy-blocked, revocation is the explanation."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")

    mock_get.side_effect = [
        {"revoked": True, "suspended": False},  # status check
        [{"is_active": True, "action_pattern": "*", "action": "block", "name": "deny-all"}],
    ]

    result = agent.preflight("llm:call")
    assert result.cleared is False
    assert "revoked" in result.explanation.lower()
