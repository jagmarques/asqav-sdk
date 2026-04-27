"""Tests for three-phase signing flow."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from asqav.client import PreflightResult, SignatureResponse
from asqav.phases import PhaseChain, sign_with_phases


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent(cleared: bool = True) -> MagicMock:
    """Create a mock Agent with sign() and preflight()."""
    agent = MagicMock()

    intent_sig = SignatureResponse(
        signature="sig-intent",
        signature_id="sid-intent-001",
        action_id="aid-intent-001",
        timestamp=1700000000.0,
        verification_url="https://asqav.com/verify/sid-intent-001",
    )
    exec_sig = SignatureResponse(
        signature="sig-exec",
        signature_id="sid-exec-001",
        action_id="aid-exec-001",
        timestamp=1700000001.0,
        verification_url="https://asqav.com/verify/sid-exec-001",
    )
    agent.sign.side_effect = [intent_sig, exec_sig]

    preflight = PreflightResult(
        cleared=cleared,
        agent_active=True,
        policy_allowed=cleared,
        reasons=[] if cleared else ["blocked by policy: test-policy"],
        explanation="Allowed" if cleared else "Blocked: action not permitted",
    )
    agent.preflight.return_value = preflight

    return agent


# ---------------------------------------------------------------------------
# Three-phase flow
# ---------------------------------------------------------------------------


def test_full_three_phase_flow():
    """All three phases execute when preflight clears."""
    agent = _mock_agent(cleared=True)

    chain = sign_with_phases(agent, "api:call", {"model": "gpt-4"})

    assert chain.intent is not None
    assert chain.decision is not None
    assert chain.execution is not None
    assert chain.approved is True

    # Intent phase signed with :intent suffix
    first_call = agent.sign.call_args_list[0]
    assert first_call[0][0] == "api:call:intent"

    # Execution phase signed with :execution suffix and parent_id
    second_call = agent.sign.call_args_list[1]
    assert second_call[0][0] == "api:call:execution"
    assert second_call[1]["parent_id"] == "sid-intent-001"


def test_blocked_action_skips_execution():
    """When preflight denies, execution phase is skipped."""
    agent = _mock_agent(cleared=False)

    chain = sign_with_phases(agent, "api:call")

    assert chain.intent is not None
    assert chain.decision is not None
    assert chain.execution is None
    assert chain.approved is False

    # Only one sign call (intent), no execution
    assert agent.sign.call_count == 1


def test_trace_id_propagation():
    """Custom trace_id is passed through all phases."""
    agent = _mock_agent(cleared=True)
    custom_tid = "trace-abc-123"

    chain = sign_with_phases(agent, "data:read", trace_id=custom_tid)

    assert chain.trace_id == custom_tid

    # Both sign calls should include the trace_id
    for call in agent.sign.call_args_list:
        assert call[1]["trace_id"] == custom_tid


def test_trace_id_auto_generated():
    """When no trace_id provided, one is auto-generated."""
    agent = _mock_agent(cleared=True)

    with patch("asqav.client.generate_trace_id", return_value="auto-tid-999"):
        chain = sign_with_phases(agent, "api:call")

    assert chain.trace_id == "auto-tid-999"


# ---------------------------------------------------------------------------
# PhaseChain serialization
# ---------------------------------------------------------------------------


def test_to_dict():
    """to_dict produces a complete serializable dict."""
    agent = _mock_agent(cleared=True)
    chain = sign_with_phases(agent, "api:call")

    d = chain.to_dict()

    assert d["trace_id"] == chain.trace_id
    assert d["approved"] is True
    assert d["intent"]["signature_id"] == "sid-intent-001"
    assert d["decision"]["cleared"] is True
    assert d["execution"]["signature_id"] == "sid-exec-001"


def test_to_dict_blocked():
    """to_dict handles blocked chain (no execution)."""
    agent = _mock_agent(cleared=False)
    chain = sign_with_phases(agent, "api:call")

    d = chain.to_dict()

    assert d["approved"] is False
    assert d["execution"] is None
    assert d["decision"]["cleared"] is False
    assert len(d["decision"]["reasons"]) > 0


def test_to_json():
    """to_json produces valid JSON matching to_dict."""
    agent = _mock_agent(cleared=True)
    chain = sign_with_phases(agent, "api:call")

    j = chain.to_json()
    parsed = json.loads(j)

    assert parsed == chain.to_dict()


# ---------------------------------------------------------------------------
# Agent convenience method
# ---------------------------------------------------------------------------


def test_agent_convenience_method():
    """Agent.sign_with_phases delegates to phases module."""
    from asqav.client import Agent

    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.client._post") as mock_post,
        patch("asqav.client._get") as mock_get,
    ):
        mock_post.return_value = {
            "agent_id": "ag-1",
            "name": "test",
            "public_key": "pk-1",
            "key_id": "kid-1",
            "algorithm": "ml-dsa-65",
            "capabilities": [],
            "created_at": 1700000000.0,
        }
        agent = Agent.create("test")

        mock_post.return_value = {
            "signature": "sig",
            "signature_id": "sid-1",
            "action_id": "aid-1",
            "timestamp": 1700000000.0,
            "verification_url": "https://asqav.com/verify/sid-1",
        }
        mock_get.side_effect = [
            {"revoked": False, "suspended": False},  # status check
            [],  # policies
        ]

        chain = agent.sign_with_phases("api:call")

    assert isinstance(chain, PhaseChain)
