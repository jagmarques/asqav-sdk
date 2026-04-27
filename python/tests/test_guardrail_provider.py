"""Tests for CrewAI GuardrailProvider."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Inject a fake crewai module so the real import succeeds without crewai installed.
_fake_crewai = types.ModuleType("crewai")
sys.modules["crewai"] = _fake_crewai

# Remove any cached version first so the fresh fake module is used.
sys.modules.pop("asqav.extras.crewai", None)

from asqav.client import PreflightResult, SignatureResponse  # noqa: E402
from asqav.extras.crewai import AsqavGuardrailProvider  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def provider():
    """Create an AsqavGuardrailProvider with mocked Agent."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent = MagicMock()
        mock_agent_cls.create.return_value = mock_agent
        p = AsqavGuardrailProvider(agent_name="test-guardrail")
    return p


def _make_request(tool_name: str = "search", tool_input: dict | None = None):
    """Create a mock tool call request."""
    req = MagicMock()
    req.tool_name = tool_name
    req.tool_input = tool_input if tool_input is not None else {"query": "test"}
    return req


# ---------------------------------------------------------------------------
# Evaluate - allow
# ---------------------------------------------------------------------------


def test_evaluate_allow(provider):
    """evaluate returns allow when preflight clears."""
    provider._agent.preflight.return_value = PreflightResult(
        cleared=True,
        agent_active=True,
        policy_allowed=True,
        reasons=[],
        explanation="Allowed: agent is active and action is permitted by policy",
    )
    sig = SignatureResponse(
        signature="sig-1",
        signature_id="sid-1",
        action_id="aid-1",
        timestamp=1700000000.0,
        verification_url="https://asqav.com/verify/sid-1",
    )
    provider._agent.sign.return_value = sig

    request = _make_request("search", {"query": "test", "limit": 10})
    result = provider.evaluate(request)

    assert result["verdict"] == "allow"
    assert result["metadata"]["signature_id"] == "sid-1"

    # Preflight was called with tool: prefix
    provider._agent.preflight.assert_called_once_with("tool:search")


# ---------------------------------------------------------------------------
# Evaluate - deny
# ---------------------------------------------------------------------------


def test_evaluate_deny(provider):
    """evaluate returns deny when preflight blocks."""
    provider._agent.preflight.return_value = PreflightResult(
        cleared=False,
        agent_active=True,
        policy_allowed=False,
        reasons=["blocked by policy: no-search"],
        explanation="Blocked: action not permitted by current policy rules",
    )

    request = _make_request("search")
    result = provider.evaluate(request)

    assert result["verdict"] == "deny"
    assert "Blocked" in result["reason"]
    assert result["metadata"]["preflight"]["cleared"] is False
    assert len(result["metadata"]["preflight"]["reasons"]) > 0

    # sign should not have been called (denied at preflight)
    provider._agent.sign.assert_not_called()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_evaluate_unknown_tool(provider):
    """evaluate handles request without tool_name attribute."""
    provider._agent.preflight.return_value = PreflightResult(
        cleared=True,
        agent_active=True,
        policy_allowed=True,
        reasons=[],
        explanation="Allowed",
    )
    sig = SignatureResponse(
        signature="sig-1",
        signature_id="sid-1",
        action_id="aid-1",
        timestamp=1700000000.0,
        verification_url="https://asqav.com/verify/sid-1",
    )
    provider._agent.sign.return_value = sig

    # Request without tool_name attribute
    request = MagicMock(spec=[])
    result = provider.evaluate(request)

    assert result["verdict"] == "allow"
    provider._agent.preflight.assert_called_once_with("tool:unknown")


def test_evaluate_input_keys_sorted(provider):
    """evaluate passes sorted input keys in signing context."""
    provider._agent.preflight.return_value = PreflightResult(
        cleared=True,
        agent_active=True,
        policy_allowed=True,
        reasons=[],
        explanation="Allowed",
    )
    sig = SignatureResponse(
        signature="sig-1",
        signature_id="sid-1",
        action_id="aid-1",
        timestamp=1700000000.0,
        verification_url="https://asqav.com/verify/sid-1",
    )
    provider._agent.sign.return_value = sig

    request = _make_request("calc", {"z_param": 1, "a_param": 2, "m_param": 3})
    provider.evaluate(request)

    # Check the context passed to sign
    sign_call = provider._agent.sign.call_args
    ctx = sign_call[0][1]
    assert ctx["input_keys"] == ["a_param", "m_param", "z_param"]


def test_evaluate_sign_failure_returns_none_signature(provider):
    """evaluate returns None signature_id when signing fails (fail-open)."""
    from asqav.client import AsqavError

    provider._agent.preflight.return_value = PreflightResult(
        cleared=True,
        agent_active=True,
        policy_allowed=True,
        reasons=[],
        explanation="Allowed",
    )
    provider._agent.sign.side_effect = AsqavError("network error")

    request = _make_request("search")
    result = provider.evaluate(request)

    assert result["verdict"] == "allow"
    assert result["metadata"]["signature_id"] is None


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _cleanup_crewai_module():
    """Remove fake crewai from sys.modules after all tests."""
    yield
    sys.modules.pop("crewai", None)
    sys.modules.pop("asqav.extras.crewai", None)
