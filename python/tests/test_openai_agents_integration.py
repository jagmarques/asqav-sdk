"""Tests for OpenAI Agents SDK guardrail integration."""

from __future__ import annotations

import asyncio
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Mock the `agents` package before importing the guardrail module.
# The real openai-agents package is not installed in test env.
# ---------------------------------------------------------------------------

_mock_agents = ModuleType("agents")
sys.modules["agents"] = _mock_agents

# Force reimport so the module picks up the mock
sys.modules.pop("asqav.extras.openai_agents", None)

from asqav.extras.openai_agents import AsqavGuardrail, GuardrailResult  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_guardrail() -> AsqavGuardrail:
    """Create a guardrail with mocked asqav internals."""
    with patch("asqav.client._api_key", "sk_test"):
        with patch("asqav.extras._base.Agent") as mock_agent_cls:
            mock_agent_cls.create.return_value = MagicMock()
            return AsqavGuardrail(agent_name="test-openai-agent")


def _mock_agent(name: str = "my-agent") -> MagicMock:
    """Create a mock OpenAI Agent object with a name attribute."""
    agent = MagicMock()
    agent.name = name
    return agent


# ---------------------------------------------------------------------------
# Input guardrail tests
# ---------------------------------------------------------------------------


def test_input_guardrail_signs_action():
    """run_input_guardrail signs an agent:input action with agent name and input info."""
    guardrail = _make_guardrail()
    agent = _mock_agent("summarizer")
    guardrail._sign_action = MagicMock()

    asyncio.run(guardrail.run_input_guardrail(agent, "Hello world"))

    guardrail._sign_action.assert_called_once()
    args = guardrail._sign_action.call_args
    assert args[0][0] == "agent:input"
    context = args[0][1]
    assert context["agent_name"] == "summarizer"
    assert context["input_type"] == "str"
    assert "input_length" in context
    assert "input_preview" in context


def test_input_guardrail_returns_passed():
    """run_input_guardrail always returns GuardrailResult with passed=True."""
    guardrail = _make_guardrail()
    guardrail._sign_action = MagicMock()

    result = asyncio.run(guardrail.run_input_guardrail(_mock_agent(), "data"))

    assert isinstance(result, GuardrailResult)
    assert result.passed is True
    assert result.output is None


def test_input_guardrail_handles_missing_agent_name():
    """Gracefully handles agent objects without a name attribute."""
    guardrail = _make_guardrail()
    guardrail._sign_action = MagicMock()

    # Agent with no name attribute - use an object without name
    nameless_agent = object()

    asyncio.run(guardrail.run_input_guardrail(nameless_agent, "data"))

    guardrail._sign_action.assert_called_once()
    context = guardrail._sign_action.call_args[0][1]
    # Falls back to str(agent)
    assert isinstance(context["agent_name"], str)
    assert len(context["agent_name"]) > 0


def test_input_truncation():
    """Long input data representation is truncated to 200 chars."""
    guardrail = _make_guardrail()
    guardrail._sign_action = MagicMock()

    long_input = "x" * 1000

    asyncio.run(guardrail.run_input_guardrail(_mock_agent(), long_input))

    context = guardrail._sign_action.call_args[0][1]
    assert len(context["input_preview"]) <= 200


# ---------------------------------------------------------------------------
# Output guardrail tests
# ---------------------------------------------------------------------------


def test_output_guardrail_signs_action():
    """run_output_guardrail signs an agent:output action with agent name and output info."""
    guardrail = _make_guardrail()
    agent = _mock_agent("writer")
    guardrail._sign_action = MagicMock()

    asyncio.run(guardrail.run_output_guardrail(agent, {"result": "done"}))

    guardrail._sign_action.assert_called_once()
    args = guardrail._sign_action.call_args
    assert args[0][0] == "agent:output"
    context = args[0][1]
    assert context["agent_name"] == "writer"
    assert context["output_type"] == "dict"
    assert "output_length" in context
    assert "output_preview" in context


def test_output_guardrail_returns_passed():
    """run_output_guardrail always returns GuardrailResult with passed=True."""
    guardrail = _make_guardrail()
    guardrail._sign_action = MagicMock()

    result = asyncio.run(guardrail.run_output_guardrail(_mock_agent(), "output"))

    assert isinstance(result, GuardrailResult)
    assert result.passed is True
    assert result.output is None


# ---------------------------------------------------------------------------
# Fail-open behavior
# ---------------------------------------------------------------------------


def test_fail_open_on_sign_error():
    """Guardrail returns passed=True even when signing fails."""
    guardrail = _make_guardrail()
    guardrail._sign_action = MagicMock(side_effect=RuntimeError("network error"))

    result = asyncio.run(guardrail.run_input_guardrail(_mock_agent(), "data"))

    assert isinstance(result, GuardrailResult)
    assert result.passed is True
    assert result.output is None


def test_fail_open_output_on_sign_error():
    """Output guardrail also returns passed=True when signing fails."""
    guardrail = _make_guardrail()
    guardrail._sign_action = MagicMock(side_effect=RuntimeError("timeout"))

    result = asyncio.run(guardrail.run_output_guardrail(_mock_agent(), "data"))

    assert isinstance(result, GuardrailResult)
    assert result.passed is True
    assert result.output is None


# ---------------------------------------------------------------------------
# GuardrailResult dataclass
# ---------------------------------------------------------------------------


def test_guardrail_result_defaults():
    """GuardrailResult defaults output to None."""
    result = GuardrailResult(passed=True)
    assert result.passed is True
    assert result.output is None


def test_guardrail_result_with_output():
    """GuardrailResult can hold arbitrary output."""
    result = GuardrailResult(passed=False, output={"reason": "blocked"})
    assert result.passed is False
    assert result.output == {"reason": "blocked"}
