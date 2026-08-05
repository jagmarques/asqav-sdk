"""Tests for OpenAI Agents SDK guardrail integration."""

from __future__ import annotations

import asyncio
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mock `agents` + `agents.tracing` before importing the guardrail.
_mock_agents = ModuleType("agents")
_mock_agents_tracing = ModuleType("agents.tracing")
_mock_agents_tracing.Span = MagicMock  # type: ignore[attr-defined]
_mock_agents_tracing.Trace = MagicMock  # type: ignore[attr-defined]
_mock_agents_tracing.TracingProcessor = MagicMock  # type: ignore[attr-defined]
_mock_agents.Agent = MagicMock  # type: ignore[attr-defined]
_mock_agents.add_trace_processor = MagicMock()  # type: ignore[attr-defined]
_mock_agents.tracing = _mock_agents_tracing  # type: ignore[attr-defined]
sys.modules["agents"] = _mock_agents
sys.modules["agents.tracing"] = _mock_agents_tracing

sys.modules.pop("asqav.extras.openai_agents", None)

from asqav.extras.openai_agents import AsqavGuardrail, GuardrailResult  # noqa: E402, I001


# === Helpers ===


    # Create a guardrail with mocked asqav internals.
def _make_guardrail() -> AsqavGuardrail:
    with patch("asqav.client._api_key", "sk_test"):
        with patch("asqav.extras._base.Agent") as mock_agent_cls:
            mock_agent_cls.create.return_value = MagicMock()
            return AsqavGuardrail(agent_name="test-openai-agent")


    # Create a mock OpenAI Agent object with a name attribute.
def _mock_agent(name: str = "my-agent") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


# === Input guardrail tests ===


    # run_input_guardrail signs an agent:input action with agent name and input info.
def test_input_guardrail_signs_action():
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


    # run_input_guardrail always returns GuardrailResult with passed=True.
def test_input_guardrail_returns_passed():
    guardrail = _make_guardrail()
    guardrail._sign_action = MagicMock()

    result = asyncio.run(guardrail.run_input_guardrail(_mock_agent(), "data"))

    assert isinstance(result, GuardrailResult)
    assert result.passed is True
    assert result.output is None


    # Gracefully handles agent objects without a name attribute.
def test_input_guardrail_handles_missing_agent_name():
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


    # Long input data representation is truncated to 200 chars.
def test_input_truncation():
    guardrail = _make_guardrail()
    guardrail._sign_action = MagicMock()

    long_input = "x" * 1000

    asyncio.run(guardrail.run_input_guardrail(_mock_agent(), long_input))

    context = guardrail._sign_action.call_args[0][1]
    assert len(context["input_preview"]) <= 200


# === Output guardrail tests ===


    # run_output_guardrail signs an agent:output action with agent name and output info.
def test_output_guardrail_signs_action():
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


    # run_output_guardrail always returns GuardrailResult with passed=True.
def test_output_guardrail_returns_passed():
    guardrail = _make_guardrail()
    guardrail._sign_action = MagicMock()

    result = asyncio.run(guardrail.run_output_guardrail(_mock_agent(), "output"))

    assert isinstance(result, GuardrailResult)
    assert result.passed is True
    assert result.output is None


# === Fail-open behavior ===


    # Guardrail returns passed=True even when signing fails.
def test_fail_open_on_sign_error():
    guardrail = _make_guardrail()
    guardrail._sign_action = MagicMock(side_effect=RuntimeError("network error"))

    result = asyncio.run(guardrail.run_input_guardrail(_mock_agent(), "data"))

    assert isinstance(result, GuardrailResult)
    assert result.passed is True
    assert result.output is None


    # Output guardrail also returns passed=True when signing fails.
def test_fail_open_output_on_sign_error():
    guardrail = _make_guardrail()
    guardrail._sign_action = MagicMock(side_effect=RuntimeError("timeout"))

    result = asyncio.run(guardrail.run_output_guardrail(_mock_agent(), "data"))

    assert isinstance(result, GuardrailResult)
    assert result.passed is True
    assert result.output is None


# === GuardrailResult dataclass ===


    # GuardrailResult defaults output to None.
def test_guardrail_result_defaults():
    result = GuardrailResult(passed=True)
    assert result.passed is True
    assert result.output is None


    # GuardrailResult can hold arbitrary output.
def test_guardrail_result_with_output():
    result = GuardrailResult(passed=False, output={"reason": "blocked"})
    assert result.passed is False
    assert result.output == {"reason": "blocked"}
