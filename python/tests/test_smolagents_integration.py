"""Tests for smolagents AsqavSmolagentsHook integration.

Uses a mock smolagents module injected into sys.modules so tests run
without smolagents installed.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

# === Inject a fake smolagents module before importing the hook ===

_fake_smolagents = types.ModuleType("smolagents")
sys.modules["smolagents"] = _fake_smolagents
sys.modules.pop("asqav.extras.smolagents", None)

from asqav.extras.smolagents import AsqavSmolagentsHook  # noqa: E402

# === Helpers ===


    # Create a minimal mock smolagents Tool.
def _make_tool(name: str, forward_result: Any = "result") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.forward = MagicMock(return_value=forward_result)
    return tool


# === Fixtures ===


    # Create an AsqavSmolagentsHook with mocked Agent internals.
@pytest.fixture()
def hook() -> AsqavSmolagentsHook:
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        h = AsqavSmolagentsHook(agent_name="test-smolagents")
    h._sign_action = MagicMock(return_value=None)
    return h


# === wrap_tool: tool:start ===


    # wrap_tool signs tool:start with tool name and input preview.
def test_wrap_tool_signs_tool_start(hook: AsqavSmolagentsHook) -> None:
    tool = _make_tool("web_search", forward_result="some result")
    hook.wrap_tool(tool)
    tool.forward("python tutorials")

    first_call = hook._sign_action.call_args_list[0]
    assert first_call[0][0] == "tool:start"
    ctx = first_call[0][1]
    assert ctx["tool"] == "web_search"
    assert ctx["input"] == "python tutorials"


    # wrap_tool signs tool:start using the first kwarg value as input preview.
def test_wrap_tool_signs_tool_start_with_kwargs(hook: AsqavSmolagentsHook) -> None:
    tool = _make_tool("named_tool")
    hook.wrap_tool(tool)
    tool.forward(query="hello world")

    start_ctx = hook._sign_action.call_args_list[0][0][1]
    assert start_ctx["input"] == "hello world"


    # wrap_tool signs tool:start with empty input when called with no args.
def test_wrap_tool_signs_tool_start_no_args(hook: AsqavSmolagentsHook) -> None:
    tool = _make_tool("no_args_tool")
    hook.wrap_tool(tool)
    tool.forward()

    start_ctx = hook._sign_action.call_args_list[0][0][1]
    assert start_ctx["input"] == ""


# === wrap_tool: tool:end ===


    # wrap_tool signs tool:end with output type and length.
def test_wrap_tool_signs_tool_end(hook: AsqavSmolagentsHook) -> None:
    tool = _make_tool("calculator", forward_result="42")
    hook.wrap_tool(tool)
    tool.forward("6 * 7")

    last_call = hook._sign_action.call_args_list[-1]
    assert last_call[0][0] == "tool:end"
    ctx = last_call[0][1]
    assert ctx["tool"] == "calculator"
    assert ctx["output_type"] == "str"
    assert ctx["output_length"] == 2


    # wrap_tool does not alter the tool's return value.
def test_wrap_tool_returns_original_result(hook: AsqavSmolagentsHook) -> None:
    tool = _make_tool("fetch", forward_result={"key": "value"})
    hook.wrap_tool(tool)
    result = tool.forward("url")
    assert result == {"key": "value"}


    # A successful tool call produces exactly tool:start + tool:end.
def test_wrap_tool_signs_two_actions_per_successful_call(
    hook: AsqavSmolagentsHook,
) -> None:
    tool = _make_tool("search")
    hook.wrap_tool(tool)
    tool.forward("query")
    assert hook._sign_action.call_count == 2
    assert hook._sign_action.call_args_list[0][0][0] == "tool:start"
    assert hook._sign_action.call_args_list[1][0][0] == "tool:end"


# === wrap_tool: tool:error ===


    # wrap_tool signs tool:error when forward raises.
def test_wrap_tool_signs_tool_error_on_exception(hook: AsqavSmolagentsHook) -> None:
    tool = _make_tool("unstable_tool")
    tool.forward.side_effect = RuntimeError("connection failed")
    hook.wrap_tool(tool)

    with pytest.raises(RuntimeError):
        tool.forward("input")

    error_call = hook._sign_action.call_args_list[-1]
    assert error_call[0][0] == "tool:error"
    ctx = error_call[0][1]
    assert ctx["tool"] == "unstable_tool"
    assert ctx["error_type"] == "RuntimeError"
    assert ctx["error"] == "connection failed"


    # wrap_tool re-raises the original exception unchanged.
def test_wrap_tool_reraises_original_exception(hook: AsqavSmolagentsHook) -> None:
    tool = _make_tool("crasher")
    tool.forward.side_effect = ValueError("bad input")
    hook.wrap_tool(tool)

    with pytest.raises(ValueError, match="bad input"):
        tool.forward("x")


    # A failed tool call produces exactly tool:start + tool:error.
def test_wrap_tool_signs_start_and_error_on_failure(
    hook: AsqavSmolagentsHook,
) -> None:
    tool = _make_tool("failing_tool")
    tool.forward.side_effect = RuntimeError("boom")
    hook.wrap_tool(tool)

    with pytest.raises(RuntimeError):
        tool.forward("input")

    assert hook._sign_action.call_count == 2
    assert hook._sign_action.call_args_list[0][0][0] == "tool:start"
    assert hook._sign_action.call_args_list[1][0][0] == "tool:error"


# === Truncation ===


    # Inputs longer than 200 chars are truncated in the audit record.
def test_input_truncated_to_200_chars(hook: AsqavSmolagentsHook) -> None:
    tool = _make_tool("search")
    hook.wrap_tool(tool)
    tool.forward("q" * 500)

    start_ctx = hook._sign_action.call_args_list[0][0][1]
    assert len(start_ctx["input"]) == 200


    # Error messages longer than 200 chars are truncated in the audit record.
def test_error_message_truncated_to_200_chars(hook: AsqavSmolagentsHook) -> None:
    tool = _make_tool("erroring")
    tool.forward.side_effect = RuntimeError("e" * 500)
    hook.wrap_tool(tool)

    with pytest.raises(RuntimeError):
        tool.forward("x")

    error_ctx = hook._sign_action.call_args_list[-1][0][1]
    assert len(error_ctx["error"]) == 200


# === Fail-open: signing must never block tool execution ===


    # If tool:start signing raises, the tool still executes and returns its result.
def test_fail_open_tool_start_does_not_block_execution(
    hook: AsqavSmolagentsHook,
) -> None:
    tool = _make_tool("important_tool", forward_result="success")
    hook._sign_action.side_effect = Exception("service unavailable")
    hook.wrap_tool(tool)

    result = tool.forward("input")
    assert result == "success"


    # If tool:end signing raises, the result is still returned to the caller.
def test_fail_open_tool_end_does_not_suppress_result(
    hook: AsqavSmolagentsHook,
) -> None:
    tool = _make_tool("end_failing_tool", forward_result="the result")
    call_count = 0

    def sign_side_effect(action_type: str, ctx: dict) -> None:
        nonlocal call_count
        call_count += 1
        if action_type == "tool:end":
            raise Exception("end signing failed")

    hook._sign_action.side_effect = sign_side_effect
    hook.wrap_tool(tool)

    result = tool.forward("input")
    assert result == "the result"


    # If tool:error signing raises, the original tool exception is still re-raised.
def test_fail_open_tool_error_does_not_swallow_exception(
    hook: AsqavSmolagentsHook,
) -> None:
    tool = _make_tool("double_failing_tool")
    tool.forward.side_effect = ValueError("tool broke")
    hook._sign_action.side_effect = Exception("signing also broke")
    hook.wrap_tool(tool)

    with pytest.raises(ValueError, match="tool broke"):
        tool.forward("input")


# === Multiple tools tracked independently ===


    # Each wrapped tool is signed independently under its own name.
def test_multiple_tools_tracked_independently(hook: AsqavSmolagentsHook) -> None:
    tool_a = _make_tool("tool_a", forward_result="a_result")
    tool_b = _make_tool("tool_b", forward_result="b_result")

    hook.wrap_tool(tool_a)
    hook.wrap_tool(tool_b)

    tool_a.forward("arg_a")
    tool_b.forward("arg_b")

    calls = hook._sign_action.call_args_list
    # tool_a: start + end (indices 0, 1), tool_b: start + end (indices 2, 3)
    assert calls[0][0][1]["tool"] == "tool_a"
    assert calls[1][0][1]["tool"] == "tool_a"
    assert calls[2][0][1]["tool"] == "tool_b"
    assert calls[3][0][1]["tool"] == "tool_b"


# === Tool name fallback ===


    # wrap_tool uses the class name when the tool has no .name attribute.
def test_tool_name_falls_back_to_class_name(hook: AsqavSmolagentsHook) -> None:
    tool = MagicMock(spec=["forward"])  # no .name attribute
    tool.forward = MagicMock(return_value="ok")
    type(tool).__name__ = "MyCustomTool"

    hook.wrap_tool(tool)
    tool.forward("x")

    start_ctx = hook._sign_action.call_args_list[0][0][1]
    assert start_ctx["tool"] == "MyCustomTool"


# === Cleanup ===


    # Remove fake smolagents from sys.modules after all tests.
@pytest.fixture(autouse=True, scope="module")
def _cleanup_smolagents_module() -> Any:
    yield
    sys.modules.pop("smolagents", None)
    sys.modules.pop("asqav.extras.smolagents", None)
