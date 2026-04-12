"""Tests for DSPy AsqavDSPyCallback integration.

Uses mock dspy modules injected into sys.modules so tests run without
dspy installed.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Inject fake dspy modules before importing the callback
# ---------------------------------------------------------------------------

_MISSING = object()
_original_modules: dict[str, ModuleType | object] = {}


def _install_dspy_mocks() -> type:
    """Inject fake dspy module tree and return the BaseCallback class."""
    for mod_name in ("dspy", "dspy.utils", "dspy.utils.callback"):
        _original_modules[mod_name] = sys.modules.get(mod_name, _MISSING)

    class BaseCallback:
        """Mock dspy BaseCallback."""

        def __init__(self) -> None:
            pass

    dspy = ModuleType("dspy")
    dspy_utils = ModuleType("dspy.utils")
    dspy_utils_callback = ModuleType("dspy.utils.callback")

    dspy_utils_callback.BaseCallback = BaseCallback  # type: ignore[attr-defined]
    dspy_utils.callback = dspy_utils_callback  # type: ignore[attr-defined]
    dspy.utils = dspy_utils  # type: ignore[attr-defined]

    sys.modules["dspy"] = dspy
    sys.modules["dspy.utils"] = dspy_utils
    sys.modules["dspy.utils.callback"] = dspy_utils_callback

    return BaseCallback


def _remove_dspy_mocks() -> None:
    for mod_name, original in _original_modules.items():
        if original is _MISSING:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original  # type: ignore[assignment]
    sys.modules.pop("asqav.extras.dspy", None)


BaseCallback = _install_dspy_mocks()

from asqav.extras.dspy import AsqavDSPyCallback  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cb() -> AsqavDSPyCallback:
    """Create an AsqavDSPyCallback with mocked Agent internals."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        callback = AsqavDSPyCallback(agent_name="test-dspy")
    callback._sign_action = MagicMock(return_value=None)
    return callback


# ---------------------------------------------------------------------------
# Subclass check
# ---------------------------------------------------------------------------


def test_is_base_callback_subclass() -> None:
    """AsqavDSPyCallback must be a subclass of dspy's BaseCallback."""
    assert issubclass(AsqavDSPyCallback, BaseCallback)


# ---------------------------------------------------------------------------
# Module handlers
# ---------------------------------------------------------------------------


def test_on_module_start_signs_action(cb: AsqavDSPyCallback) -> None:
    """on_module_start signs dspy.module.start with module name and input keys."""

    class MyModule:
        pass

    cb.on_module_start("call-1", MyModule(), {"question": "hi", "context": "..."})
    cb._sign_action.assert_called_once_with(
        "dspy.module.start",
        {"call_id": "call-1", "module": "MyModule", "input_keys": ["context", "question"]},
    )


def test_on_module_end_signs_ok(cb: AsqavDSPyCallback) -> None:
    """on_module_end signs dspy.module.end with ok=True on success."""
    cb.on_module_end("call-1", {"answer": "hello"}, None)
    cb._sign_action.assert_called_once_with(
        "dspy.module.end",
        {"call_id": "call-1", "ok": True, "exception": None},
    )


def test_on_module_end_signs_exception(cb: AsqavDSPyCallback) -> None:
    """on_module_end signs dspy.module.end with ok=False and exception name on error."""
    cb.on_module_end("call-2", None, ValueError("bad input"))
    cb._sign_action.assert_called_once_with(
        "dspy.module.end",
        {"call_id": "call-2", "ok": False, "exception": "ValueError"},
    )


def test_on_module_start_empty_inputs(cb: AsqavDSPyCallback) -> None:
    """on_module_start handles empty inputs dict gracefully."""
    cb.on_module_start("call-x", object(), {})
    ctx = cb._sign_action.call_args[0][1]
    assert ctx["input_keys"] == []


def test_on_module_start_non_dict_inputs(cb: AsqavDSPyCallback) -> None:
    """on_module_start handles non-dict inputs without raising."""
    cb.on_module_start("call-x", object(), None)  # type: ignore[arg-type]
    ctx = cb._sign_action.call_args[0][1]
    assert ctx["input_keys"] == []


# ---------------------------------------------------------------------------
# LM handlers
# ---------------------------------------------------------------------------


def test_on_lm_start_signs_action(cb: AsqavDSPyCallback) -> None:
    """on_lm_start signs dspy.lm.start with model name and input keys."""

    class OpenAI:
        model = "gpt-4o"

    cb.on_lm_start("call-2", OpenAI(), {"prompt": "x", "temperature": 0.7})
    cb._sign_action.assert_called_once_with(
        "dspy.lm.start",
        {
            "call_id": "call-2",
            "lm": "OpenAI",
            "model": "gpt-4o",
            "input_keys": ["prompt", "temperature"],
        },
    )


def test_on_lm_start_no_model_attr(cb: AsqavDSPyCallback) -> None:
    """on_lm_start uses None for model when instance has no .model attribute."""
    cb.on_lm_start("call-x", object(), {})
    ctx = cb._sign_action.call_args[0][1]
    assert ctx["model"] is None


def test_on_lm_end_signs_ok(cb: AsqavDSPyCallback) -> None:
    """on_lm_end signs dspy.lm.end with ok=True on success."""
    cb.on_lm_end("call-2", {"text": "result"}, None)
    cb._sign_action.assert_called_once_with(
        "dspy.lm.end",
        {"call_id": "call-2", "ok": True, "exception": None},
    )


def test_on_lm_end_signs_exception(cb: AsqavDSPyCallback) -> None:
    """on_lm_end signs dspy.lm.end with ok=False on error."""
    cb.on_lm_end("call-2", None, TimeoutError("rate limit"))
    ctx = cb._sign_action.call_args[0][1]
    assert ctx["ok"] is False
    assert ctx["exception"] == "TimeoutError"


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def test_on_tool_start_signs_action(cb: AsqavDSPyCallback) -> None:
    """on_tool_start signs dspy.tool.start with tool name and input keys."""

    class SearchTool:
        name = "web_search"

    cb.on_tool_start("call-3", SearchTool(), {"query": "asqav"})
    cb._sign_action.assert_called_once_with(
        "dspy.tool.start",
        {"call_id": "call-3", "tool": "web_search", "input_keys": ["query"]},
    )


def test_on_tool_start_falls_back_to_class_name(cb: AsqavDSPyCallback) -> None:
    """on_tool_start uses class name when instance has no .name attribute."""

    class CalculatorTool:
        pass

    cb.on_tool_start("call-x", CalculatorTool(), {})
    ctx = cb._sign_action.call_args[0][1]
    assert ctx["tool"] == "CalculatorTool"


def test_on_tool_end_signs_ok(cb: AsqavDSPyCallback) -> None:
    """on_tool_end signs dspy.tool.end with ok=True on success."""
    cb.on_tool_end("call-3", {"results": ["a", "b"]}, None)
    cb._sign_action.assert_called_once_with(
        "dspy.tool.end",
        {"call_id": "call-3", "ok": True, "exception": None},
    )


def test_on_tool_end_signs_exception(cb: AsqavDSPyCallback) -> None:
    """on_tool_end signs dspy.tool.end with ok=False on error."""
    cb.on_tool_end("call-3", None, RuntimeError("tool failed"))
    ctx = cb._sign_action.call_args[0][1]
    assert ctx["ok"] is False
    assert ctx["exception"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Input keys are sorted
# ---------------------------------------------------------------------------


def test_input_keys_are_sorted(cb: AsqavDSPyCallback) -> None:
    """Input keys are always sorted alphabetically in audit records."""
    cb.on_module_start("call-s", object(), {"z": 1, "a": 2, "m": 3})
    ctx = cb._sign_action.call_args[0][1]
    assert ctx["input_keys"] == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# Fail-open: signing must never block DSPy execution
# ---------------------------------------------------------------------------


def test_fail_open_on_asqav_error(cb: AsqavDSPyCallback) -> None:
    """AsqavError from the real _sign_action is swallowed (fail-open)."""
    from asqav.client import AsqavError

    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent = MagicMock()
        mock_agent.sign.side_effect = AsqavError("network down")
        mock_agent_cls.create.return_value = mock_agent
        live_cb = AsqavDSPyCallback(agent_name="fail-open-test")

    # None of these should raise
    live_cb.on_module_start("c", object(), {"k": "v"})
    live_cb.on_module_end("c", None, None)
    live_cb.on_lm_start("c", object(), {})
    live_cb.on_lm_end("c", None, None)
    live_cb.on_tool_start("c", object(), {})
    live_cb.on_tool_end("c", None, None)

    assert mock_agent.sign.call_count == 6
    assert len(live_cb._signatures) == 0


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def teardown_module() -> None:
    _remove_dspy_mocks()
