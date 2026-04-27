"""Tests for CrewAI integration hook."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Inject a fake crewai module so the real import succeeds without crewai installed.
_fake_crewai = types.ModuleType("crewai")
sys.modules["crewai"] = _fake_crewai

# Now we can import the hook (module-level ImportError is satisfied).
# Remove any cached version first so the fresh fake module is used.
sys.modules.pop("asqav.extras.crewai", None)

from asqav.extras.crewai import AsqavCrewHook  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hook():
    """Create an AsqavCrewHook with mocked Agent."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        h = AsqavCrewHook(agent_name="test-crew")
    return h


# ---------------------------------------------------------------------------
# on_task_start
# ---------------------------------------------------------------------------


def test_on_task_start_signs_action(hook):
    """on_task_start signs task:start with description and agent role."""
    with patch.object(hook, "_sign_action") as mock_sign:
        hook.on_task_start("Analyze market data", agent_role="Researcher")
    mock_sign.assert_called_once_with("task:start", {
        "task_description": "Analyze market data",
        "agent_role": "Researcher",
    })


def test_on_task_start_without_role(hook):
    """on_task_start omits agent_role when not provided."""
    with patch.object(hook, "_sign_action") as mock_sign:
        hook.on_task_start("Analyze data")
    ctx = mock_sign.call_args[0][1]
    assert "agent_role" not in ctx
    assert ctx["task_description"] == "Analyze data"


# ---------------------------------------------------------------------------
# on_task_complete
# ---------------------------------------------------------------------------


def test_on_task_complete_signs_action(hook):
    """on_task_complete signs task:complete with output length."""
    with patch.object(hook, "_sign_action") as mock_sign:
        hook.on_task_complete(
            "Write report",
            output="Final analysis results here",
            agent_role="Writer",
        )
    mock_sign.assert_called_once_with("task:complete", {
        "task_description": "Write report",
        "output_length": 27,
        "agent_role": "Writer",
    })


def test_on_task_complete_without_output(hook):
    """on_task_complete omits output_length when output is None."""
    with patch.object(hook, "_sign_action") as mock_sign:
        hook.on_task_complete("Write report")
    ctx = mock_sign.call_args[0][1]
    assert "output_length" not in ctx


# ---------------------------------------------------------------------------
# on_task_fail
# ---------------------------------------------------------------------------


def test_on_task_fail_signs_action(hook):
    """on_task_fail signs task:fail with error message."""
    with patch.object(hook, "_sign_action") as mock_sign:
        hook.on_task_fail(
            "Fetch data",
            error="Connection timeout",
            agent_role="Fetcher",
        )
    mock_sign.assert_called_once_with("task:fail", {
        "task_description": "Fetch data",
        "error": "Connection timeout",
        "agent_role": "Fetcher",
    })


def test_on_task_fail_without_error(hook):
    """on_task_fail omits error when not provided."""
    with patch.object(hook, "_sign_action") as mock_sign:
        hook.on_task_fail("Fetch data")
    ctx = mock_sign.call_args[0][1]
    assert "error" not in ctx


# ---------------------------------------------------------------------------
# step_callback
# ---------------------------------------------------------------------------


def test_step_callback_signs_action(hook):
    """step_callback signs step:execute with type and truncated output."""
    step = MagicMock()
    step.__class__.__name__ = "AgentAction"
    step.__str__ = lambda self: "Tool call: search('query')"

    with patch.object(hook, "_sign_action") as mock_sign:
        hook.step_callback(step)
    mock_sign.assert_called_once()
    action, ctx = mock_sign.call_args[0]
    assert action == "step:execute"
    assert ctx["step_type"] == "AgentAction"
    assert "search" in ctx["step_output"]


# ---------------------------------------------------------------------------
# task_callback
# ---------------------------------------------------------------------------


def test_task_callback_signs_action(hook):
    """task_callback signs task:complete from TaskOutput attributes."""
    task_output = MagicMock()
    task_output.description = "Summarize findings"
    task_output.raw = "The key findings are..."

    with patch.object(hook, "_sign_action") as mock_sign:
        hook.task_callback(task_output)
    mock_sign.assert_called_once()
    action, ctx = mock_sign.call_args[0]
    assert action == "task:complete"
    assert ctx["task_description"] == "Summarize findings"
    assert ctx["output_length"] == len("The key findings are...")


def test_task_callback_handles_missing_attributes(hook):
    """task_callback falls back to str() when attributes are missing."""

    class _PlainOutput:
        def __str__(self) -> str:
            return "plain output text"

    task_output = _PlainOutput()

    with patch.object(hook, "_sign_action") as mock_sign:
        hook.task_callback(task_output)
    mock_sign.assert_called_once()
    action, ctx = mock_sign.call_args[0]
    assert action == "task:complete"
    assert ctx["task_description"] == "plain output text"
    assert "output_length" not in ctx


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_description_truncated(hook):
    """Long descriptions are truncated to 200 chars."""
    long_desc = "x" * 500

    with patch.object(hook, "_sign_action") as mock_sign:
        hook.on_task_start(long_desc)
    ctx = mock_sign.call_args[0][1]
    assert len(ctx["task_description"]) == 200


def test_step_output_truncated(hook):
    """Long step output is truncated to 200 chars."""
    step = MagicMock()
    step.__str__ = lambda self: "y" * 500

    with patch.object(hook, "_sign_action") as mock_sign:
        hook.step_callback(step)
    ctx = mock_sign.call_args[0][1]
    assert len(ctx["step_output"]) == 200


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------


def test_fail_open_on_sign_error(hook):
    """Signing errors do not propagate - fail-open behavior."""
    with patch.object(hook, "_sign_action", side_effect=Exception("boom")):
        # None of these should raise
        with pytest.raises(Exception):
            hook.on_task_start("test")

    # Verify that the real _sign_action swallows AsqavError
    # by testing at the adapter level instead
    from asqav.client import AsqavError

    hook._agent.sign.side_effect = AsqavError("network error")
    # These should NOT raise (fail-open in _sign_action)
    hook.on_task_start("test task")
    hook.on_task_complete("test task")
    hook.on_task_fail("test task")
    hook.step_callback("step data")
    hook.task_callback(MagicMock(spec=[]))


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _cleanup_crewai_module():
    """Remove fake crewai from sys.modules after all tests."""
    yield
    sys.modules.pop("crewai", None)
    sys.modules.pop("asqav.extras.crewai", None)
