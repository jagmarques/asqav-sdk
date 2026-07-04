"""Tests for CrewAI integration hook and default-on entrypoint.

The adapter duck-types the crew object and does not import crewai, so these
tests run with crewai absent (it is CVE-blocked from the extra).
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from asqav.extras.crewai import AsqavCrewHook, enable_crew_governance

# === Fixtures ===


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


class _DuckCrew:
    """Minimal duck-typed stand-in for a crewai ``Crew`` (no crewai import)."""

    def __init__(self) -> None:
        self.step_callback = None
        self.task_callback = None


# === on_task_start ===


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


# === on_task_complete ===


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


# === on_task_fail ===


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


# === step_callback ===


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


# === task_callback ===


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


# === Truncation ===


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


# === Fail-open ===


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


# === Default-on: enable_crew_governance ===


def test_enable_wires_both_callbacks():
    """enable_crew_governance sets step_callback and task_callback on the crew."""
    crew = _DuckCrew()
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        hook = enable_crew_governance(crew, agent_name="crew")
    assert callable(crew.step_callback)
    assert callable(crew.task_callback)
    assert isinstance(hook, AsqavCrewHook)


def test_default_on_step_signs_by_default():
    """A crew step signs a receipt by default after one enable call.

    Non-vacuous: if enable_crew_governance did not set crew.step_callback (the
    auto-wire), crew.step_callback stays None and calling it raises TypeError,
    so this assertion fails. Break the ``crew.step_callback = ...`` line in
    enable_crew_governance and this test fails; restore it and it passes.
    """
    crew = _DuckCrew()
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        hook = enable_crew_governance(crew, agent_name="crew")
    hook._sign_action = MagicMock()

    # crewai calls crew.step_callback(step_output) after each agent step.
    crew.step_callback("a search step")

    hook._sign_action.assert_called_once()
    assert hook._sign_action.call_args[0][0] == "step:execute"


def test_default_on_task_signs_by_default():
    """A crew task signs a task:complete receipt by default after enable."""
    crew = _DuckCrew()
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        hook = enable_crew_governance(crew, agent_name="crew")
    hook._sign_action = MagicMock()

    crew.task_callback(MagicMock(spec=[]))

    hook._sign_action.assert_called_once()
    assert hook._sign_action.call_args[0][0] == "task:complete"


def test_existing_callbacks_preserved():
    """Existing step/task callbacks still run (additive), plus asqav signs."""
    crew = _DuckCrew()
    seen_steps: list[object] = []
    seen_tasks: list[object] = []
    crew.step_callback = lambda x: seen_steps.append(x)
    crew.task_callback = lambda x: seen_tasks.append(x)

    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        hook = enable_crew_governance(crew, agent_name="crew")
    hook._sign_action = MagicMock()

    crew.step_callback("step-x")
    crew.task_callback(MagicMock(spec=[]))

    assert seen_steps == ["step-x"]
    assert len(seen_tasks) == 1
    assert hook._sign_action.call_count == 2


def test_enable_crew_governance_double_enable_idempotent():
    """Calling enable_crew_governance twice on the same crew signs exactly once per event.

    The check-first guard returns the first hook on the second call, so the
    crew callbacks are wired only once. hook1 is hook2 (same object), and
    each event signs exactly once. Remove the check-first guard and this test
    fails (signs == 2 per event because two hooks are wired).
    """
    crew = _DuckCrew()
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        hook1 = enable_crew_governance(crew, agent_name="crew1")
        hook2 = enable_crew_governance(crew, agent_name="crew2")

    # hook1 and hook2 are the same object now (check-first returns the stored hook)
    assert hook1 is hook2

    signs: list[str] = []
    hook1._sign_action = MagicMock(side_effect=lambda *a, **k: signs.append("sign"))

    crew.step_callback("a step")
    crew.task_callback(MagicMock(spec=[]))

    # Wired once: step + task each sign exactly once
    assert len(signs) == 2


def test_enable_crew_governance_double_enable_same_hook_no_orphan():
    """Double enable returns the same hook and calls Agent.create exactly once.

    Non-vacuous: without the check-first guard, Agent.create is called twice
    (one per enable call) and hook1 is not hook2. Remove the existing-check
    block in enable_crew_governance and this test fails (create_count==2,
    hook1 is not hook2).
    """
    crew = _DuckCrew()
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        hook1 = enable_crew_governance(crew, agent_name="crew1")
        hook2 = enable_crew_governance(crew, agent_name="crew2")
        create_count = mock_agent_cls.create.call_count

    assert hook1 is hook2, "second enable must return the first hook, not a new one"
    assert create_count == 1, f"Agent.create called {create_count} times, expected 1"


def test_module_imports_without_crewai():
    """The adapter imports with crewai absent - duck-typed, no hard import."""
    saved = sys.modules.pop("crewai", None)
    sys.modules.pop("asqav.extras.crewai", None)
    try:
        assert "crewai" not in sys.modules
        mod = importlib.import_module("asqav.extras.crewai")
        assert hasattr(mod, "enable_crew_governance")
        assert "crewai" not in sys.modules
    finally:
        if saved is not None:
            sys.modules["crewai"] = saved
