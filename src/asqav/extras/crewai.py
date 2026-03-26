"""CrewAI integration for asqav.

Install: ``pip install asqav[crewai]``

Usage::

    from asqav.extras.crewai import AsqavCrewHook
    hook = AsqavCrewHook(agent_name="my-crew")
    crew = Crew(
        agents=[...],
        tasks=[...],
        step_callback=hook.step_callback,
        task_callback=hook.task_callback,
    )
"""

from __future__ import annotations

from typing import Any

try:
    import crewai  # noqa: F401
except ImportError:
    raise ImportError(
        "CrewAI integration requires crewai. "
        "Install with: pip install asqav[crewai]"
    )

from ._base import AsqavAdapter

_MAX_DESC = 200


class AsqavCrewHook(AsqavAdapter):
    """Hook that signs CrewAI task lifecycle events.

    Provides callable methods that can be passed directly to CrewAI's
    ``step_callback`` and ``task_callback`` parameters. All signing is
    fail-open - governance failures are logged but never interrupt the
    crew's execution.

    Args:
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for a new agent (calls ``Agent.create``).
        agent_id: ID of an existing agent (calls ``Agent.get``).
    """

    # -- Explicit lifecycle methods --

    def on_task_start(
        self,
        task_description: str,
        agent_role: str | None = None,
    ) -> None:
        """Sign a task:start event.

        Args:
            task_description: Description of the task being started.
            agent_role: Role of the CrewAI agent executing the task.
        """
        ctx: dict[str, Any] = {
            "task_description": task_description[:_MAX_DESC],
        }
        if agent_role is not None:
            ctx["agent_role"] = agent_role
        self._sign_action("task:start", ctx)

    def on_task_complete(
        self,
        task_description: str,
        output: str | None = None,
        agent_role: str | None = None,
    ) -> None:
        """Sign a task:complete event.

        Args:
            task_description: Description of the completed task.
            output: Raw output produced by the task.
            agent_role: Role of the CrewAI agent that executed the task.
        """
        ctx: dict[str, Any] = {
            "task_description": task_description[:_MAX_DESC],
        }
        if output is not None:
            ctx["output_length"] = len(output)
        if agent_role is not None:
            ctx["agent_role"] = agent_role
        self._sign_action("task:complete", ctx)

    def on_task_fail(
        self,
        task_description: str,
        error: str | None = None,
        agent_role: str | None = None,
    ) -> None:
        """Sign a task:fail event.

        Args:
            task_description: Description of the failed task.
            error: Error message from the failure.
            agent_role: Role of the CrewAI agent that was executing the task.
        """
        ctx: dict[str, Any] = {
            "task_description": task_description[:_MAX_DESC],
        }
        if error is not None:
            ctx["error"] = error
        if agent_role is not None:
            ctx["agent_role"] = agent_role
        self._sign_action("task:fail", ctx)

    # -- CrewAI callback callables --

    def step_callback(self, step_output: Any) -> None:
        """Callback for CrewAI ``step_callback`` parameter.

        Signs a step:execute event with the step output type and a
        truncated string representation.

        Args:
            step_output: The step output object from CrewAI.
        """
        ctx: dict[str, Any] = {
            "step_type": type(step_output).__name__,
            "step_output": str(step_output)[:_MAX_DESC],
        }
        self._sign_action("step:execute", ctx)

    def task_callback(self, task_output: Any) -> None:
        """Callback for CrewAI ``task_callback`` parameter.

        Signs a task:complete event by extracting description and raw
        output from the ``TaskOutput`` object. Falls back to ``str()``
        if expected attributes are missing.

        Args:
            task_output: The ``TaskOutput`` object from CrewAI.
        """
        description = getattr(task_output, "description", None) or str(task_output)
        raw = getattr(task_output, "raw", None)

        ctx: dict[str, Any] = {
            "task_description": description[:_MAX_DESC],
        }
        if raw is not None:
            ctx["output_length"] = len(raw)
        self._sign_action("task:complete", ctx)
