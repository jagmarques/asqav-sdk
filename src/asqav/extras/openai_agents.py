"""OpenAI Agents SDK integration for asqav.

Install: pip install asqav[openai-agents]

Usage::

    from asqav.extras.openai_agents import AsqavGuardrail
    from agents import Agent

    agent = Agent(name="my-agent", guardrails=[AsqavGuardrail(agent_name="my-openai-agent")])
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

try:
    import agents  # noqa: F401
except ImportError:
    raise ImportError(
        "OpenAI Agents integration requires openai-agents. "
        "Install with: pip install asqav[openai-agents]"
    )

from ._base import AsqavAdapter

logger = logging.getLogger("asqav")


@dataclasses.dataclass
class GuardrailResult:
    """Lightweight result matching the OpenAI Agents SDK guardrail interface.

    Defined locally so the integration does not break when the SDK's
    internal types change between versions (adapter pattern).
    """

    passed: bool
    output: Any | None = None


class AsqavGuardrail(AsqavAdapter):
    """Guardrail that signs input/output events on OpenAI agent runs.

    Asqav guardrails are audit-only: they sign governance events but never
    block agent execution. Both ``run_input_guardrail`` and
    ``run_output_guardrail`` always return ``GuardrailResult(passed=True)``.

    Args:
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for a new asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing asqav agent (calls ``Agent.get``).
    """

    def _resolve_agent_name(self, agent: Any) -> str:
        """Extract agent name from the OpenAI Agent object."""
        name = getattr(agent, "name", None)
        if name:
            return str(name)
        try:
            return str(agent)
        except Exception:
            return "unknown"

    async def run_input_guardrail(
        self, agent: Any, input_data: Any
    ) -> GuardrailResult:
        """Sign an agent:input governance event.

        Called by the OpenAI Agents SDK before an agent processes input.
        Signing failures are swallowed (fail-open) so the agent run
        is never interrupted by governance issues.
        """
        try:
            agent_name = self._resolve_agent_name(agent)
            data_repr = repr(input_data)
            if len(data_repr) > 200:
                data_repr = data_repr[:200]
            self._sign_action(
                "agent:input",
                {
                    "agent_name": agent_name,
                    "input_type": type(input_data).__name__,
                    "input_length": len(data_repr),
                    "input_preview": data_repr,
                },
            )
        except Exception as exc:
            logger.warning("asqav input guardrail signing failed (fail-open): %s", exc)
        return GuardrailResult(passed=True, output=None)

    async def run_output_guardrail(
        self, agent: Any, output_data: Any
    ) -> GuardrailResult:
        """Sign an agent:output governance event.

        Called by the OpenAI Agents SDK after an agent produces output.
        Signing failures are swallowed (fail-open) so the agent run
        is never interrupted by governance issues.
        """
        try:
            agent_name = self._resolve_agent_name(agent)
            data_repr = repr(output_data)
            if len(data_repr) > 200:
                data_repr = data_repr[:200]
            self._sign_action(
                "agent:output",
                {
                    "agent_name": agent_name,
                    "output_type": type(output_data).__name__,
                    "output_length": len(data_repr),
                    "output_preview": data_repr,
                },
            )
        except Exception as exc:
            logger.warning("asqav output guardrail signing failed (fail-open): %s", exc)
        return GuardrailResult(passed=True, output=None)
