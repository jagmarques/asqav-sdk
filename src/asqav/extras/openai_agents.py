"""OpenAI Agents SDK integration for asqav.

Install: pip install asqav[openai-agents]

Provides two integration points:

- ``AsqavGuardrail`` - signs input/output events on agent runs.
- ``AsqavTracingProcessor`` - signs trace and span lifecycle events
  as governance records via the OpenAI Agents tracing pipeline.

Usage::

    from asqav.extras.openai_agents import AsqavGuardrail, AsqavTracingProcessor
    from agents import Agent, add_trace_processor

    # Guardrail (audit-only, never blocks)
    agent = Agent(name="my-agent", guardrails=[AsqavGuardrail(agent_name="my-openai-agent")])

    # Tracing processor (signs every trace/span lifecycle event)
    add_trace_processor(AsqavTracingProcessor(agent_name="my-openai-agent"))
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

try:
    from agents.tracing import Span, Trace, TracingProcessor
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


class AsqavTracingProcessor(AsqavAdapter, TracingProcessor):
    """Tracing processor that signs trace and span lifecycle events.

    Registers with the OpenAI Agents tracing pipeline via
    ``add_trace_processor``. Each trace start/end and span start/end
    is signed as a governance event through asqav.

    All signing is fail-open - tracing failures are logged but never
    raise, so the agent pipeline is never interrupted.

    Args:
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for a new asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing asqav agent (calls ``Agent.get``).
    """

    def on_trace_start(self, trace: Trace) -> None:
        """Sign a trace:start governance event."""
        try:
            self._sign_action(
                "trace:start",
                {
                    "trace_id": trace.trace_id,
                    "trace_name": trace.name,
                },
            )
        except Exception as exc:
            logger.warning("asqav trace:start signing failed (fail-open): %s", exc)

    def on_trace_end(self, trace: Trace) -> None:
        """Sign a trace:end governance event."""
        try:
            self._sign_action(
                "trace:end",
                {
                    "trace_id": trace.trace_id,
                    "trace_name": trace.name,
                },
            )
        except Exception as exc:
            logger.warning("asqav trace:end signing failed (fail-open): %s", exc)

    def on_span_start(self, span: Span[Any]) -> None:
        """Sign a span:start governance event."""
        try:
            span_data = span.span_data
            self._sign_action(
                "span:start",
                {
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "span_type": type(span_data).__name__,
                },
            )
        except Exception as exc:
            logger.warning("asqav span:start signing failed (fail-open): %s", exc)

    def on_span_end(self, span: Span[Any]) -> None:
        """Sign a span:end governance event."""
        try:
            span_data = span.span_data
            exported = span.export() if hasattr(span, "export") else None
            context: dict[str, Any] = {
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "span_type": type(span_data).__name__,
            }
            if exported:
                # Include a preview of the span output for the audit record
                output = exported.get("span_data", {}).get("output")
                if output is not None:
                    output_repr = repr(output)
                    if len(output_repr) > 200:
                        output_repr = output_repr[:200]
                    context["output_preview"] = output_repr
            self._sign_action("span:end", context)
        except Exception as exc:
            logger.warning("asqav span:end signing failed (fail-open): %s", exc)

    def shutdown(self) -> None:
        """End the asqav session on processor shutdown."""
        self._end_session(status="completed")

    def force_flush(self) -> None:
        """No-op - asqav signs events synchronously."""
        pass
