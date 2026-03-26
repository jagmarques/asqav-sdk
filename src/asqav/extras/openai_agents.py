"""OpenAI Agents SDK integration for asqav.

Install: pip install asqav[openai-agents]

Usage (Phase 8):
    from asqav.extras.openai_agents import AsqavGuardrail
    guardrail = AsqavGuardrail()
"""

try:
    import agents  # noqa: F401
except ImportError:
    raise ImportError(
        "OpenAI Agents integration requires openai-agents. "
        "Install with: pip install asqav[openai-agents]"
    )


class AsqavGuardrail:
    """Guardrail that enforces asqav signing on OpenAI agent runs."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "AsqavGuardrail will be implemented in Phase 8. "
            "See: https://asqav.com/docs/integrations/openai-agents"
        )
