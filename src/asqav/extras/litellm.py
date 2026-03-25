"""LiteLLM integration for asqav.

Install: pip install asqav[litellm]

Usage (Phase 8):
    from asqav.extras.litellm import AsqavGuardrail
    guardrail = AsqavGuardrail()
"""

try:
    import litellm  # noqa: F401
except ImportError:
    raise ImportError(
        "LiteLLM integration requires litellm. "
        "Install with: pip install asqav[litellm]"
    )


class AsqavGuardrail:
    """Guardrail that enforces asqav signing on LiteLLM calls."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "AsqavGuardrail will be implemented in Phase 8. "
            "See: https://asqav.com/docs/integrations/litellm"
        )
