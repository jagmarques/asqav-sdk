"""LiteLLM integration for asqav.

Install: pip install asqav[litellm]

Usage::

    import asqav
    from asqav.extras.litellm import AsqavGuardrail
    import litellm

    asqav.init("sk_live_...")
    litellm.callbacks = [AsqavGuardrail(agent_name="my-litellm-proxy")]
"""

from __future__ import annotations

from typing import Any

try:
    import litellm  # noqa: F401
except ImportError:
    raise ImportError(
        "LiteLLM integration requires litellm. "
        "Install with: pip install asqav[litellm]"
    )

from ._base import AsqavAdapter


class AsqavGuardrail(AsqavAdapter):
    """Guardrail that signs LiteLLM calls via asqav governance.

    Registers as a LiteLLM callback to sign pre-call, post-call,
    error, and moderation events. Signing failures are fail-open -
    they never block LLM requests.

    Args:
        api_key: Optional API key override (uses asqav.init() default).
        agent_name: Name for a new agent (calls Agent.create).
        agent_id: ID of an existing agent (calls Agent.get).
    """

    # -- LiteLLM guardrail callback interface --

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> None:
        """Sign before an LLM call is made."""
        messages = data.get("messages", [])
        self._sign_action(
            "llm:pre_call",
            {
                "call_type": call_type,
                "model": data.get("model"),
                "message_count": len(messages),
            },
        )

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: dict,
        response: Any,
    ) -> None:
        """Sign after a successful LLM call."""
        context: dict[str, Any] = {
            "model": data.get("model"),
            "response_type": type(response).__name__,
        }
        # Extract token usage if available on the response
        usage = getattr(response, "usage", None)
        if usage is not None:
            context["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        self._sign_action("llm:post_call", context)

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: dict,
    ) -> None:
        """Sign after a failed LLM call."""
        self._sign_action(
            "llm:error",
            {
                "error_type": type(original_exception).__name__,
                "error_message": str(original_exception),
            },
        )

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: dict,
        call_type: str,
    ) -> None:
        """Sign moderation events."""
        self._sign_action(
            "llm:moderation",
            {
                "call_type": call_type,
            },
        )
