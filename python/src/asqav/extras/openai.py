"""OpenAI SDK integration for asqav.

Install: pip install asqav[openai]

Wraps ``openai.OpenAI`` so every ``chat.completions.create`` call produces a
signed asqav receipt automatically. The OpenAI response is returned unchanged;
the receipt metadata is attached as ``response._asqav_receipt``.

Usage::

    from asqav.extras.openai import AsqavOpenAI

    client = AsqavOpenAI(
        openai_api_key="sk-...",
        asqav_api_key="sk_live_...",
        agent_name="my-openai-agent",
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)
    print(response._asqav_receipt.verification_url)
"""

from __future__ import annotations

import logging
from typing import Any

import asqav

logger = logging.getLogger("asqav.extras.openai")


class _SignedCompletions:
    """Wraps openai.Chat.Completions to sign every create() call."""

    def __init__(self, real_completions: Any, agent: Any, agent_name: str) -> None:
        self._real = real_completions
        self._agent = agent
        self._agent_name = agent_name

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        response = self._real.create(**kwargs)
        try:
            context: dict[str, Any] = {
                "model": response.model if hasattr(response, "model") else model,
                "openai_id": getattr(response, "id", None),
                "action_type": f"openai:chat:{model}",
            }
            usage = getattr(response, "usage", None)
            if usage:
                context["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
                context["completion_tokens"] = getattr(usage, "completion_tokens", None)
                context["total_tokens"] = getattr(usage, "total_tokens", None)
            sig = self._agent.sign(
                action_type=context["action_type"],
                context=context,
            )
            response._asqav_receipt = sig
        except Exception:
            logger.debug("asqav sign failed (fail-soft)", exc_info=True)
        return response


class _SignedChat:
    """Wraps openai.Chat to inject signed completions."""

    def __init__(self, real_chat: Any, agent: Any, agent_name: str) -> None:
        self._real = real_chat
        self.completions = _SignedCompletions(real_chat.completions, agent, agent_name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class AsqavOpenAI:
    """Drop-in replacement for openai.OpenAI that signs every call.

    Parameters
    ----------
    openai_api_key:
        Your OpenAI API key (sk-...).
    asqav_api_key:
        Your asqav API key (sk_live_... or sk_test_...).
    agent_name:
        A label for this agent in your asqav dashboard.
    **kwargs:
        Passed through to openai.OpenAI (base_url, organization, etc.).
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        asqav_api_key: str | None = None,
        agent_name: str = "openai-agent",
        **kwargs: Any,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=openai_api_key, **kwargs)
        asqav.init(asqav_api_key)
        self._agent = asqav.Agent.create(name=agent_name)
        self.chat = _SignedChat(self._client.chat, self._agent, agent_name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
