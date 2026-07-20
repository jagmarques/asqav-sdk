"""Anthropic SDK integration for asqav.

Install: pip install asqav[anthropic]

Wraps anthropic.Anthropic so every messages.create call produces a
signed asqav receipt automatically.

Usage::

    from asqav.extras.anthropic import AsqavAnthropic
    client = AsqavAnthropic(
        anthropic_api_key="sk-ant-...",
        asqav_api_key="sk_live_...",
        agent_name="my-claude-agent",
    )
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.content[0].text)
    print(response._asqav_receipt.verification_url)
"""
from __future__ import annotations
import logging
from typing import Any
import asqav

logger = logging.getLogger("asqav.extras.anthropic")


class _SignedMessages:
    def __init__(self, real_messages: Any, agent: Any) -> None:
        self._real = real_messages
        self._agent = agent

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        response = self._real.create(**kwargs)
        try:
            context: dict[str, Any] = {
                "model": getattr(response, "model", model),
                "anthropic_id": getattr(response, "id", None),
                "action_type": f"anthropic:chat:{model}",
            }
            usage = getattr(response, "usage", None)
            if usage:
                context["input_tokens"] = getattr(usage, "input_tokens", None)
                context["output_tokens"] = getattr(usage, "output_tokens", None)
            sig = self._agent.sign(action_type=context["action_type"], context=context)
            response._asqav_receipt = sig
        except Exception:
            logger.debug("asqav sign failed (fail-soft)", exc_info=True)
        return response


class _SignedClient:
    def __init__(self, real_client: Any, agent: Any) -> None:
        self._real = real_client
        self.messages = _SignedMessages(real_client.messages, agent)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class AsqavAnthropic:
    """Drop-in replacement for anthropic.Anthropic that signs every call."""

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        asqav_api_key: str | None = None,
        agent_name: str = "anthropic-agent",
        **kwargs: Any,
    ) -> None:
        from anthropic import Anthropic
        self._client = Anthropic(api_key=anthropic_api_key, **kwargs)
        asqav.init(asqav_api_key)
        self._agent = asqav.Agent.create(name=agent_name)
        self._wrapped = _SignedClient(self._client, self._agent)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)
