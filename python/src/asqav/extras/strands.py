"""Strands Agents integration for asqav.

Install: pip install asqav[strands]

Auto-signs every model call from a Strands ``Agent`` via the typed hooks API
(``BeforeModelCallEvent`` and ``AfterModelCallEvent``). Each model call yields
an ``strands:model_call`` governance signature with input/output hashes, model
name, and latency. Signing is fail-open so governance never breaks the agent.

This adapter is implemented as a thin wrapper over the Asqav SDK's generic
hooks system (:mod:`asqav.hooks`). Input/output hashing is registered as an
internal ``before`` hook so the same pattern is available to any user
integration.

Usage::

    import asqav
    from asqav.extras.strands import AsqavStrandsHooks
    from strands import Agent

    asqav.init(api_key="sk_...")
    hooks = AsqavStrandsHooks(agent_name="my-strands-agent")

    agent = Agent(hooks=[hooks])
    agent("What is the capital of France?")

References:
    https://strandsagents.com/latest/documentation/docs/user-guide/safety-security/hooks/
    https://github.com/strands-agents/sdk-python/issues/2079
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

try:
    from strands.hooks import (
        AfterModelCallEvent,
        BeforeModelCallEvent,
        HookProvider,
        HookRegistry,
    )
except ImportError:
    raise ImportError(
        "Strands integration requires strands-agents. "
        "Install with: pip install asqav[strands]"
    )

from .. import hooks as _asqav_hooks
from ._base import AsqavAdapter

logger = logging.getLogger("asqav")

_STRANDS_ACTION = "strands:model_call"
_HOOK_REGISTERED = False


def _hash_repr(value: Any) -> str:
    """Return a short SHA-256 hash of the repr of ``value``.

    Used so the audit trail records a stable fingerprint of the model
    input/output without leaking the raw content.
    """
    try:
        data = repr(value).encode("utf-8", errors="replace")
    except Exception:
        data = b"<unreprable>"
    return hashlib.sha256(data).hexdigest()[:16]


def _extract_model_name(event: Any) -> str:
    """Best-effort extraction of the model identifier from a hook event."""
    agent = getattr(event, "agent", None)
    if agent is None:
        return "unknown"
    model = getattr(agent, "model", None)
    if model is None:
        return "unknown"
    for attr in ("model_id", "model_name", "name", "id"):
        value = getattr(model, attr, None)
        if value:
            return str(value)
    return type(model).__name__


def _extract_messages(event: Any) -> Any:
    """Pull the current message list from the agent if available."""
    agent = getattr(event, "agent", None)
    if agent is None:
        return None
    return getattr(agent, "messages", None)


def _strands_before_hook(action_type: str, context: dict) -> dict:
    """Internal Asqav before-hook that normalises strands:model_call context.

    Ensures ``input_hash`` and ``output_hash`` keys exist with a stable
    fallback even when the caller did not provide them. This demonstrates the
    hooks pattern enriching context generically.
    """
    if action_type != _STRANDS_ACTION:
        return context
    context.setdefault("input_hash", "none")
    context.setdefault("output_hash", "none")
    return context


def _ensure_hook_registered() -> None:
    """Register the strands before-hook exactly once per process."""
    global _HOOK_REGISTERED
    if _HOOK_REGISTERED:
        return
    _asqav_hooks.register_before("strands:*", _strands_before_hook)
    _HOOK_REGISTERED = True


class AsqavStrandsHooks(AsqavAdapter, HookProvider):
    """Strands ``HookProvider`` that signs every model call with asqav.

    Registers callbacks for ``BeforeModelCallEvent`` and
    ``AfterModelCallEvent``. Every model call produces one
    ``strands:model_call`` signature containing:

    - ``model``: model identifier resolved from the agent
    - ``input_hash``: SHA-256 prefix of the messages sent to the model
    - ``output_hash``: SHA-256 prefix of the model response
    - ``latency_ms``: time between before and after events
    - ``stop_reason``: reason the model stopped, if available
    - ``error``: error message if the model call raised

    Args:
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for an Asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing Asqav agent (calls ``Agent.get``).
        algorithm: Optional signing algorithm hint recorded in the context.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        algorithm: str | None = None,
    ) -> None:
        AsqavAdapter.__init__(
            self, api_key=api_key, agent_name=agent_name, agent_id=agent_id
        )
        self._algorithm = algorithm
        self._call_start: float | None = None
        self._input_hash: str | None = None
        self._model_name: str = "unknown"
        _ensure_hook_registered()

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register Before/AfterModelCall callbacks with the Strands registry."""
        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)
        registry.add_callback(AfterModelCallEvent, self._on_after_model_call)

    def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Capture pre-call state used by the after-call signature."""
        try:
            self._call_start = time.perf_counter()
            self._model_name = _extract_model_name(event)
            self._input_hash = _hash_repr(_extract_messages(event))
        except Exception as exc:
            logger.warning(
                "asqav strands before-model-call capture failed (fail-open): %s", exc
            )

    def _on_after_model_call(self, event: AfterModelCallEvent) -> None:
        """Sign the strands:model_call governance event."""
        try:
            latency_ms: float | None = None
            if self._call_start is not None:
                latency_ms = round((time.perf_counter() - self._call_start) * 1000, 2)

            output_hash = "none"
            stop_reason: str | None = None
            stop_response = getattr(event, "stop_response", None)
            if stop_response is not None:
                message = getattr(stop_response, "message", None)
                output_hash = _hash_repr(message)
                stop_reason = getattr(stop_response, "stop_reason", None)

            error: str | None = None
            exception = getattr(event, "exception", None)
            if exception is not None:
                error = f"{type(exception).__name__}: {str(exception)[:200]}"

            context: dict[str, Any] = {
                "model": self._model_name,
                "input_hash": self._input_hash or "none",
                "output_hash": output_hash,
            }
            if latency_ms is not None:
                context["latency_ms"] = latency_ms
            if stop_reason is not None:
                context["stop_reason"] = str(stop_reason)
            if error is not None:
                context["error"] = error
            if self._algorithm is not None:
                context["algorithm"] = self._algorithm

            self._sign_action(_STRANDS_ACTION, context)
        except Exception as exc:
            logger.warning(
                "asqav strands after-model-call signing failed (fail-open): %s", exc
            )
        finally:
            self._call_start = None
            self._input_hash = None
