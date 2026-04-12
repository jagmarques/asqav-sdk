"""LlamaIndex integration for asqav.

Install: pip install asqav[llamaindex]

Usage::

    import asqav
    from llama_index.core.callbacks import CallbackManager
    from asqav.extras.llamaindex import AsqavLlamaIndexHandler

    asqav.init(api_key="...")
    handler = AsqavLlamaIndexHandler(agent_name="my-agent")
    callback_manager = CallbackManager([handler])

    # Pass callback_manager to your LlamaIndex components.
    # All LLM, query, retrieval, tool, and agent events are automatically signed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType
except ImportError:
    raise ImportError(
        "LlamaIndex integration requires llama-index-core. "
        "Install with: pip install asqav[llamaindex]"
    )

from ._base import AsqavAdapter

logger = logging.getLogger("asqav")

# Event types we sign governance events for.
_SIGNED_EVENT_TYPES = frozenset(
    {
        CBEventType.LLM,
        CBEventType.QUERY,
        CBEventType.RETRIEVE,
        CBEventType.SYNTHESIZE,
        CBEventType.FUNCTION_CALL,
        CBEventType.AGENT_STEP,
        CBEventType.EMBEDDING,
        CBEventType.RERANKING,
        CBEventType.SUB_QUESTION,
    }
)

_MAX_LEN = 200


def _safe_payload_keys(payload: Optional[Dict[str, Any]]) -> list[str]:
    """Extract sorted payload keys, or empty list if payload is not a dict."""
    if isinstance(payload, dict):
        return sorted(payload.keys())
    return []


class AsqavLlamaIndexHandler(AsqavAdapter, BaseCallbackHandler):  # type: ignore[misc]
    """LlamaIndex callback handler that signs governance events via asqav.

    Implements ``llama_index.core.callbacks.base_handler.BaseCallbackHandler``
    and records signed governance events for LLM calls, queries, retrieval,
    function/tool calls, agent steps, embeddings, reranking, and sub-questions.

    Signing is fail-open: asqav failures are logged but never raised, so
    LlamaIndex pipelines keep running even when asqav is unreachable.

    Args:
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for a new asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing asqav agent (calls ``Agent.get``).
        event_starts_to_ignore: LlamaIndex event types to skip on start.
        event_ends_to_ignore: LlamaIndex event types to skip on end.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        event_starts_to_ignore: list[CBEventType] | None = None,
        event_ends_to_ignore: list[CBEventType] | None = None,
    ) -> None:
        AsqavAdapter.__init__(
            self, api_key=api_key, agent_name=agent_name, agent_id=agent_id
        )
        BaseCallbackHandler.__init__(
            self,
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )

    # ------------------------------------------------------------------
    # BaseCallbackHandler interface
    # ------------------------------------------------------------------

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Sign a llamaindex.<event_type>.start governance event."""
        if event_type not in self.event_starts_to_ignore and event_type in _SIGNED_EVENT_TYPES:
            context: dict[str, Any] = {
                "event_id": event_id,
                "event_type": event_type.value,
                "payload_keys": _safe_payload_keys(payload),
            }
            if parent_id:
                context["parent_id"] = parent_id
            self._sign_action(f"llamaindex.{event_type.value}.start", context)
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Sign a llamaindex.<event_type>.end governance event."""
        if event_type not in self.event_ends_to_ignore and event_type in _SIGNED_EVENT_TYPES:
            context: dict[str, Any] = {
                "event_id": event_id,
                "event_type": event_type.value,
                "payload_keys": _safe_payload_keys(payload),
            }
            self._sign_action(f"llamaindex.{event_type.value}.end", context)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Start a trace - begins an asqav session for grouped signing."""
        if trace_id:
            self._start_session()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """End a trace - closes the asqav session."""
        if self._session_id is not None:
            self._end_session("completed")
