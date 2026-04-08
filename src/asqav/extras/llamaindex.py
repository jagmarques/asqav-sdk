"""LlamaIndex integration for asqav.

Install: pip install asqav[llamaindex]

Usage::

    import asqav
    from asqav.extras.llamaindex import AsqavCallbackHandler
    from llama_index.core.callbacks import CallbackManager

    asqav.init("sk_live_...")
    handler = AsqavCallbackHandler(agent_name="my-rag-app")
    callback_manager = CallbackManager([handler])

    # Attach to Settings for global usage
    from llama_index.core import Settings
    Settings.callback_manager = callback_manager
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
except ImportError:
    raise ImportError(
        "LlamaIndex integration requires llama-index-core. "
        "Install with: pip install asqav[llamaindex]"
    )

from ._base import AsqavAdapter


class AsqavCallbackHandler(BaseCallbackHandler, AsqavAdapter):  # type: ignore[misc]
    """LlamaIndex callback handler that signs query, retrieval, and LLM events.

    Every LLM call, retrieval, query, and synthesis event is signed via asqav
    for governance and audit. Signing failures are logged but never raise -
    governance must not break the user's AI pipeline.

    Args:
        api_key: Optional API key override (uses asqav.init() default).
        agent_name: Name for a new agent (calls Agent.create).
        agent_id: ID of an existing agent (calls Agent.get).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        AsqavAdapter.__init__(
            self, api_key=api_key, agent_name=agent_name, agent_id=agent_id
        )
        BaseCallbackHandler.__init__(
            self,
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Sign event start with event type and relevant payload details."""
        context: Dict[str, Any] = {"event_type": event_type.value}
        payload = payload or {}

        if event_type == CBEventType.LLM:
            model = payload.get(EventPayload.MODEL_NAME)
            if model:
                context["model"] = str(model)
            messages = payload.get(EventPayload.MESSAGES)
            if messages:
                context["message_count"] = len(messages)

        elif event_type == CBEventType.QUERY:
            query_str = payload.get(EventPayload.QUERY_STR)
            if query_str:
                context["query_length"] = len(str(query_str))

        elif event_type == CBEventType.RETRIEVE:
            query_str = payload.get(EventPayload.QUERY_STR)
            if query_str:
                context["query_length"] = len(str(query_str))

        elif event_type == CBEventType.EMBEDDING:
            chunks = payload.get(EventPayload.CHUNKS)
            if chunks:
                context["chunk_count"] = len(chunks)

        elif event_type == CBEventType.AGENT_STEP:
            tool = payload.get(EventPayload.TOOL)
            if tool:
                context["tool"] = str(tool)[:200]

        self._sign_action(f"{event_type.value}:start", context)
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Sign event end with event type and result details."""
        context: Dict[str, Any] = {"event_type": event_type.value}
        payload = payload or {}

        if event_type == CBEventType.LLM:
            response = payload.get(EventPayload.RESPONSE)
            completion = payload.get(EventPayload.COMPLETION)
            if response:
                context["response_length"] = len(str(response))
            elif completion:
                context["response_length"] = len(str(completion))

        elif event_type == CBEventType.RETRIEVE:
            nodes = payload.get(EventPayload.NODES)
            if nodes:
                context["nodes_retrieved"] = len(nodes)

        elif event_type == CBEventType.SYNTHESIZE:
            response = payload.get(EventPayload.RESPONSE)
            if response:
                context["response_length"] = len(str(response))

        elif event_type == CBEventType.EMBEDDING:
            embeddings = payload.get(EventPayload.EMBEDDINGS)
            if embeddings:
                context["embedding_count"] = len(embeddings)

        self._sign_action(f"{event_type.value}:end", context)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Start a trace session for grouped signing."""
        self._start_session()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """End the trace session."""
        self._end_session()
