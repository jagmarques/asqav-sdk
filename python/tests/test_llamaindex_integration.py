"""Tests for LlamaIndex AsqavLlamaIndexHandler integration.

Uses mock llama_index modules injected into sys.modules so tests run
without llama-index-core installed.
"""

from __future__ import annotations

import os
import sys
from enum import Enum
from types import ModuleType
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Inject fake llama_index modules before importing the handler
# ---------------------------------------------------------------------------

_MISSING = object()
_original_modules: dict[str, ModuleType | object] = {}

_MOCK_MODULE_NAMES = (
    "llama_index",
    "llama_index.core",
    "llama_index.core.callbacks",
    "llama_index.core.callbacks.base_handler",
    "llama_index.core.callbacks.schema",
)


class CBEventType(str, Enum):
    """Faithful reproduction of llama_index.core.callbacks.schema.CBEventType."""

    CHUNKING = "chunking"
    NODE_PARSING = "node_parsing"
    EMBEDDING = "embedding"
    LLM = "llm"
    QUERY = "query"
    RETRIEVE = "retrieve"
    SYNTHESIZE = "synthesize"
    TREE = "tree"
    SUB_QUESTION = "sub_question"
    TEMPLATING = "templating"
    FUNCTION_CALL = "function_call"
    RERANKING = "reranking"
    EXCEPTION = "exception"
    AGENT_STEP = "agent_step"


class BaseCallbackHandler:
    """Faithful reproduction of llama_index BaseCallbackHandler."""

    def __init__(
        self,
        event_starts_to_ignore: List[Any],
        event_ends_to_ignore: List[Any],
    ) -> None:
        self.event_starts_to_ignore = tuple(event_starts_to_ignore)
        self.event_ends_to_ignore = tuple(event_ends_to_ignore)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        raise NotImplementedError

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        raise NotImplementedError


def _install_llamaindex_mocks() -> None:
    """Inject fake llama_index module tree into sys.modules."""
    for mod_name in _MOCK_MODULE_NAMES:
        _original_modules[mod_name] = sys.modules.get(mod_name, _MISSING)

    llama_index = ModuleType("llama_index")
    llama_index_core = ModuleType("llama_index.core")
    llama_index_core_callbacks = ModuleType("llama_index.core.callbacks")
    llama_index_core_callbacks_base_handler = ModuleType(
        "llama_index.core.callbacks.base_handler"
    )
    llama_index_core_callbacks_schema = ModuleType(
        "llama_index.core.callbacks.schema"
    )

    llama_index_core_callbacks_base_handler.BaseCallbackHandler = BaseCallbackHandler  # type: ignore[attr-defined]
    llama_index_core_callbacks_schema.CBEventType = CBEventType  # type: ignore[attr-defined]

    llama_index_core_callbacks.base_handler = llama_index_core_callbacks_base_handler  # type: ignore[attr-defined]
    llama_index_core_callbacks.schema = llama_index_core_callbacks_schema  # type: ignore[attr-defined]
    llama_index_core.callbacks = llama_index_core_callbacks  # type: ignore[attr-defined]
    llama_index.core = llama_index_core  # type: ignore[attr-defined]

    sys.modules["llama_index"] = llama_index
    sys.modules["llama_index.core"] = llama_index_core
    sys.modules["llama_index.core.callbacks"] = llama_index_core_callbacks
    sys.modules["llama_index.core.callbacks.base_handler"] = (
        llama_index_core_callbacks_base_handler
    )
    sys.modules["llama_index.core.callbacks.schema"] = (
        llama_index_core_callbacks_schema
    )


def _remove_llamaindex_mocks() -> None:
    for mod_name, original in _original_modules.items():
        if original is _MISSING:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original  # type: ignore[assignment]
    sys.modules.pop("asqav.extras.llamaindex", None)


_install_llamaindex_mocks()

from asqav.extras.llamaindex import AsqavLlamaIndexHandler  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def handler() -> AsqavLlamaIndexHandler:
    """Create an AsqavLlamaIndexHandler with mocked Agent internals."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        h = AsqavLlamaIndexHandler(agent_name="test-llamaindex")
    h._sign_action = MagicMock(return_value=None)
    return h


# ---------------------------------------------------------------------------
# Subclass check
# ---------------------------------------------------------------------------


def test_is_base_callback_handler_subclass() -> None:
    """AsqavLlamaIndexHandler must be a subclass of LlamaIndex BaseCallbackHandler."""
    assert issubclass(AsqavLlamaIndexHandler, BaseCallbackHandler)


# ---------------------------------------------------------------------------
# on_event_start - signed event types
# ---------------------------------------------------------------------------


def test_on_event_start_llm(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_start signs llamaindex.llm.start for LLM events."""
    handler.on_event_start(CBEventType.LLM, {"prompt": "hi"}, "evt-1", "parent-1")
    handler._sign_action.assert_called_once_with(
        "llamaindex.llm.start",
        {
            "event_id": "evt-1",
            "event_type": "llm",
            "payload_keys": ["prompt"],
            "parent_id": "parent-1",
        },
    )


def test_on_event_start_query(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_start signs llamaindex.query.start for query events."""
    handler.on_event_start(CBEventType.QUERY, {"query_str": "test"}, "evt-2")
    handler._sign_action.assert_called_once_with(
        "llamaindex.query.start",
        {
            "event_id": "evt-2",
            "event_type": "query",
            "payload_keys": ["query_str"],
        },
    )


def test_on_event_start_retrieve(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_start signs llamaindex.retrieve.start for retrieval events."""
    handler.on_event_start(CBEventType.RETRIEVE, None, "evt-3")
    handler._sign_action.assert_called_once_with(
        "llamaindex.retrieve.start",
        {
            "event_id": "evt-3",
            "event_type": "retrieve",
            "payload_keys": [],
        },
    )


def test_on_event_start_function_call(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_start signs llamaindex.function_call.start for tool calls."""
    handler.on_event_start(
        CBEventType.FUNCTION_CALL,
        {"function_call": "search", "tool": "web"},
        "evt-4",
    )
    handler._sign_action.assert_called_once_with(
        "llamaindex.function_call.start",
        {
            "event_id": "evt-4",
            "event_type": "function_call",
            "payload_keys": ["function_call", "tool"],
        },
    )


def test_on_event_start_agent_step(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_start signs llamaindex.agent_step.start for agent events."""
    handler.on_event_start(CBEventType.AGENT_STEP, {}, "evt-5")
    handler._sign_action.assert_called_once_with(
        "llamaindex.agent_step.start",
        {
            "event_id": "evt-5",
            "event_type": "agent_step",
            "payload_keys": [],
        },
    )


def test_on_event_start_embedding(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_start signs llamaindex.embedding.start for embedding events."""
    handler.on_event_start(CBEventType.EMBEDDING, {"embeddings": [1, 2]}, "evt-6")
    handler._sign_action.assert_called_once_with(
        "llamaindex.embedding.start",
        {
            "event_id": "evt-6",
            "event_type": "embedding",
            "payload_keys": ["embeddings"],
        },
    )


def test_on_event_start_reranking(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_start signs llamaindex.reranking.start for reranking events."""
    handler.on_event_start(CBEventType.RERANKING, {"nodes": []}, "evt-7")
    handler._sign_action.assert_called_once_with(
        "llamaindex.reranking.start",
        {
            "event_id": "evt-7",
            "event_type": "reranking",
            "payload_keys": ["nodes"],
        },
    )


def test_on_event_start_sub_question(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_start signs llamaindex.sub_question.start for sub-question events."""
    handler.on_event_start(CBEventType.SUB_QUESTION, {"sub_question": "why?"}, "evt-8")
    handler._sign_action.assert_called_once_with(
        "llamaindex.sub_question.start",
        {
            "event_id": "evt-8",
            "event_type": "sub_question",
            "payload_keys": ["sub_question"],
        },
    )


def test_on_event_start_synthesize(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_start signs llamaindex.synthesize.start for synthesize events."""
    handler.on_event_start(CBEventType.SYNTHESIZE, {}, "evt-9")
    handler._sign_action.assert_called_once_with(
        "llamaindex.synthesize.start",
        {
            "event_id": "evt-9",
            "event_type": "synthesize",
            "payload_keys": [],
        },
    )


# ---------------------------------------------------------------------------
# on_event_start - ignored event types (not in _SIGNED_EVENT_TYPES)
# ---------------------------------------------------------------------------


def test_on_event_start_ignores_chunking(handler: AsqavLlamaIndexHandler) -> None:
    """Chunking events are not signed (low-value for governance)."""
    handler.on_event_start(CBEventType.CHUNKING, {}, "evt-x")
    handler._sign_action.assert_not_called()


def test_on_event_start_ignores_node_parsing(handler: AsqavLlamaIndexHandler) -> None:
    """Node parsing events are not signed."""
    handler.on_event_start(CBEventType.NODE_PARSING, {}, "evt-x")
    handler._sign_action.assert_not_called()


def test_on_event_start_ignores_templating(handler: AsqavLlamaIndexHandler) -> None:
    """Templating events are not signed."""
    handler.on_event_start(CBEventType.TEMPLATING, {}, "evt-x")
    handler._sign_action.assert_not_called()


def test_on_event_start_ignores_tree(handler: AsqavLlamaIndexHandler) -> None:
    """Tree events are not signed."""
    handler.on_event_start(CBEventType.TREE, {}, "evt-x")
    handler._sign_action.assert_not_called()


def test_on_event_start_ignores_exception(handler: AsqavLlamaIndexHandler) -> None:
    """Exception events are not signed (they are metadata, not actions)."""
    handler.on_event_start(CBEventType.EXCEPTION, {}, "evt-x")
    handler._sign_action.assert_not_called()


# ---------------------------------------------------------------------------
# on_event_start - returns event_id
# ---------------------------------------------------------------------------


def test_on_event_start_returns_event_id(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_start must return the event_id per the LlamaIndex contract."""
    result = handler.on_event_start(CBEventType.LLM, {}, "my-event-id")
    assert result == "my-event-id"


def test_on_event_start_returns_event_id_for_ignored(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_start returns event_id even for ignored event types."""
    result = handler.on_event_start(CBEventType.CHUNKING, {}, "skip-id")
    assert result == "skip-id"


# ---------------------------------------------------------------------------
# on_event_start - parent_id handling
# ---------------------------------------------------------------------------


def test_on_event_start_no_parent_id(handler: AsqavLlamaIndexHandler) -> None:
    """When parent_id is empty, it is not included in context."""
    handler.on_event_start(CBEventType.LLM, {}, "evt-np", "")
    ctx = handler._sign_action.call_args[0][1]
    assert "parent_id" not in ctx


def test_on_event_start_with_parent_id(handler: AsqavLlamaIndexHandler) -> None:
    """When parent_id is provided, it is included in context."""
    handler.on_event_start(CBEventType.LLM, {}, "evt-wp", "parent-abc")
    ctx = handler._sign_action.call_args[0][1]
    assert ctx["parent_id"] == "parent-abc"


# ---------------------------------------------------------------------------
# on_event_end
# ---------------------------------------------------------------------------


def test_on_event_end_llm(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_end signs llamaindex.llm.end for LLM events."""
    handler.on_event_end(CBEventType.LLM, {"completion": "hello"}, "evt-1")
    handler._sign_action.assert_called_once_with(
        "llamaindex.llm.end",
        {
            "event_id": "evt-1",
            "event_type": "llm",
            "payload_keys": ["completion"],
        },
    )


def test_on_event_end_function_call(handler: AsqavLlamaIndexHandler) -> None:
    """on_event_end signs llamaindex.function_call.end for tool events."""
    handler.on_event_end(
        CBEventType.FUNCTION_CALL,
        {"function_call_response": "ok"},
        "evt-4",
    )
    handler._sign_action.assert_called_once_with(
        "llamaindex.function_call.end",
        {
            "event_id": "evt-4",
            "event_type": "function_call",
            "payload_keys": ["function_call_response"],
        },
    )


def test_on_event_end_ignores_chunking(handler: AsqavLlamaIndexHandler) -> None:
    """Chunking end events are not signed."""
    handler.on_event_end(CBEventType.CHUNKING, {}, "evt-x")
    handler._sign_action.assert_not_called()


# ---------------------------------------------------------------------------
# on_event_start/end - user-configured event_starts_to_ignore
# ---------------------------------------------------------------------------


def test_event_starts_to_ignore(handler: AsqavLlamaIndexHandler) -> None:
    """Events in event_starts_to_ignore are skipped even if in _SIGNED_EVENT_TYPES."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        h = AsqavLlamaIndexHandler(
            agent_name="ignore-test",
            event_starts_to_ignore=[CBEventType.LLM],
        )
    h._sign_action = MagicMock(return_value=None)
    h.on_event_start(CBEventType.LLM, {}, "evt-ign")
    h._sign_action.assert_not_called()
    # But end is still signed
    h.on_event_end(CBEventType.LLM, {}, "evt-ign")
    h._sign_action.assert_called_once()


def test_event_ends_to_ignore(handler: AsqavLlamaIndexHandler) -> None:
    """Events in event_ends_to_ignore are skipped on end."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        h = AsqavLlamaIndexHandler(
            agent_name="ignore-end-test",
            event_ends_to_ignore=[CBEventType.RETRIEVE],
        )
    h._sign_action = MagicMock(return_value=None)
    h.on_event_end(CBEventType.RETRIEVE, {}, "evt-ign")
    h._sign_action.assert_not_called()


# ---------------------------------------------------------------------------
# Payload keys
# ---------------------------------------------------------------------------


def test_payload_keys_sorted(handler: AsqavLlamaIndexHandler) -> None:
    """Payload keys are always sorted alphabetically."""
    handler.on_event_start(
        CBEventType.LLM, {"z_key": 1, "a_key": 2, "m_key": 3}, "evt-sort"
    )
    ctx = handler._sign_action.call_args[0][1]
    assert ctx["payload_keys"] == ["a_key", "m_key", "z_key"]


def test_payload_none_gives_empty_keys(handler: AsqavLlamaIndexHandler) -> None:
    """None payload results in empty payload_keys list."""
    handler.on_event_start(CBEventType.LLM, None, "evt-none")
    ctx = handler._sign_action.call_args[0][1]
    assert ctx["payload_keys"] == []


# ---------------------------------------------------------------------------
# start_trace / end_trace
# ---------------------------------------------------------------------------


def test_start_trace_begins_session(handler: AsqavLlamaIndexHandler) -> None:
    """start_trace with a trace_id calls _start_session."""
    handler._start_session = MagicMock()
    handler.start_trace("query-trace-1")
    handler._start_session.assert_called_once()


def test_start_trace_no_id_skips_session(handler: AsqavLlamaIndexHandler) -> None:
    """start_trace without a trace_id does not start a session."""
    handler._start_session = MagicMock()
    handler.start_trace(None)
    handler._start_session.assert_not_called()


def test_end_trace_ends_session(handler: AsqavLlamaIndexHandler) -> None:
    """end_trace closes an active session."""
    handler._session_id = "active-session"
    handler._end_session = MagicMock()
    handler.end_trace("query-trace-1", {"root": ["child-1"]})
    handler._end_session.assert_called_once_with("completed")


def test_end_trace_no_session_is_noop(handler: AsqavLlamaIndexHandler) -> None:
    """end_trace is a no-op when no session is active."""
    handler._session_id = None
    handler._end_session = MagicMock()
    handler.end_trace("query-trace-1")
    handler._end_session.assert_not_called()


# ---------------------------------------------------------------------------
# Fail-open: signing must never block LlamaIndex execution
# ---------------------------------------------------------------------------


def test_fail_open_on_asqav_error() -> None:
    """AsqavError from the real _sign_action is swallowed (fail-open)."""
    from asqav.client import AsqavError

    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent = MagicMock()
        mock_agent.sign.side_effect = AsqavError("network down")
        mock_agent._session_id = None
        mock_agent_cls.create.return_value = mock_agent
        live_handler = AsqavLlamaIndexHandler(agent_name="fail-open-test")

    # None of these should raise
    live_handler.on_event_start(CBEventType.LLM, {"prompt": "hi"}, "e1")
    live_handler.on_event_end(CBEventType.LLM, {"completion": "bye"}, "e1")
    live_handler.on_event_start(CBEventType.QUERY, {}, "e2")
    live_handler.on_event_end(CBEventType.QUERY, {}, "e2")
    live_handler.on_event_start(CBEventType.RETRIEVE, {}, "e3")
    live_handler.on_event_end(CBEventType.RETRIEVE, {}, "e3")
    live_handler.on_event_start(CBEventType.FUNCTION_CALL, {}, "e4")
    live_handler.on_event_end(CBEventType.FUNCTION_CALL, {}, "e4")
    live_handler.on_event_start(CBEventType.AGENT_STEP, {}, "e5")
    live_handler.on_event_end(CBEventType.AGENT_STEP, {}, "e5")

    assert mock_agent.sign.call_count == 10
    assert len(live_handler._signatures) == 0


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def teardown_module() -> None:
    _remove_llamaindex_mocks()
