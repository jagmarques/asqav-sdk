"""Tests for reproducibility metadata in agent.sign()."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav.client import Agent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_SIGN_RESPONSE: dict = {
    "signature": "sig_abc123",
    "signature_id": "sid_001",
    "action_id": "act_001",
    "timestamp": "2026-04-12T10:00:00",
    "verification_url": "https://api.asqav.com/verify/sid_001",
}


# ---------------------------------------------------------------------------
# Reproducibility metadata tests
# ---------------------------------------------------------------------------


@patch("asqav.client._post")
def test_sign_with_system_prompt_hash(mock_post: object) -> None:
    """sign() includes _system_prompt_hash in context when provided."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]

    agent = Agent.__new__(Agent)
    agent.agent_id = "agent_test"
    agent._session_id = "sess_001"

    agent.sign(
        "api:call",
        context={"model": "gpt-4"},
        system_prompt_hash="abc123hash",
    )

    call_body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    assert call_body["context"]["_system_prompt_hash"] == "abc123hash"
    assert call_body["context"]["model"] == "gpt-4"


@patch("asqav.client._post")
def test_sign_with_model_params(mock_post: object) -> None:
    """sign() includes _model_params in context when provided."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]

    agent = Agent.__new__(Agent)
    agent.agent_id = "agent_test"
    agent._session_id = "sess_001"

    params = {"temperature": 0.7, "top_p": 0.9}
    agent.sign("api:call", model_params=params)

    call_body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    assert call_body["context"]["_model_params"] == params


@patch("asqav.client._post")
def test_sign_with_tool_inputs_hash(mock_post: object) -> None:
    """sign() includes _tool_inputs_hash in context when provided."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]

    agent = Agent.__new__(Agent)
    agent.agent_id = "agent_test"
    agent._session_id = "sess_001"

    agent.sign("api:call", tool_inputs_hash="def456hash")

    call_body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    assert call_body["context"]["_tool_inputs_hash"] == "def456hash"


@patch("asqav.client._post")
def test_sign_with_all_metadata(mock_post: object) -> None:
    """sign() includes all three metadata fields when all are provided."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]

    agent = Agent.__new__(Agent)
    agent.agent_id = "agent_test"
    agent._session_id = "sess_001"

    agent.sign(
        "api:call",
        context={"key": "value"},
        system_prompt_hash="prompt_hash",
        model_params={"temperature": 0.5},
        tool_inputs_hash="inputs_hash",
    )

    call_body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    ctx = call_body["context"]
    assert ctx["_system_prompt_hash"] == "prompt_hash"
    assert ctx["_model_params"] == {"temperature": 0.5}
    assert ctx["_tool_inputs_hash"] == "inputs_hash"
    assert ctx["key"] == "value"


@patch("asqav.client._post")
def test_sign_without_metadata_no_underscore_keys(mock_post: object) -> None:
    """sign() does not add metadata keys when none are provided."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]

    agent = Agent.__new__(Agent)
    agent.agent_id = "agent_test"
    agent._session_id = "sess_001"

    agent.sign("api:call", context={"key": "value"})

    call_body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    ctx = call_body["context"]
    assert "_system_prompt_hash" not in ctx
    assert "_model_params" not in ctx
    assert "_tool_inputs_hash" not in ctx


@patch("asqav.client._post")
def test_sign_metadata_does_not_mutate_original_context(mock_post: object) -> None:
    """sign() does not mutate the original context dict passed in."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]

    agent = Agent.__new__(Agent)
    agent.agent_id = "agent_test"
    agent._session_id = "sess_001"

    original_ctx = {"key": "value"}
    agent.sign("api:call", context=original_ctx, system_prompt_hash="hash123")

    assert "_system_prompt_hash" not in original_ctx


@patch("asqav.client._post")
def test_sign_with_none_context_and_metadata(mock_post: object) -> None:
    """sign() creates context dict from scratch when context is None and metadata provided."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]

    agent = Agent.__new__(Agent)
    agent.agent_id = "agent_test"
    agent._session_id = "sess_001"

    agent.sign("api:call", system_prompt_hash="hash_value")

    call_body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    assert call_body["context"]["_system_prompt_hash"] == "hash_value"


# ---------------------------------------------------------------------------
# Pattern resolution in sign()
# ---------------------------------------------------------------------------


@patch("asqav.client._post")
def test_sign_resolves_semantic_pattern(mock_post: object) -> None:
    """sign() expands semantic pattern names to globs."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]

    agent = Agent.__new__(Agent)
    agent.agent_id = "agent_test"
    agent._session_id = "sess_001"

    agent.sign("sql-destructive", context={"table": "users"})

    call_body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    assert call_body["action_type"] == "data:delete:*"


@patch("asqav.client._post")
def test_sign_passes_through_unknown_pattern(mock_post: object) -> None:
    """sign() passes through action types that are not semantic patterns."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]

    agent = Agent.__new__(Agent)
    agent.agent_id = "agent_test"
    agent._session_id = "sess_001"

    agent.sign("custom:action:type")

    call_body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    assert call_body["action_type"] == "custom:action:type"
