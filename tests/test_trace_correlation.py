"""Tests for trace correlation support in agent.sign()."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav.client import Agent, SignatureResponse, generate_trace_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_SIGN_RESPONSE: dict = {
    "signature": "sig_abc123",
    "signature_id": "sid_001",
    "action_id": "act_001",
    "timestamp": "2026-04-13T12:00:00",
    "verification_url": "https://asqav.com/verify/sid_001",
}


def _make_agent() -> Agent:
    return Agent(
        agent_id="agent_test",
        name="test-agent",
        public_key="pk_test",
        key_id="kid_test",
        algorithm="ml-dsa-65",
        capabilities=["read:data"],
        created_at=1700000000.0,
    )


# ---------------------------------------------------------------------------
# generate_trace_id tests
# ---------------------------------------------------------------------------


def test_generate_trace_id_returns_hex_string() -> None:
    """generate_trace_id returns a 32-char hex string (uuid4 without hyphens)."""
    tid = generate_trace_id()
    assert isinstance(tid, str)
    assert len(tid) == 32
    int(tid, 16)  # should not raise


def test_generate_trace_id_is_unique() -> None:
    """Each call produces a different trace ID."""
    ids = {generate_trace_id() for _ in range(100)}
    assert len(ids) == 100


def test_generate_trace_id_exported() -> None:
    """generate_trace_id is accessible from the top-level asqav module."""
    assert hasattr(asqav, "generate_trace_id")
    assert callable(asqav.generate_trace_id)


# ---------------------------------------------------------------------------
# sign() with trace_id and parent_id
# ---------------------------------------------------------------------------


@patch("asqav.client._post")
def test_sign_with_trace_id(mock_post: object) -> None:
    """sign() includes _trace_id in context when trace_id is provided."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]
    agent = _make_agent()

    agent.sign("read:data", trace_id="abc123")

    call_args = mock_post.call_args  # type: ignore[attr-defined]
    body = call_args[0][1]
    assert body["context"]["_trace_id"] == "abc123"


@patch("asqav.client._post")
def test_sign_with_parent_id(mock_post: object) -> None:
    """sign() includes _parent_id in context when parent_id is provided."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]
    agent = _make_agent()

    agent.sign("read:data", parent_id="parent_xyz")

    call_args = mock_post.call_args  # type: ignore[attr-defined]
    body = call_args[0][1]
    assert body["context"]["_parent_id"] == "parent_xyz"


@patch("asqav.client._post")
def test_sign_with_both_trace_and_parent(mock_post: object) -> None:
    """sign() includes both _trace_id and _parent_id when both provided."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]
    agent = _make_agent()

    agent.sign("read:data", trace_id="t1", parent_id="p1")

    call_args = mock_post.call_args  # type: ignore[attr-defined]
    body = call_args[0][1]
    assert body["context"]["_trace_id"] == "t1"
    assert body["context"]["_parent_id"] == "p1"


@patch("asqav.client._post")
def test_sign_trace_merges_with_existing_context(mock_post: object) -> None:
    """Trace fields merge into user-provided context without overwriting."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]
    agent = _make_agent()

    agent.sign("read:data", context={"user": "alice"}, trace_id="t1")

    call_args = mock_post.call_args  # type: ignore[attr-defined]
    body = call_args[0][1]
    assert body["context"]["user"] == "alice"
    assert body["context"]["_trace_id"] == "t1"


@patch("asqav.client._post")
def test_sign_without_trace_no_trace_keys(mock_post: object) -> None:
    """When no trace_id/parent_id given, context has no _trace_id/_parent_id."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]
    agent = _make_agent()

    agent.sign("read:data", context={"user": "alice"})

    call_args = mock_post.call_args  # type: ignore[attr-defined]
    body = call_args[0][1]
    assert "_trace_id" not in body["context"]
    assert "_parent_id" not in body["context"]


@patch("asqav.client._post")
def test_sign_returns_signature_response(mock_post: object) -> None:
    """sign() still returns a proper SignatureResponse with trace params."""
    mock_post.return_value = MOCK_SIGN_RESPONSE  # type: ignore[attr-defined]
    agent = _make_agent()

    result = agent.sign("read:data", trace_id="t1", parent_id="p1")

    assert isinstance(result, SignatureResponse)
    assert result.signature_id == "sid_001"
