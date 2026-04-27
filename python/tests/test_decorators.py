"""Tests for asqav sign decorator and session context manager."""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav import client as asqav_client
from asqav.decorators import Session, async_session, session, sign

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_AGENT_RESPONSE: dict = {
    "agent_id": "agent_test_123",
    "name": "test-agent",
    "public_key": "pk_test",
    "key_id": "key_test",
    "algorithm": "ml-dsa-65",
    "capabilities": [],
    "created_at": 1700000000.0,
}

MOCK_SIGN_RESPONSE: dict = {
    "signature": "sig_abc123",
    "signature_id": "sid_abc123",
    "action_id": "act_abc123",
    "timestamp": 1700000001.0,
    "verification_url": "https://api.asqav.com/verify/sid_abc123",
}

MOCK_SESSION_RESPONSE: dict = {
    "session_id": "sess_abc123",
    "agent_id": "agent_test_123",
    "status": "active",
    "started_at": "2026-03-26T09:00:00Z",
}

MOCK_SESSION_END_RESPONSE: dict = {
    "session_id": "sess_abc123",
    "agent_id": "agent_test_123",
    "status": "completed",
    "started_at": "2026-03-26T09:00:00Z",
    "ended_at": "2026-03-26T09:05:00Z",
}

MOCK_SESSION_ERROR_END_RESPONSE: dict = {
    "session_id": "sess_abc123",
    "agent_id": "agent_test_123",
    "status": "error",
    "started_at": "2026-03-26T09:00:00Z",
    "ended_at": "2026-03-26T09:05:00Z",
}


def _mock_post_side_effect(path: str, data: dict) -> dict:
    """Route mock _post calls to appropriate fixture responses."""
    if path == "/agents/create":
        return MOCK_AGENT_RESPONSE
    if "/sign" in path:
        return MOCK_SIGN_RESPONSE
    if path == "/sessions/":
        return MOCK_SESSION_RESPONSE
    return {}


def _mock_patch_side_effect(path: str, data: dict) -> dict:
    """Route mock _patch calls to appropriate fixture responses."""
    if "sessions" in path:
        if data.get("status") == "error":
            return MOCK_SESSION_ERROR_END_RESPONSE
        return MOCK_SESSION_END_RESPONSE
    return {}


def _reset_global_agent() -> None:
    """Reset the global agent between tests."""
    asqav_client._global_agent = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("asqav.client._patch", side_effect=_mock_patch_side_effect)
@patch("asqav.client._post", side_effect=_mock_post_side_effect)
def test_sign_decorator_sync(mock_post: MagicMock, mock_patch: MagicMock) -> None:
    """@asqav.sign on sync function signs call and result."""
    _reset_global_agent()

    @sign
    def add(a: int, b: int) -> int:
        return a + b

    result = add(2, 3)

    assert result == 5

    # Should have called _post 3 times: create agent, sign call, sign result
    assert mock_post.call_count == 3

    # Second call is the function:call sign
    call_args = mock_post.call_args_list[1]
    assert "/sign" in call_args[0][0]
    assert call_args[0][1]["action_type"] == "function:call"
    assert call_args[0][1]["context"]["function"] == "add"

    # Third call is the function:result sign
    result_args = mock_post.call_args_list[2]
    assert result_args[0][1]["action_type"] == "function:result"
    assert result_args[0][1]["context"]["success"] is True


@patch("asqav.client._patch", side_effect=_mock_patch_side_effect)
@patch("asqav.client._post", side_effect=_mock_post_side_effect)
def test_sign_decorator_async(mock_post: MagicMock, mock_patch: MagicMock) -> None:
    """@asqav.sign on async function works with asyncio.run()."""
    _reset_global_agent()

    @sign
    async def fetch(url: str) -> str:
        return f"data from {url}"

    result = asyncio.run(fetch("https://example.com"))

    assert result == "data from https://example.com"

    # create agent + sign call + sign result = 3
    assert mock_post.call_count == 3

    call_args = mock_post.call_args_list[1]
    assert call_args[0][1]["action_type"] == "function:call"
    assert call_args[0][1]["context"]["function"] == "fetch"


@patch("asqav.client._patch", side_effect=_mock_patch_side_effect)
@patch("asqav.client._post", side_effect=_mock_post_side_effect)
def test_sign_decorator_with_action_type(mock_post: MagicMock, mock_patch: MagicMock) -> None:
    """@asqav.sign(action_type="deploy:prod") overrides default action type."""
    _reset_global_agent()

    @sign(action_type="deploy:prod")
    def deploy(env: str) -> str:
        return f"deployed to {env}"

    result = deploy("production")

    assert result == "deployed to production"

    # Second call should use the custom action type
    call_args = mock_post.call_args_list[1]
    assert call_args[0][1]["action_type"] == "deploy:prod"


@patch("asqav.client._patch", side_effect=_mock_patch_side_effect)
@patch("asqav.client._post", side_effect=_mock_post_side_effect)
def test_sign_decorator_reraises_exception(mock_post: MagicMock, mock_patch: MagicMock) -> None:
    """Decorated function that raises propagates exception and signs error."""
    _reset_global_agent()

    @sign
    def fail() -> None:
        raise ValueError("something broke")

    try:
        fail()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "something broke"

    # create agent + sign call + sign error = 3
    assert mock_post.call_count == 3

    error_args = mock_post.call_args_list[2]
    assert error_args[0][1]["action_type"] == "function:error"
    assert "something broke" in error_args[0][1]["context"]["error"]


@patch("asqav.client._patch", side_effect=_mock_patch_side_effect)
@patch("asqav.client._post", side_effect=_mock_post_side_effect)
def test_session_context_manager(mock_post: MagicMock, mock_patch: MagicMock) -> None:
    """with asqav.session() as s: s.sign(...) groups signs under a session."""
    _reset_global_agent()

    with session() as s:
        assert s.session_id == "sess_abc123"
        s.sign("data:read", {"file": "config.json"})
        s.sign("data:write", {"file": "output.json"})

    # Calls: create agent, start session, sign x2, end session (via _patch)
    assert mock_post.call_count == 4  # create + start_session + 2 signs
    assert mock_patch.call_count == 1  # end_session

    # Verify start session was called
    session_call = mock_post.call_args_list[1]
    assert session_call[0][0] == "/sessions/"

    # Verify end session was called with "completed"
    end_call = mock_patch.call_args_list[0]
    assert end_call[0][1]["status"] == "completed"


@patch("asqav.client._patch", side_effect=_mock_patch_side_effect)
@patch("asqav.client._post", side_effect=_mock_post_side_effect)
def test_session_signs_error_on_exception(
    mock_post: MagicMock, mock_patch: MagicMock
) -> None:
    """Session that encounters an exception signs error before ending."""
    _reset_global_agent()

    try:
        with session() as s:
            s.sign("data:read", {"file": "config.json"})
            raise RuntimeError("connection lost")
    except RuntimeError:
        pass

    # create agent + start session + sign data:read + sign session:error = 4
    assert mock_post.call_count == 4

    # Last _post call should be the error sign
    error_call = mock_post.call_args_list[3]
    assert error_call[0][1]["action_type"] == "session:error"
    assert "connection lost" in error_call[0][1]["context"]["error"]

    # Session should have ended with "error" status
    assert mock_patch.call_count == 1
    end_call = mock_patch.call_args_list[0]
    assert end_call[0][1]["status"] == "error"


def test_existing_secure_still_works() -> None:
    """Verify asqav.secure is still importable and callable (backward compat)."""
    assert hasattr(asqav, "secure")
    assert callable(asqav.secure)
    assert hasattr(asqav, "secure_async")
    assert callable(asqav.secure_async)

    # Also verify new exports exist alongside old ones
    assert hasattr(asqav, "sign")
    assert hasattr(asqav, "session")
    assert hasattr(asqav, "async_session")
