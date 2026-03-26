"""Tests for retry logic and async client."""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import APIError, AuthenticationError, RateLimitError
from asqav.retry import _calculate_delay, _is_retryable, with_async_retry, with_retry


# ---------------------------------------------------------------------------
# Retry: rate limit error is retried
# ---------------------------------------------------------------------------


@patch("asqav.retry.time.sleep")
def test_retry_on_rate_limit(mock_sleep: MagicMock) -> None:
    """RateLimitError triggers retry with backoff, succeeds on second attempt."""
    call_count = 0

    @with_retry(max_retries=3, base_delay=1.0, jitter=False)
    def flaky_call() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("Rate limit exceeded")
        return "success"

    result = flaky_call()
    assert result == "success"
    assert call_count == 2
    mock_sleep.assert_called_once_with(1.0)


# ---------------------------------------------------------------------------
# Retry: server error (5xx) is retried
# ---------------------------------------------------------------------------


@patch("asqav.retry.time.sleep")
def test_retry_on_server_error(mock_sleep: MagicMock) -> None:
    """APIError with 500 status triggers retry."""
    call_count = 0

    @with_retry(max_retries=3, base_delay=0.5, jitter=False)
    def flaky_call() -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise APIError("Internal server error", status_code=500)
        return "recovered"

    result = flaky_call()
    assert result == "recovered"
    assert call_count == 3
    assert mock_sleep.call_count == 2
    # First retry: 0.5 * 2^0 = 0.5
    assert mock_sleep.call_args_list[0][0][0] == 0.5
    # Second retry: 0.5 * 2^1 = 1.0
    assert mock_sleep.call_args_list[1][0][0] == 1.0


# ---------------------------------------------------------------------------
# Retry: auth error is NOT retried
# ---------------------------------------------------------------------------


def test_no_retry_on_auth_error() -> None:
    """AuthenticationError is raised immediately without retry."""
    call_count = 0

    @with_retry(max_retries=3, base_delay=0.5)
    def auth_call() -> str:
        nonlocal call_count
        call_count += 1
        raise AuthenticationError("Invalid API key")

    with pytest.raises(AuthenticationError, match="Invalid API key"):
        auth_call()
    assert call_count == 1  # No retry


# ---------------------------------------------------------------------------
# Retry: client error (4xx, not 429) is NOT retried
# ---------------------------------------------------------------------------


def test_no_retry_on_client_error() -> None:
    """APIError with 400 status is raised immediately without retry."""
    call_count = 0

    @with_retry(max_retries=3, base_delay=0.5)
    def bad_request() -> str:
        nonlocal call_count
        call_count += 1
        raise APIError("Bad request", status_code=400)

    with pytest.raises(APIError, match="Bad request"):
        bad_request()
    assert call_count == 1  # No retry


# ---------------------------------------------------------------------------
# Retry: exhaustion raises last error
# ---------------------------------------------------------------------------


@patch("asqav.retry.time.sleep")
def test_retry_exhaustion(mock_sleep: MagicMock) -> None:
    """After max_retries attempts, the last error is raised."""
    call_count = 0

    @with_retry(max_retries=2, base_delay=0.5, jitter=False)
    def always_fails() -> str:
        nonlocal call_count
        call_count += 1
        raise RateLimitError("Rate limit exceeded")

    with pytest.raises(RateLimitError, match="Rate limit exceeded"):
        always_fails()
    assert call_count == 3  # initial + 2 retries
    assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# Retry: connection error is retried
# ---------------------------------------------------------------------------


@patch("asqav.retry.time.sleep")
def test_retry_on_connection_error(mock_sleep: MagicMock) -> None:
    """ConnectionError triggers retry."""
    call_count = 0

    @with_retry(max_retries=3, base_delay=0.5, jitter=False)
    def unstable_call() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Connection refused")
        return "connected"

    result = unstable_call()
    assert result == "connected"
    assert call_count == 2
    mock_sleep.assert_called_once_with(0.5)


# ---------------------------------------------------------------------------
# Async: create agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("asqav.async_client._async_post")
async def test_async_create_agent(mock_post: AsyncMock) -> None:
    """AsyncAgent.create calls _async_post and returns AsyncAgent."""
    from asqav.async_client import AsyncAgent

    mock_post.return_value = {
        "agent_id": "agent_async_001",
        "name": "test-async",
        "public_key": "pk_test",
        "key_id": "kid_test",
        "algorithm": "ml-dsa-65",
        "capabilities": ["read", "write"],
        "created_at": "2026-03-26T10:00:00",
    }

    agent = await AsyncAgent.create("test-async")

    mock_post.assert_called_once_with(
        "/agents/create",
        {
            "name": "test-async",
            "algorithm": "ml-dsa-65",
            "capabilities": [],
        },
    )
    assert agent.agent_id == "agent_async_001"
    assert agent.name == "test-async"
    assert isinstance(agent, AsyncAgent)


# ---------------------------------------------------------------------------
# Async: sign action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("asqav.async_client._async_post")
async def test_async_sign(mock_post: AsyncMock) -> None:
    """AsyncAgent.sign calls _async_post and returns SignatureResponse."""
    from asqav.async_client import AsyncAgent
    from asqav.client import SignatureResponse

    mock_post.return_value = {
        "signature": "sig_abc123",
        "signature_id": "sigid_001",
        "action_id": "act_001",
        "timestamp": 1711440000.0,
        "verification_url": "https://api.asqav.com/verify/sigid_001",
    }

    agent = AsyncAgent(
        agent_id="agent_async_001",
        name="test",
        public_key="pk",
        key_id="kid",
        algorithm="ml-dsa-65",
        capabilities=[],
        created_at=0.0,
    )

    result = await agent.sign("api:call", {"model": "gpt-4"})

    mock_post.assert_called_once_with(
        "/agents/agent_async_001/sign",
        {
            "action_type": "api:call",
            "context": {"model": "gpt-4"},
            "session_id": None,
        },
    )
    assert isinstance(result, SignatureResponse)
    assert result.signature == "sig_abc123"
    assert result.signature_id == "sigid_001"


# ---------------------------------------------------------------------------
# Async: verify signature
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_verify() -> None:
    """AsyncAgent.verify calls httpx.AsyncClient.get and returns VerificationResponse."""
    from unittest.mock import patch as sync_patch

    from asqav.async_client import AsyncAgent
    from asqav.client import VerificationResponse

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "signature_id": "sigid_001",
        "agent_id": "agent_001",
        "agent_name": "test-agent",
        "action_id": "act_001",
        "action_type": "api:call",
        "payload": {"model": "gpt-4"},
        "signature": "sig_abc",
        "algorithm": "ml-dsa-65",
        "signed_at": 1711440000.0,
        "verified": True,
        "verification_url": "https://api.asqav.com/verify/sigid_001",
    }

    agent = AsyncAgent(
        agent_id="agent_001",
        name="test",
        public_key="pk",
        key_id="kid",
        algorithm="ml-dsa-65",
        capabilities=[],
        created_at=0.0,
    )

    # Mock _get_config to return valid config
    with sync_patch("asqav.async_client._get_config", return_value=("https://api.asqav.com/api/v1", "sk_test")):
        # Mock httpx.AsyncClient
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with sync_patch("asqav.async_client.httpx.AsyncClient", return_value=mock_client):
            result = await agent.verify("sigid_001")

    assert isinstance(result, VerificationResponse)
    assert result.verified is True
    assert result.signature_id == "sigid_001"
    assert result.agent_name == "test-agent"


# ---------------------------------------------------------------------------
# Async: retry works with async functions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("asqav.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_async_retry_on_rate_limit(mock_sleep: AsyncMock) -> None:
    """with_async_retry retries RateLimitError with backoff."""
    call_count = 0

    @with_async_retry(max_retries=3, base_delay=1.0, jitter=False)
    async def flaky_async() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("Rate limit exceeded")
        return "async_success"

    result = await flaky_async()
    assert result == "async_success"
    assert call_count == 2
    mock_sleep.assert_called_once_with(1.0)
