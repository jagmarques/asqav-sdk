"""AsyncAgent.preflight does not silently clear when a check cannot complete.

Mirrors the sync test_preflight_fail_closed guarantees for the async client. A
transient status/policy fetch error must not leave the agent cleared. The result
reports checks_complete=False and cleared=False so a network blip never reads as
a pass for a revoked or policy-blocked agent.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.async_client import AsyncAgent

AGENT = AsyncAgent(
    agent_id="agent_test123",
    name="test-agent",
    public_key="pk_test123",
    key_id="key_test123",
    algorithm="ml-dsa-65",
    capabilities=[],
    created_at=1700000000.0,
)


@pytest.mark.asyncio
@patch("asqav.async_client._async_get", new_callable=AsyncMock)
async def test_status_fetch_error_does_not_clear(mock_get: AsyncMock) -> None:
    """A status fetch error blocks rather than silently clearing the agent."""
    mock_get.side_effect = [
        ConnectionError("network down"),  # status check
        [],  # policies check
    ]

    result = await AGENT.preflight("llm:call")
    assert result.cleared is False
    assert result.checks_complete is False
    assert "could not verify" in result.explanation.lower()


@pytest.mark.asyncio
@patch("asqav.async_client._async_get", new_callable=AsyncMock)
async def test_policy_fetch_error_does_not_clear(mock_get: AsyncMock) -> None:
    """A policy fetch error blocks rather than silently clearing the agent."""
    mock_get.side_effect = [
        {"revoked": False, "suspended": False},  # status check
        ConnectionError("network down"),  # policies check
    ]

    result = await AGENT.preflight("llm:call")
    assert result.cleared is False
    assert result.checks_complete is False


@pytest.mark.asyncio
@patch("asqav.async_client._async_get", new_callable=AsyncMock)
async def test_non_list_policies_does_not_clear(mock_get: AsyncMock) -> None:
    """A non-list /policies response fails closed instead of silently clearing."""
    mock_get.side_effect = [
        {"revoked": False, "suspended": False},  # status check
        {"policies": []},  # anomalous non-list policies response
    ]

    result = await AGENT.preflight("llm:call")
    assert result.cleared is False
    assert result.checks_complete is False
    assert any("unexpected response" in r for r in result.reasons)


@pytest.mark.asyncio
@patch("asqav.async_client._async_get", new_callable=AsyncMock)
async def test_happy_path_still_clears(mock_get: AsyncMock) -> None:
    """An active agent with no blocking policy clears, checks_complete True."""
    mock_get.side_effect = [
        {"revoked": False, "suspended": False},  # status check
        [],  # policies check
    ]

    result = await AGENT.preflight("llm:call")
    assert result.cleared is True
    assert result.checks_complete is True


@pytest.mark.asyncio
@patch("asqav.async_client._async_get", new_callable=AsyncMock)
async def test_revoked_agent_is_a_real_verdict(mock_get: AsyncMock) -> None:
    """A revoked agent blocks with checks_complete True, distinct from an error."""
    mock_get.side_effect = [
        {"revoked": True, "suspended": False},  # status check
        [],  # policies check
    ]

    result = await AGENT.preflight("llm:call")
    assert result.cleared is False
    assert result.checks_complete is True
    assert "revoked" in result.explanation.lower()
