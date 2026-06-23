"""AsyncAgent.preflight resolves semantic pattern names before policy checks.

Parity test for sync Agent.preflight (client.py:2592): resolve_pattern must
expand a semantic alias (e.g. 'sql-read' -> 'data:read:sql:*') so that a glob
policy matching the resolved name fires correctly.
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

# Policy whose action_pattern matches 'data:read:sql:*' (the resolved form of 'sql-read').
_SQL_READ_BLOCK_POLICY = [
    {
        "is_active": True,
        "action_pattern": "data:read:sql:*",
        "action": "block",
        "name": "no-sql-reads",
    }
]


@pytest.mark.asyncio
@patch("asqav.async_client._async_get", new_callable=AsyncMock)
async def test_preflight_resolves_pattern_before_policy_check(mock_get: AsyncMock) -> None:
    """Passing the semantic alias 'sql-read' triggers a glob policy on the resolved form."""
    mock_get.side_effect = [
        {"revoked": False, "suspended": False},
        _SQL_READ_BLOCK_POLICY,
    ]

    result = await AGENT.preflight("sql-read")
    assert result.cleared is False, (
        "policy on 'data:read:sql:*' must fire when caller passes 'sql-read'"
    )
    assert result.policy_allowed is False
    assert any("no-sql-reads" in r for r in result.reasons)


@pytest.mark.asyncio
@patch("asqav.async_client._async_get", new_callable=AsyncMock)
async def test_preflight_glob_passthrough_still_matches(mock_get: AsyncMock) -> None:
    """Already-expanded glob bypasses resolve_pattern transparently."""
    mock_get.side_effect = [
        {"revoked": False, "suspended": False},
        _SQL_READ_BLOCK_POLICY,
    ]

    result = await AGENT.preflight("data:read:sql:specific_table")
    assert result.cleared is False
    assert result.policy_allowed is False
