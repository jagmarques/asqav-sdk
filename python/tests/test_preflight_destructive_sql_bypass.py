"""Destructive SQL bypass fix: augment-match policy coverage.

A data:write:sql: action containing a destructive verb (DELETE, DROP, TRUNCATE,
ALTER) must also match a data:delete:* block policy. The original write match
must still fire too (regression guard). Reads and non-destructive writes must
not be affected.

These tests fail against pre-fix code (bare startswith matcher) and pass after.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import Agent
from asqav.async_client import AsyncAgent

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MOCK_AGENT_DATA = {
    "agent_id": "agent_test123",
    "name": "test-agent",
    "public_key": "pk_test123",
    "key_id": "key_test123",
    "algorithm": "ml-dsa-65",
    "capabilities": [],
    "created_at": 1700000000.0,
}

ASYNC_AGENT = AsyncAgent(
    agent_id="agent_test123",
    name="test-agent",
    public_key="pk_test123",
    key_id="key_test123",
    algorithm="ml-dsa-65",
    capabilities=[],
    created_at=1700000000.0,
)

_DELETE_POLICY = [
    {
        "is_active": True,
        "action_pattern": "data:delete:*",
        "action": "block",
        "name": "no-deletions",
    }
]

_WRITE_SQL_POLICY = [
    {
        "is_active": True,
        "action_pattern": "data:write:sql:*",
        "action": "block",
        "name": "no-sql-writes",
    }
]


# ---------------------------------------------------------------------------
# Sync Agent tests
# ---------------------------------------------------------------------------


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_delete_policy_blocks_write_sql_delete_from(mock_post, mock_get):
    """data:delete:* policy blocks data:write:sql:DELETE FROM users."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _DELETE_POLICY]

    result = agent.preflight("data:write:sql:DELETE FROM users")

    assert result.cleared is False
    assert result.policy_allowed is False
    assert any("no-deletions" in r for r in result.reasons)


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_delete_policy_blocks_write_sql_delete_from_underscore(mock_post, mock_get):
    """data:delete:* policy blocks data:write:sql:DELETE_FROM_users (underscore form)."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _DELETE_POLICY]

    result = agent.preflight("data:write:sql:DELETE_FROM_users")

    assert result.cleared is False
    assert result.policy_allowed is False
    assert any("no-deletions" in r for r in result.reasons)


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_delete_policy_blocks_write_sql_drop(mock_post, mock_get):
    """data:delete:* policy blocks data:write:sql:DROP TABLE t."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _DELETE_POLICY]

    result = agent.preflight("data:write:sql:DROP TABLE t")

    assert result.cleared is False
    assert result.policy_allowed is False


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_delete_policy_blocks_write_sql_truncate(mock_post, mock_get):
    """data:delete:* policy blocks data:write:sql:TRUNCATE TABLE t."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _DELETE_POLICY]

    result = agent.preflight("data:write:sql:TRUNCATE TABLE t")

    assert result.cleared is False
    assert result.policy_allowed is False


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_regression_write_policy_still_blocks_destructive_write(mock_post, mock_get):
    """Regression guard: data:write:sql:* policy still blocks a DELETE write action."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _WRITE_SQL_POLICY]

    result = agent.preflight("data:write:sql:DELETE FROM users")

    assert result.cleared is False
    assert result.policy_allowed is False
    assert any("no-sql-writes" in r for r in result.reasons)


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_negative_delete_policy_does_not_block_benign_write(mock_post, mock_get):
    """data:delete:* policy must not block data:write:sql:INSERT INTO t."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _DELETE_POLICY]

    result = agent.preflight("data:write:sql:INSERT INTO t VALUES (1)")

    assert result.cleared is True
    assert result.policy_allowed is True


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_negative_delete_policy_does_not_block_read_mentioning_delete(mock_post, mock_get):
    """data:delete:* policy must not block data:read:sql:SELECT delete_flag FROM t."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _DELETE_POLICY]

    result = agent.preflight("data:read:sql:SELECT delete_flag FROM t")

    assert result.cleared is True
    assert result.policy_allowed is True


@patch("asqav.client._get")
@patch("asqav.client._post")
@patch("asqav.client._api_key", "sk_test")
def test_negative_deleted_at_is_not_destructive(mock_post, mock_get):
    """deleted_at in a write action does not trigger the destructive verb detection."""
    mock_post.return_value = MOCK_AGENT_DATA
    agent = Agent.create("test-agent")
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _DELETE_POLICY]

    result = agent.preflight("data:write:sql:UPDATE t SET deleted_at=NOW()")

    assert result.cleared is True
    assert result.policy_allowed is True


# ---------------------------------------------------------------------------
# Async Agent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("asqav.async_client._async_get", new_callable=AsyncMock)
async def test_async_delete_policy_blocks_write_sql_delete(mock_get):
    """Async: data:delete:* policy blocks data:write:sql:DELETE FROM users."""
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _DELETE_POLICY]

    result = await ASYNC_AGENT.preflight("data:write:sql:DELETE FROM users")

    assert result.cleared is False
    assert result.policy_allowed is False
    assert any("no-deletions" in r for r in result.reasons)


@pytest.mark.asyncio
@patch("asqav.async_client._async_get", new_callable=AsyncMock)
async def test_async_delete_policy_blocks_write_sql_drop(mock_get):
    """Async: data:delete:* policy blocks data:write:sql:DROP TABLE t."""
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _DELETE_POLICY]

    result = await ASYNC_AGENT.preflight("data:write:sql:DROP TABLE t")

    assert result.cleared is False
    assert result.policy_allowed is False


@pytest.mark.asyncio
@patch("asqav.async_client._async_get", new_callable=AsyncMock)
async def test_async_regression_write_policy_still_fires(mock_get):
    """Async regression guard: data:write:sql:* still blocks a destructive write."""
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _WRITE_SQL_POLICY]

    result = await ASYNC_AGENT.preflight("data:write:sql:DELETE FROM users")

    assert result.cleared is False
    assert result.policy_allowed is False
    assert any("no-sql-writes" in r for r in result.reasons)


@pytest.mark.asyncio
@patch("asqav.async_client._async_get", new_callable=AsyncMock)
async def test_async_negative_delete_policy_does_not_block_insert(mock_get):
    """Async: data:delete:* policy must not block a benign INSERT."""
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _DELETE_POLICY]

    result = await ASYNC_AGENT.preflight("data:write:sql:INSERT INTO t VALUES (1)")

    assert result.cleared is True
    assert result.policy_allowed is True


@pytest.mark.asyncio
@patch("asqav.async_client._async_get", new_callable=AsyncMock)
async def test_async_negative_delete_policy_does_not_block_read(mock_get):
    """Async: data:delete:* does not block data:read:sql:SELECT delete_flag FROM t."""
    mock_get.side_effect = [{"revoked": False, "suspended": False}, _DELETE_POLICY]

    result = await ASYNC_AGENT.preflight("data:read:sql:SELECT delete_flag FROM t")

    assert result.cleared is True
    assert result.policy_allowed is True
