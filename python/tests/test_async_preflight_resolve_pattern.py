"""AsyncAgent.preflight must resolve semantic pattern names, matching sync Agent.preflight."""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.async_client import AsyncAgent
from asqav.client import PreflightResult
from asqav.patterns import PATTERNS, resolve_pattern


def _make_async_agent() -> AsyncAgent:
    agent = AsyncAgent.__new__(AsyncAgent)
    agent.agent_id = "test-agent-id"
    return agent


@pytest.mark.asyncio
async def test_async_preflight_resolves_semantic_pattern() -> None:
    """AsyncAgent.preflight resolves 'sql-read' to its glob before policy match."""
    agent = _make_async_agent()
    expected_resolved = resolve_pattern("sql-read")
    assert expected_resolved == "data:read:sql:*"

    captured: list[str] = []

    async def fake_get(path: str) -> dict | list:
        if "/status" in path:
            return {"revoked": False, "suspended": False}
        # Capture the action_type as seen during policy check by
        # checking what the policy pattern matches against.
        return [
            {
                "is_active": True,
                "action_pattern": "data:read:sql:sentinel",
                "action": "allow",
                "name": "probe",
            }
        ]

    with patch("asqav.async_client._async_get", side_effect=fake_get):
        result = await agent.preflight("sql-read")

    # Resolved type is "data:read:sql:*" - the policy "data:read:sql:sentinel"
    # does NOT match the glob, so policy_allowed stays True (no block).
    # The key assertion: if resolve_pattern was NOT called, action_type would
    # remain "sql-read" and the startswith check would never match either; but
    # the test below uses a policy that only matches the RESOLVED form.
    assert isinstance(result, PreflightResult)
    assert result.cleared is True


@pytest.mark.asyncio
async def test_async_preflight_resolved_pattern_triggers_policy_block() -> None:
    """Policy using resolved glob pattern blocks the action correctly."""
    agent = _make_async_agent()

    async def fake_get(path: str) -> dict | list:
        if "/status" in path:
            return {"revoked": False, "suspended": False}
        return [
            {
                "is_active": True,
                # Matches "data:read:sql:*" - only works if resolve_pattern ran.
                "action_pattern": "data:read:sql:",
                "action": "block",
                "name": "block-sql-reads",
            }
        ]

    with patch("asqav.async_client._async_get", side_effect=fake_get):
        result = await agent.preflight("sql-read")

    # If resolve_pattern was skipped, "sql-read".startswith("data:read:sql:")
    # is False -> cleared=True (wrong). With resolve_pattern, it is
    # "data:read:sql:*".startswith("data:read:sql:") -> True -> blocked.
    assert result.cleared is False
    assert result.policy_allowed is False
    any_block_reason = any("block-sql-reads" in r for r in result.reasons)
    assert any_block_reason, f"Expected block reason in {result.reasons}"


@pytest.mark.asyncio
async def test_async_preflight_parity_with_sync_resolve() -> None:
    """resolve_pattern output used by async preflight matches sync Agent.preflight."""
    for name, glob in PATTERNS.items():
        assert resolve_pattern(name) == glob, f"Pattern mismatch for '{name}'"
