"""Tests for the one-call govern() entrypoint."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav.client import Agent, govern

# === Fixtures ===

MOCK_AGENT_DATA: dict = {
    "agent_id": "agt_govern_01",
    "name": "my-agent",
    "public_key": "pk_abc",
    "key_id": "kid_abc",
    "algorithm": "ml-dsa-65",
    "capabilities": [],
    "created_at": "2026-01-01T00:00:00Z",
}


def _make_agent() -> Agent:
    return Agent(
        agent_id=MOCK_AGENT_DATA["agent_id"],
        name=MOCK_AGENT_DATA["name"],
        public_key=MOCK_AGENT_DATA["public_key"],
        key_id=MOCK_AGENT_DATA["key_id"],
        algorithm=MOCK_AGENT_DATA["algorithm"],
        capabilities=MOCK_AGENT_DATA["capabilities"],
        created_at=0.0,
    )


# === Tests ===


def test_govern_calls_init_and_create() -> None:
    """govern() calls init() then Agent.create() with the right args."""
    agent = _make_agent()

    with patch("asqav.client.init") as mock_init, patch.object(
        Agent, "create", return_value=agent
    ) as mock_create:
        result = govern(api_key="sk_test", agent_name="my-agent")

    mock_init.assert_called_once_with(
        api_key="sk_test",
        base_url=None,
        mode="auto",
        org_salt=None,
    )
    mock_create.assert_called_once_with(
        "my-agent", algorithm="ml-dsa-65", capabilities=None
    )
    assert result is agent
    assert result.agent_id == "agt_govern_01"


def test_govern_retrieves_existing_agent_by_id() -> None:
    """When agent_id is supplied govern() calls Agent.get, not Agent.create."""
    agent = _make_agent()

    with patch("asqav.client.init"), patch.object(
        Agent, "get", return_value=agent
    ) as mock_get, patch.object(Agent, "create") as mock_create:
        result = govern(api_key="sk_test", agent_id="agt_govern_01")

    mock_get.assert_called_once_with("agt_govern_01")
    mock_create.assert_not_called()
    assert result is agent


def test_govern_passes_capabilities_and_algorithm() -> None:
    """govern() forwards algorithm and capabilities to Agent.create."""
    agent = _make_agent()

    with patch("asqav.client.init"), patch.object(
        Agent, "create", return_value=agent
    ) as mock_create:
        govern(
            api_key="sk_test",
            agent_name="cap-agent",
            algorithm="ml-dsa-44",
            capabilities=["read", "write"],
        )

    mock_create.assert_called_once_with(
        "cap-agent", algorithm="ml-dsa-44", capabilities=["read", "write"]
    )


def test_govern_exported_from_package() -> None:
    """asqav.govern is importable directly from the package."""
    assert callable(asqav.govern)


def test_govern_default_agent_name() -> None:
    """govern() uses 'default-agent' when agent_name is omitted."""
    agent = _make_agent()

    with patch("asqav.client.init"), patch.object(
        Agent, "create", return_value=agent
    ) as mock_create:
        govern(api_key="sk_test")

    mock_create.assert_called_once_with(
        "default-agent", algorithm="ml-dsa-65", capabilities=None
    )
