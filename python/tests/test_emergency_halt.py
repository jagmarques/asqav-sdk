"""Tests for emergency_halt functionality."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav.client import emergency_halt

# ---------------------------------------------------------------------------
# emergency_halt tests
# ---------------------------------------------------------------------------


@patch("asqav.client._post")
def test_emergency_halt_calls_correct_endpoint(mock_post: object) -> None:
    """emergency_halt POSTs to /agents/emergency-halt."""
    mock_post.return_value = []  # type: ignore[attr-defined]

    emergency_halt()

    mock_post.assert_called_once_with(  # type: ignore[attr-defined]
        "/agents/emergency-halt",
        {"reason": "emergency"},
    )


@patch("asqav.client._post")
def test_emergency_halt_custom_reason(mock_post: object) -> None:
    """emergency_halt passes custom reason to the API."""
    mock_post.return_value = []  # type: ignore[attr-defined]

    emergency_halt(reason="security breach detected")

    mock_post.assert_called_once_with(  # type: ignore[attr-defined]
        "/agents/emergency-halt",
        {"reason": "security breach detected"},
    )


@patch("asqav.client._post")
def test_emergency_halt_returns_list(mock_post: object) -> None:
    """emergency_halt returns the API response (list of revoked agents)."""
    mock_post.return_value = [  # type: ignore[attr-defined]
        {"agent_id": "a1", "status": "revoked"},
        {"agent_id": "a2", "status": "revoked"},
    ]

    result = emergency_halt(reason="test")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["agent_id"] == "a1"
    assert result[1]["status"] == "revoked"


def test_emergency_halt_exported() -> None:
    """emergency_halt is accessible from the top-level asqav module."""
    assert hasattr(asqav, "emergency_halt")
    assert callable(asqav.emergency_halt)


@patch("asqav.client._post")
def test_emergency_halt_default_reason(mock_post: object) -> None:
    """emergency_halt uses 'emergency' as default reason."""
    mock_post.return_value = []  # type: ignore[attr-defined]

    emergency_halt()

    call_args = mock_post.call_args  # type: ignore[attr-defined]
    assert call_args[0][1]["reason"] == "emergency"
