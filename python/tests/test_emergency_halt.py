"""Tests for emergency_halt functionality."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav.client import emergency_halt

# === emergency_halt tests ===


@patch("asqav.client._post")
def test_emergency_halt_calls_correct_endpoint(mock_post: object) -> None:
    """emergency_halt POSTs to /orgs/{org_id}/emergency-halt."""
    mock_post.return_value = {"emergency_halt": True}  # type: ignore[attr-defined]

    emergency_halt("org-123")

    mock_post.assert_called_once_with(  # type: ignore[attr-defined]
        "/orgs/org-123/emergency-halt",
        {"reason": "emergency"},
    )


@patch("asqav.client._post")
def test_emergency_halt_custom_reason(mock_post: object) -> None:
    """emergency_halt passes custom reason to the API."""
    mock_post.return_value = {"emergency_halt": True}  # type: ignore[attr-defined]

    emergency_halt("org-456", reason="security breach detected")

    mock_post.assert_called_once_with(  # type: ignore[attr-defined]
        "/orgs/org-456/emergency-halt",
        {"reason": "security breach detected"},
    )


@patch("asqav.client._post")
def test_emergency_halt_returns_dict(mock_post: object) -> None:
    """emergency_halt returns the API response dict."""
    mock_post.return_value = {  # type: ignore[attr-defined]
        "organization_id": "org-123",
        "emergency_halt": True,
        "emergency_halt_reason": "test",
    }

    result = emergency_halt("org-123", reason="test")

    assert isinstance(result, dict)
    assert result["emergency_halt"] is True
    assert result["organization_id"] == "org-123"


def test_emergency_halt_exported() -> None:
    """emergency_halt is accessible from the top-level asqav module."""
    assert hasattr(asqav, "emergency_halt")
    assert callable(asqav.emergency_halt)


@patch("asqav.client._post")
def test_emergency_halt_default_reason(mock_post: object) -> None:
    """emergency_halt uses 'emergency' as default reason."""
    mock_post.return_value = {"emergency_halt": True}  # type: ignore[attr-defined]

    emergency_halt("org-789")

    call_args = mock_post.call_args  # type: ignore[attr-defined]
    assert call_args[0][1]["reason"] == "emergency"


@patch("asqav.client._post")
def test_emergency_halt_path_includes_org_id(mock_post: object) -> None:
    """emergency_halt path includes the org_id (proves /agents/ path is gone)."""
    mock_post.return_value = {"emergency_halt": True}  # type: ignore[attr-defined]

    emergency_halt("my-org")

    path = mock_post.call_args[0][0]  # type: ignore[attr-defined]
    assert "/orgs/my-org/emergency-halt" == path
    assert "/agents/" not in path
