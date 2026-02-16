"""Tests for asqav SDK client -- threshold (multi-key authorization) methods."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav.client import (
    ThresholdApproveResponse,
    ThresholdSessionResponse,
    ThresholdSignatureDetail,
    _parse_threshold_session,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_SESSION_RESPONSE: dict = {
    "session_id": "thr_abc123",
    "config_id": "cfg_xyz789",
    "agent_id": "agent_001",
    "action_type": "deploy:production",
    "action_params": None,
    "threshold_required": 2,
    "signatures_collected": 0,
    "status": "pending",
    "signatures": [],
    "policy_attestation_hash": None,
    "created_at": "2026-02-16T12:00:00",
    "expires_at": "2026-02-17T12:00:00",
    "resolved_at": None,
}

MOCK_SESSION_WITH_PARAMS: dict = {
    **MOCK_SESSION_RESPONSE,
    "action_params": {"target": "us-east-1", "version": "2.0"},
}

MOCK_SESSION_APPROVED: dict = {
    **MOCK_SESSION_RESPONSE,
    "status": "approved",
    "signatures_collected": 2,
    "signatures": [
        {
            "entity_id": "ent_001",
            "entity_name": "Agent Key",
            "entity_class": "A",
            "signed_at": "2026-02-16T12:01:00",
        },
        {
            "entity_id": "ent_002",
            "entity_name": "Human Approver",
            "entity_class": "B",
            "signed_at": "2026-02-16T12:02:00",
        },
    ],
    "policy_attestation_hash": "abcdef1234567890",
    "resolved_at": "2026-02-16T12:02:00",
}

MOCK_APPROVE_RESPONSE: dict = {
    "session_id": "thr_abc123",
    "entity_id": "ent_001",
    "signatures_collected": 1,
    "threshold_required": 2,
    "status": "pending",
    "threshold_met": False,
}

MOCK_APPROVE_THRESHOLD_MET: dict = {
    "session_id": "thr_abc123",
    "entity_id": "ent_002",
    "signatures_collected": 2,
    "threshold_required": 2,
    "status": "approved",
    "threshold_met": True,
}


# ---------------------------------------------------------------------------
# request_action tests
# ---------------------------------------------------------------------------


@patch("asqav.client._post")
def test_request_action(mock_post: object) -> None:
    """request_action calls POST /threshold/sessions with correct body."""
    mock_post.return_value = MOCK_SESSION_RESPONSE  # type: ignore[attr-defined]

    result = asqav.request_action(
        agent_id="agent_001",
        action_type="deploy:production",
    )

    mock_post.assert_called_once_with(  # type: ignore[attr-defined]
        "/threshold/sessions",
        {"agent_id": "agent_001", "action_type": "deploy:production"},
    )
    assert isinstance(result, ThresholdSessionResponse)
    assert result.session_id == "thr_abc123"
    assert result.agent_id == "agent_001"
    assert result.action_type == "deploy:production"
    assert result.threshold_required == 2
    assert result.signatures_collected == 0
    assert result.status == "pending"
    assert result.signatures == []
    assert result.action_params is None


@patch("asqav.client._post")
def test_request_action_with_params(mock_post: object) -> None:
    """request_action includes params in the request body when provided."""
    mock_post.return_value = MOCK_SESSION_WITH_PARAMS  # type: ignore[attr-defined]

    params = {"target": "us-east-1", "version": "2.0"}
    result = asqav.request_action(
        agent_id="agent_001",
        action_type="deploy:production",
        params=params,
    )

    mock_post.assert_called_once_with(  # type: ignore[attr-defined]
        "/threshold/sessions",
        {
            "agent_id": "agent_001",
            "action_type": "deploy:production",
            "params": {"target": "us-east-1", "version": "2.0"},
        },
    )
    assert result.action_params == {"target": "us-east-1", "version": "2.0"}


@patch("asqav.client._post")
def test_request_action_without_params_omits_key(mock_post: object) -> None:
    """request_action does not send 'params' key when params is None."""
    mock_post.return_value = MOCK_SESSION_RESPONSE  # type: ignore[attr-defined]

    asqav.request_action(agent_id="agent_001", action_type="read:data")

    call_body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    assert "params" not in call_body


# ---------------------------------------------------------------------------
# approve_action tests
# ---------------------------------------------------------------------------


@patch("asqav.client._post")
def test_approve_action(mock_post: object) -> None:
    """approve_action calls POST /threshold/sessions/{id}/approve."""
    mock_post.return_value = MOCK_APPROVE_RESPONSE  # type: ignore[attr-defined]

    result = asqav.approve_action(
        session_id="thr_abc123",
        entity_id="ent_001",
    )

    mock_post.assert_called_once_with(  # type: ignore[attr-defined]
        "/threshold/sessions/thr_abc123/approve",
        {"entity_id": "ent_001"},
    )
    assert isinstance(result, ThresholdApproveResponse)
    assert result.session_id == "thr_abc123"
    assert result.entity_id == "ent_001"
    assert result.signatures_collected == 1
    assert result.threshold_required == 2
    assert result.status == "pending"
    assert result.threshold_met is False


@patch("asqav.client._post")
def test_approve_action_threshold_met(mock_post: object) -> None:
    """approve_action returns threshold_met=True when threshold is reached."""
    mock_post.return_value = MOCK_APPROVE_THRESHOLD_MET  # type: ignore[attr-defined]

    result = asqav.approve_action(
        session_id="thr_abc123",
        entity_id="ent_002",
    )

    assert result.threshold_met is True
    assert result.status == "approved"
    assert result.signatures_collected == 2


# ---------------------------------------------------------------------------
# get_action_status tests
# ---------------------------------------------------------------------------


@patch("asqav.client._get")
def test_get_action_status(mock_get: object) -> None:
    """get_action_status calls GET /threshold/sessions/{id}."""
    mock_get.return_value = MOCK_SESSION_RESPONSE  # type: ignore[attr-defined]

    result = asqav.get_action_status("thr_abc123")

    mock_get.assert_called_once_with(  # type: ignore[attr-defined]
        "/threshold/sessions/thr_abc123",
    )
    assert isinstance(result, ThresholdSessionResponse)
    assert result.session_id == "thr_abc123"
    assert result.status == "pending"


@patch("asqav.client._get")
def test_get_action_status_approved(mock_get: object) -> None:
    """get_action_status returns full details for approved session."""
    mock_get.return_value = MOCK_SESSION_APPROVED  # type: ignore[attr-defined]

    result = asqav.get_action_status("thr_abc123")

    assert result.status == "approved"
    assert result.signatures_collected == 2
    assert result.policy_attestation_hash == "abcdef1234567890"
    assert result.resolved_at == "2026-02-16T12:02:00"
    assert len(result.signatures) == 2

    sig0 = result.signatures[0]
    assert isinstance(sig0, ThresholdSignatureDetail)
    assert sig0.entity_id == "ent_001"
    assert sig0.entity_name == "Agent Key"
    assert sig0.entity_class == "A"

    sig1 = result.signatures[1]
    assert sig1.entity_id == "ent_002"
    assert sig1.entity_class == "B"


# ---------------------------------------------------------------------------
# Regression: sign() method unchanged
# ---------------------------------------------------------------------------


def test_sign_method_exists_on_agent() -> None:
    """sign() method is still present on Agent with correct signature."""
    import inspect

    sig = inspect.signature(asqav.Agent.sign)
    params = list(sig.parameters.keys())
    # self, action_type, context
    assert "self" in params
    assert "action_type" in params
    assert "context" in params
    assert len(params) == 3


def test_sign_method_returns_signature_response_annotation() -> None:
    """sign() is annotated to return SignatureResponse."""
    import inspect

    sig = inspect.signature(asqav.Agent.sign)
    # With 'from __future__ import annotations', return_annotation is a string
    assert "SignatureResponse" in str(sig.return_annotation)


# ---------------------------------------------------------------------------
# _parse_threshold_session helper
# ---------------------------------------------------------------------------


def test_parse_threshold_session_minimal() -> None:
    """_parse_threshold_session handles response with empty signatures."""
    result = _parse_threshold_session(MOCK_SESSION_RESPONSE)
    assert result.session_id == "thr_abc123"
    assert result.signatures == []
    assert result.resolved_at is None


def test_parse_threshold_session_with_signatures() -> None:
    """_parse_threshold_session correctly parses signature details."""
    result = _parse_threshold_session(MOCK_SESSION_APPROVED)
    assert len(result.signatures) == 2
    assert result.signatures[0].entity_name == "Agent Key"
    assert result.signatures[1].entity_class == "B"
