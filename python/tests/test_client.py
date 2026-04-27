"""Tests for asqav SDK client -- multi-party signing methods."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav.client import (
    ApprovalResponse,
    SignatureDetail,
    SigningSessionResponse,
    _parse_signing_session,
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
    "approvals_required": 2,
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
    "approvals_required": 2,
    "status": "pending",
    "approved": False,
}

MOCK_APPROVE_APPROVED: dict = {
    "session_id": "thr_abc123",
    "entity_id": "ent_002",
    "signatures_collected": 2,
    "approvals_required": 2,
    "status": "approved",
    "approved": True,
}


# ---------------------------------------------------------------------------
# request_action tests
# ---------------------------------------------------------------------------


@patch("asqav.client._post")
def test_request_action(mock_post: object) -> None:
    """request_action calls POST /signing-groups/sessions with correct body."""
    mock_post.return_value = MOCK_SESSION_RESPONSE  # type: ignore[attr-defined]

    result = asqav.request_action(
        agent_id="agent_001",
        action_type="deploy:production",
    )

    mock_post.assert_called_once_with(  # type: ignore[attr-defined]
        "/signing-groups/sessions",
        {"agent_id": "agent_001", "action_type": "deploy:production"},
    )
    assert isinstance(result, SigningSessionResponse)
    assert result.session_id == "thr_abc123"
    assert result.agent_id == "agent_001"
    assert result.action_type == "deploy:production"
    assert result.approvals_required == 2
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
        "/signing-groups/sessions",
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
    """approve_action calls POST /signing-groups/sessions/{id}/approve."""
    mock_post.return_value = MOCK_APPROVE_RESPONSE  # type: ignore[attr-defined]

    result = asqav.approve_action(
        session_id="thr_abc123",
        entity_id="ent_001",
    )

    mock_post.assert_called_once_with(  # type: ignore[attr-defined]
        "/signing-groups/sessions/thr_abc123/approve",
        {"entity_id": "ent_001"},
    )
    assert isinstance(result, ApprovalResponse)
    assert result.session_id == "thr_abc123"
    assert result.entity_id == "ent_001"
    assert result.signatures_collected == 1
    assert result.approvals_required == 2
    assert result.status == "pending"
    assert result.approved is False


@patch("asqav.client._post")
def test_approve_action_approved(mock_post: object) -> None:
    """approve_action returns approved=True when approvals are met."""
    mock_post.return_value = MOCK_APPROVE_APPROVED  # type: ignore[attr-defined]

    result = asqav.approve_action(
        session_id="thr_abc123",
        entity_id="ent_002",
    )

    assert result.approved is True
    assert result.status == "approved"
    assert result.signatures_collected == 2


# ---------------------------------------------------------------------------
# get_action_status tests
# ---------------------------------------------------------------------------


@patch("asqav.client._get")
def test_get_action_status(mock_get: object) -> None:
    """get_action_status calls GET /signing-groups/sessions/{id}."""
    mock_get.return_value = MOCK_SESSION_RESPONSE  # type: ignore[attr-defined]

    result = asqav.get_action_status("thr_abc123")

    mock_get.assert_called_once_with(  # type: ignore[attr-defined]
        "/signing-groups/sessions/thr_abc123",
    )
    assert isinstance(result, SigningSessionResponse)
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
    assert isinstance(sig0, SignatureDetail)
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
    """sign() method is still present on Agent with the core required params."""
    import inspect

    sig = inspect.signature(asqav.Agent.sign)
    params = list(sig.parameters.keys())
    assert "self" in params
    assert "action_type" in params
    assert "context" in params


def test_sign_method_returns_signature_response_annotation() -> None:
    """sign() is annotated to return SignatureResponse."""
    import inspect

    sig = inspect.signature(asqav.Agent.sign)
    # With 'from __future__ import annotations', return_annotation is a string
    assert "SignatureResponse" in str(sig.return_annotation)


def test_sign_accepts_counterparty_kwarg() -> None:
    """sign() exposes a counterparty kwarg for endpoint-identity binding."""
    import inspect

    sig = inspect.signature(asqav.Agent.sign)
    assert "counterparty" in sig.parameters
    assert sig.parameters["counterparty"].default is None


@patch("asqav.client._post")
def test_sign_passes_counterparty_into_context(mock_post: object) -> None:
    """counterparty kwarg lands in context under _counterparty."""
    mock_post.return_value = {  # type: ignore[attr-defined]
        "signature": "sig_bytes",
        "signature_id": "sig_01",
        "action_id": "act_01",
        "timestamp": "2026-04-22T10:00:00Z",
        "verification_url": "https://api.asqav.com/api/v1/verify/sig_01",
    }
    agent = _make_agent()
    cp = {"kind": "tls_spki_sha256", "value": "abc123"}
    agent.sign(action_type="api:call", counterparty=cp)

    call_kwargs = mock_post.call_args  # type: ignore[attr-defined]
    sent_body = call_kwargs[0][1]
    assert sent_body["context"]["_counterparty"] == cp


# ---------------------------------------------------------------------------
# Agent.sign_batch
# ---------------------------------------------------------------------------


def _make_agent() -> "asqav.Agent":
    """Construct an Agent without hitting the API."""
    return asqav.Agent(
        agent_id="agent_test",
        name="test",
        public_key="pub",
        key_id="key_test",
        algorithm="ML-DSA-65",
        capabilities=[],
        created_at=0.0,
    )


@patch("asqav.client._post")
def test_sign_batch_returns_n_signatures(mock_post: object) -> None:
    """sign_batch returns one SignatureResponse per input action."""
    mock_post.return_value = {  # type: ignore[attr-defined]
        "signatures": [
            {
                "signature": f"sig_{i}",
                "signature_id": f"sid_{i}",
                "action_id": f"act_{i}",
                "timestamp": 1700000000.0 + i,
                "verification_url": f"https://api.asqav.com/verify/sid_{i}",
            }
            for i in range(10)
        ],
        "errors": [None] * 10,
    }
    agent = _make_agent()
    actions = [{"action_type": f"test:{i}", "context": {"i": i}} for i in range(10)]
    out = agent.sign_batch(actions)
    assert len(out) == 10
    assert all(s is not None for s in out)
    assert out[0].signature_id == "sid_0"
    assert out[9].signature_id == "sid_9"


@patch("asqav.client._post")
def test_sign_batch_handles_partial_failure(mock_post: object) -> None:
    """sign_batch returns None for items the server failed to sign."""
    mock_post.return_value = {  # type: ignore[attr-defined]
        "signatures": [
            {
                "signature": "sig_0",
                "signature_id": "sid_0",
                "action_id": "act_0",
                "timestamp": 1700000000.0,
                "verification_url": "https://api.asqav.com/verify/sid_0",
            },
            None,
        ],
        "errors": [None, {"status_code": 403, "detail": "policy_block"}],
    }
    agent = _make_agent()
    out = agent.sign_batch([
        {"action_type": "ok"},
        {"action_type": "blocked"},
    ])
    assert out[0] is not None
    assert out[1] is None


def test_sign_batch_rejects_empty_input() -> None:
    """sign_batch raises ValueError on empty list."""
    import pytest

    agent = _make_agent()
    with pytest.raises(ValueError, match="at least one"):
        agent.sign_batch([])


def test_sign_batch_rejects_oversize_input() -> None:
    """sign_batch raises ValueError when over 100 actions."""
    import pytest

    agent = _make_agent()
    with pytest.raises(ValueError, match="at most 100"):
        agent.sign_batch([{"action_type": "x"}] * 101)


def test_sign_batch_rejects_missing_action_type() -> None:
    """sign_batch raises ValueError when an action lacks 'action_type'."""
    import pytest

    agent = _make_agent()
    with pytest.raises(ValueError, match="action_type"):
        agent.sign_batch([{"context": {"x": 1}}])


# ---------------------------------------------------------------------------
# _parse_signing_session helper
# ---------------------------------------------------------------------------


def test_parse_signing_session_minimal() -> None:
    """_parse_signing_session handles response with empty signatures."""
    result = _parse_signing_session(MOCK_SESSION_RESPONSE)
    assert result.session_id == "thr_abc123"
    assert result.signatures == []
    assert result.resolved_at is None


def test_parse_signing_session_with_signatures() -> None:
    """_parse_signing_session correctly parses signature details."""
    result = _parse_signing_session(MOCK_SESSION_APPROVED)
    assert len(result.signatures) == 2
    assert result.signatures[0].entity_name == "Agent Key"
    assert result.signatures[1].entity_class == "B"
