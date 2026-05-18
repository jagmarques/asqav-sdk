"""Tests for multi-party countersigning (sync + async)."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest

import asqav
from asqav.async_client import AsyncAgent


def _make_agent() -> "asqav.Agent":
    return asqav.Agent(
        agent_id="agent_alice",
        name="alice",
        public_key="pub",
        key_id="key_alice",
        algorithm="ML-DSA-65",
        capabilities=[],
        created_at=0.0,
    )


def _make_async_agent() -> AsyncAgent:
    return AsyncAgent(
        agent_id="agent_bob",
        name="bob",
        public_key="pub",
        key_id="key_bob",
        algorithm="ML-DSA-65",
        capabilities=[],
        created_at=0.0,
    )


def test_sign_accepts_co_signers_kwarg() -> None:
    sig = inspect.signature(asqav.Agent.sign)
    assert "co_signers" in sig.parameters
    assert sig.parameters["co_signers"].default is None
    # keyword-only
    assert sig.parameters["co_signers"].kind == inspect.Parameter.KEYWORD_ONLY


def test_async_sign_accepts_co_signers_kwarg() -> None:
    sig = inspect.signature(AsyncAgent.sign)
    assert "co_signers" in sig.parameters
    assert sig.parameters["co_signers"].kind == inspect.Parameter.KEYWORD_ONLY


def test_countersign_method_exists() -> None:
    assert hasattr(asqav.Agent, "countersign")
    sig = inspect.signature(asqav.Agent.countersign)
    assert "signature_id" in sig.parameters


def test_async_countersign_method_exists() -> None:
    assert hasattr(AsyncAgent, "countersign")
    sig = inspect.signature(AsyncAgent.countersign)
    assert "signature_id" in sig.parameters


@patch("asqav.client._post")
def test_sign_forwards_co_signers_in_body(mock_post: object) -> None:
    mock_post.return_value = {  # type: ignore[attr-defined]
        "signature": "sig_bytes",
        "signature_id": "sig_01",
        "action_id": "act_01",
        "timestamp": "2026-04-28T10:00:00Z",
        "verification_url": "https://api.asqav.com/api/v1/verify/sig_01",
        "required_co_signers": ["agent_bob", "agent_carol"],
        "co_signatures": [],
        "countersign_url": "/api/v1/agents/{agent_id}/countersign/sig_01",
    }
    agent = _make_agent()
    resp = agent.sign(
        action_type="verify:peer-output",
        context={"task": "review"},
        co_signers=["agent_bob", "agent_carol"],
    )

    sent_body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    assert sent_body["co_signers"] == ["agent_bob", "agent_carol"]
    assert resp.required_co_signers == ["agent_bob", "agent_carol"]
    assert resp.co_signatures == []
    assert resp.countersign_url is not None


@patch("asqav.client._post")
def test_sign_without_co_signers_omits_field(mock_post: object) -> None:
    mock_post.return_value = {  # type: ignore[attr-defined]
        "signature": "sig_bytes",
        "signature_id": "sig_02",
        "action_id": "act_02",
        "timestamp": "2026-04-28T10:00:00Z",
        "verification_url": "https://api.asqav.com/api/v1/verify/sig_02",
    }
    agent = _make_agent()
    resp = agent.sign(action_type="api:call", context={})

    sent_body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    assert "co_signers" not in sent_body
    assert resp.required_co_signers is None
    assert resp.co_signatures is None


@patch("asqav.client._post")
def test_countersign_posts_to_correct_path(mock_post: object) -> None:
    mock_post.return_value = {  # type: ignore[attr-defined]
        "signature": "sig_bytes",
        "signature_id": "sig_01",
        "action_id": "act_01",
        "timestamp": "2026-04-28T10:00:00Z",
        "verification_url": "https://api.asqav.com/api/v1/verify/sig_01",
        "required_co_signers": ["agent_alice"],
        "co_signatures": [
            {"agent_id": "agent_alice", "signature": "co_sig_bytes", "signed_at": "2026-04-28T10:00:01Z"}
        ],
    }
    agent = _make_agent()
    resp = agent.countersign("sig_01")

    path = mock_post.call_args[0][0]  # type: ignore[attr-defined]
    body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    assert path == "/agents/agent_alice/countersign/sig_01"
    assert body == {}
    assert resp.co_signatures is not None
    assert len(resp.co_signatures) == 1
    assert resp.co_signatures[0]["agent_id"] == "agent_alice"


@pytest.mark.asyncio
@patch("asqav.async_client._async_post", new_callable=AsyncMock)
async def test_async_sign_forwards_co_signers(mock_post: AsyncMock) -> None:
    mock_post.return_value = {
        "signature": "sig_bytes",
        "signature_id": "sig_async",
        "action_id": "act_async",
        "timestamp": "2026-04-28T10:00:00Z",
        "verification_url": "https://api.asqav.com/api/v1/verify/sig_async",
        "required_co_signers": ["agent_alice"],
        "co_signatures": [],
    }
    agent = _make_async_agent()
    resp = await agent.sign(
        action_type="verify:peer-output",
        context={},
        co_signers=["agent_alice"],
    )
    sent_body = mock_post.call_args[0][1]
    assert sent_body["co_signers"] == ["agent_alice"]
    assert resp.required_co_signers == ["agent_alice"]


@pytest.mark.asyncio
@patch("asqav.async_client._async_post", new_callable=AsyncMock)
async def test_async_countersign_posts_to_correct_path(mock_post: AsyncMock) -> None:
    mock_post.return_value = {
        "signature": "sig_bytes",
        "signature_id": "sig_async",
        "action_id": "act_async",
        "timestamp": "2026-04-28T10:00:00Z",
        "verification_url": "https://api.asqav.com/api/v1/verify/sig_async",
        "required_co_signers": ["agent_bob"],
        "co_signatures": [
            {"agent_id": "agent_bob", "signature": "co_sig", "signed_at": "2026-04-28T10:00:01Z"}
        ],
    }
    agent = _make_async_agent()
    resp = await agent.countersign("sig_async")
    path = mock_post.call_args[0][0]
    assert path == "/agents/agent_bob/countersign/sig_async"
    assert resp.co_signatures is not None
    assert resp.co_signatures[0]["agent_id"] == "agent_bob"
