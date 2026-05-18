"""Tests for the Asqav hooks/middleware system."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asqav
from asqav import hooks as hooks_module


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


def _ok_response() -> dict:
    return {
        "signature": "sig_bytes",
        "signature_id": "sig_01",
        "action_id": "act_01",
        "timestamp": "2026-04-22T10:00:00Z",
        "verification_url": "https://api.asqav.com/api/v1/verify/sig_01",
    }


@pytest.fixture(autouse=True)
def _clean_hooks():
    """Clear hook registry before and after each test."""
    asqav.clear_hooks()
    yield
    asqav.clear_hooks()


@patch("asqav.client._post")
def test_register_before_mutates_context_reaching_sign(mock_post: object) -> None:
    """A before-hook can add fields that land in the body sent to the API."""
    mock_post.return_value = _ok_response()  # type: ignore[attr-defined]

    def add_field(action_type: str, context: dict) -> dict:
        context["injected"] = "by_hook"
        return context

    asqav.register_before("*", add_field)

    agent = _make_agent()
    agent.sign("api:call", {"orig": "yes"})

    sent_body = mock_post.call_args[0][1]  # type: ignore[attr-defined]
    assert sent_body["context"]["injected"] == "by_hook"
    assert sent_body["context"]["orig"] == "yes"


@patch("asqav.client._post")
def test_register_after_fires_with_signature_response(mock_post: object) -> None:
    """After-hooks receive the SignatureResponse object."""
    mock_post.return_value = _ok_response()  # type: ignore[attr-defined]

    seen: list[object] = []
    asqav.register_after("*", lambda resp: seen.append(resp))

    agent = _make_agent()
    response = agent.sign("api:call")

    assert len(seen) == 1
    assert seen[0] is response
    assert seen[0].signature_id == "sig_01"  # type: ignore[attr-defined]


@patch("asqav.client._post")
def test_failing_hook_does_not_crash_sign(mock_post: object) -> None:
    """A raising hook is logged and signing still returns a response."""
    mock_post.return_value = _ok_response()  # type: ignore[attr-defined]

    def boom_before(action_type: str, context: dict) -> dict:
        raise RuntimeError("before exploded")

    def boom_after(resp: object) -> None:
        raise RuntimeError("after exploded")

    asqav.register_before("*", boom_before)
    asqav.register_after("*", boom_after)

    agent = _make_agent()
    response = agent.sign("api:call")

    assert response.signature_id == "sig_01"


@patch("asqav.client._post")
def test_pattern_matching_strands_glob(mock_post: object) -> None:
    """``strands:*`` matches strands:model_call but not langchain:tool_call."""
    mock_post.return_value = _ok_response()  # type: ignore[attr-defined]

    matched: list[str] = []

    def record(action_type: str, context: dict) -> dict:
        matched.append(action_type)
        return context

    asqav.register_before("strands:*", record)

    agent = _make_agent()
    agent.sign("strands:model_call")
    agent.sign("langchain:tool_call")

    assert matched == ["strands:model_call"]


def test_clear_hooks_empties_registry() -> None:
    """clear_hooks removes both before and after registrations."""
    asqav.register_before("*", lambda a, c: c)
    asqav.register_after("*", lambda r: None)

    assert len(hooks_module._before_hooks) == 1
    assert len(hooks_module._after_hooks) == 1

    asqav.clear_hooks()

    assert hooks_module._before_hooks == []
    assert hooks_module._after_hooks == []
