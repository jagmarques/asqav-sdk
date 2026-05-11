"""Verify user_intent kwarg passes through to the wire body.

The SDK does not interpret user_intent; it only serializes the dict as-is.
Backend verification is covered by the Asqav backend test suite.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav import client as client_mod  # noqa: E402
from asqav.client import Agent  # noqa: E402

MOCK_SIGN_RESPONSE: dict = {
    "signature": "sig-bytes",
    "signature_id": "sig_001",
    "action_id": "act_001",
    "timestamp": "2026-04-28T12:00:00Z",
    "verification_url": "https://verify.asqav.com/sig_001",
    "algorithm": "ml-dsa-65",
    "chain_hash": "ch_abc",
    "rfc3161_timestamp": None,
    "scan_result": None,
    "policy_digest": "pd",
    "policy_decision": "permit",
    "authorization_ref": None,
    "bitcoin_anchor": None,
    "user_intent_verified": True,
}


def _make_agent() -> Agent:
    return Agent(
        agent_id="agt_001",
        name="t",
        public_key="pk",
        key_id="kid",
        algorithm="ml-dsa-65",
        capabilities=[],
        created_at=0.0,
    )


@pytest.fixture(autouse=True)
def _isolate_module_state() -> None:
    saved = (client_mod._mode, client_mod._org_salt, client_mod._api_key)
    client_mod._api_key = "sk_test"
    yield
    client_mod._mode, client_mod._org_salt, client_mod._api_key = saved


def _intent() -> dict:
    return {
        "signature": "AAAA",
        "public_key": "BBBB",
        "algorithm": "ed25519",
        "signed_message": "CCCC",
        "signed_at": "2026-04-28T12:00:00Z",
    }


def test_user_intent_passthrough_full_payload() -> None:
    client_mod._mode = "full-payload"
    client_mod._org_salt = None
    agent = _make_agent()
    intent = _intent()

    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        resp = agent.sign("api:call", {"x": 1}, user_intent=intent)

    _path, body = mock_post.call_args[0]
    assert body["user_intent"] == intent
    assert resp.user_intent_verified is True


def test_user_intent_passthrough_hash_only() -> None:
    client_mod._mode = "hash-only"
    client_mod._org_salt = None
    agent = _make_agent()
    intent = _intent()

    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign("api:call", {"x": 1}, user_intent=intent)

    _path, body = mock_post.call_args[0]
    assert body["user_intent"] == intent
    assert "context" not in body  # hash-only invariant preserved


def test_user_intent_omitted_when_not_set() -> None:
    client_mod._mode = "full-payload"
    client_mod._org_salt = None
    agent = _make_agent()

    with patch("asqav.client._post", return_value=MOCK_SIGN_RESPONSE) as mock_post:
        agent.sign("api:call", {"x": 1})

    _path, body = mock_post.call_args[0]
    assert "user_intent" not in body
