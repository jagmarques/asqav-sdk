"""Tests for portable scope tokens."""

from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import Agent, SDTokenResponse
from asqav.scope import ScopeToken, create_scope_token, is_replay, verify_scope_token

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_SD_RESPONSE = {
    "token": "eyJ0eXAiOiJzZC1qd3QifQ.payload~disc_a~disc_b~disc_c~",
    "jwt": "eyJ0eXAiOiJzZC1qd3QifQ.payload",
    "disclosures": {
        "scope_actions": "disc_a",
        "agent_id": "disc_b",
        "agent_name": "disc_c",
    },
    "expires_at": "2026-12-31T23:59:59Z",
}

MOCK_SD_WITH_META = {
    **MOCK_SD_RESPONSE,
    "disclosures": {
        **MOCK_SD_RESPONSE["disclosures"],
        "metadata": "disc_d",
    },
}


def _make_agent() -> Agent:
    return Agent(
        agent_id="agent_test_01",
        name="test-agent",
        public_key="pk_test",
        key_id="kid_test",
        algorithm="ml-dsa-65",
        capabilities=["data:read:*", "api:call:*"],
        created_at=time.time(),
    )


def _sd_token_response(data: dict | None = None) -> SDTokenResponse:
    d = data or MOCK_SD_RESPONSE
    return SDTokenResponse(
        token=d["token"],
        jwt=d["jwt"],
        disclosures=d["disclosures"],
        expires_at=1893455999.0,
    )


# ---------------------------------------------------------------------------
# ScopeToken dataclass tests
# ---------------------------------------------------------------------------


class TestScopeToken:
    def test_to_header_full(self):
        st = ScopeToken(
            token="full~token~value~",
            jwt="jwt_part",
            disclosures={"scope_actions": "d1", "agent_id": "d2"},
            expires_at=time.time() + 3600,
            actions=["data:read:*"],
            agent_id="agent_01",
        )
        h = st.to_header()
        assert h == {"Authorization": "Bearer full~token~value~"}

    def test_to_header_selective(self):
        st = ScopeToken(
            token="full~token~",
            jwt="jwt_part",
            disclosures={"scope_actions": "d1", "agent_id": "d2"},
            expires_at=time.time() + 3600,
            actions=["data:read:*"],
            agent_id="agent_01",
        )
        h = st.to_header(disclose=["scope_actions"])
        assert h == {"Authorization": "Bearer jwt_part~d1~"}

    def test_to_header_unknown_claim_skipped(self):
        st = ScopeToken(
            token="full~token~",
            jwt="jwt_part",
            disclosures={"scope_actions": "d1"},
            expires_at=time.time() + 3600,
            actions=["data:read:*"],
            agent_id="agent_01",
        )
        h = st.to_header(disclose=["scope_actions", "nonexistent"])
        assert h == {"Authorization": "Bearer jwt_part~d1~"}

    def test_present(self):
        st = ScopeToken(
            token="full~token~",
            jwt="jwt_part",
            disclosures={"scope_actions": "d1", "agent_id": "d2"},
            expires_at=time.time() + 3600,
            actions=["data:read:*"],
            agent_id="agent_01",
        )
        result = st.present(["agent_id"])
        assert result == "jwt_part~d2~"

    def test_is_expired_false(self):
        st = ScopeToken(
            token="t",
            jwt="j",
            disclosures={},
            expires_at=time.time() + 3600,
            actions=[],
            agent_id="a",
        )
        assert st.is_expired() is False

    def test_is_expired_true(self):
        st = ScopeToken(
            token="t",
            jwt="j",
            disclosures={},
            expires_at=time.time() - 10,
            actions=[],
            agent_id="a",
        )
        assert st.is_expired() is True

    def test_to_dict(self):
        st = ScopeToken(
            token="tok",
            jwt="j",
            disclosures={"a": "b"},
            expires_at=123.0,
            actions=["x:y"],
            agent_id="ag",
            metadata={"env": "prod"},
        )
        d = st.to_dict()
        assert d["token"] == "tok"
        assert d["actions"] == ["x:y"]
        assert d["metadata"] == {"env": "prod"}

    def test_from_dict_roundtrip(self):
        original = ScopeToken(
            token="tok",
            jwt="j",
            disclosures={"a": "b"},
            expires_at=999.0,
            actions=["read:*"],
            agent_id="ag1",
            metadata={"k": "v"},
        )
        restored = ScopeToken.from_dict(original.to_dict())
        assert restored.token == original.token
        assert restored.actions == original.actions
        assert restored.metadata == original.metadata
        assert restored.expires_at == original.expires_at

    def test_from_dict_missing_metadata(self):
        data = {
            "token": "t",
            "jwt": "j",
            "disclosures": {},
            "expires_at": 1.0,
            "actions": [],
            "agent_id": "a",
        }
        st = ScopeToken.from_dict(data)
        assert st.metadata == {}


# ---------------------------------------------------------------------------
# create_scope_token tests
# ---------------------------------------------------------------------------


class TestCreateScopeToken:
    @patch("asqav.scope.resolve_pattern", side_effect=lambda p: p)
    def test_basic_creation(self, _mock_resolve):
        agent = _make_agent()
        sd_resp = _sd_token_response()
        with patch.object(agent, "issue_sd_token", return_value=sd_resp) as mock_sd:
            token = create_scope_token(agent, actions=["data:read:*"])

        assert isinstance(token, ScopeToken)
        assert token.agent_id == "agent_test_01"
        assert token.actions == ["data:read:*"]
        mock_sd.assert_called_once()
        call_kwargs = mock_sd.call_args
        assert "scope_actions" in call_kwargs[1]["claims"] or "scope_actions" in call_kwargs.kwargs.get("claims", {})

    def test_semantic_pattern_resolved(self):
        agent = _make_agent()
        sd_resp = _sd_token_response()
        with patch.object(agent, "issue_sd_token", return_value=sd_resp):
            token = create_scope_token(agent, actions=["sql-read", "http-external"])

        assert token.actions == ["data:read:sql:*", "api:call:external:*"]

    def test_custom_ttl(self):
        agent = _make_agent()
        sd_resp = _sd_token_response()
        with patch.object(agent, "issue_sd_token", return_value=sd_resp) as mock_sd:
            create_scope_token(agent, actions=["data:read:*"], ttl=7200)

        assert mock_sd.call_args.kwargs["ttl"] == 7200

    def test_metadata_included(self):
        agent = _make_agent()
        sd_resp = _sd_token_response(MOCK_SD_WITH_META)
        with patch.object(agent, "issue_sd_token", return_value=sd_resp) as mock_sd:
            token = create_scope_token(
                agent, actions=["data:read:*"], metadata={"org": "acme"}
            )

        claims = mock_sd.call_args.kwargs["claims"]
        assert claims["metadata"] == {"org": "acme"}
        assert "metadata" in mock_sd.call_args.kwargs["disclosable"]
        assert token.metadata == {"org": "acme"}

    def test_no_metadata_not_in_disclosable(self):
        agent = _make_agent()
        sd_resp = _sd_token_response()
        with patch.object(agent, "issue_sd_token", return_value=sd_resp) as mock_sd:
            create_scope_token(agent, actions=["data:read:*"])

        assert "metadata" not in mock_sd.call_args.kwargs["disclosable"]


# ---------------------------------------------------------------------------
# Agent.create_scope_token convenience method
# ---------------------------------------------------------------------------


class TestAgentScopeTokenMethod:
    def test_agent_method_delegates(self):
        agent = _make_agent()
        sd_resp = _sd_token_response()
        with patch.object(agent, "issue_sd_token", return_value=sd_resp):
            token = agent.create_scope_token(actions=["data:read:*"])

        assert isinstance(token, ScopeToken)
        assert token.agent_id == "agent_test_01"

    def test_agent_method_passes_ttl(self):
        agent = _make_agent()
        sd_resp = _sd_token_response()
        with patch.object(agent, "issue_sd_token", return_value=sd_resp) as mock_sd:
            agent.create_scope_token(actions=["data:read:*"], ttl=1800)

        assert mock_sd.call_args.kwargs["ttl"] == 1800

    def test_agent_method_passes_metadata(self):
        agent = _make_agent()
        sd_resp = _sd_token_response(MOCK_SD_WITH_META)
        with patch.object(agent, "issue_sd_token", return_value=sd_resp):
            token = agent.create_scope_token(
                actions=["data:read:*"], metadata={"env": "staging"}
            )

        assert token.metadata == {"env": "staging"}


# ---------------------------------------------------------------------------
# verify_scope_token tests
# ---------------------------------------------------------------------------


class TestVerifyScopeToken:
    @patch("asqav.client.verify_signature")
    def test_verify_delegates(self, mock_verify):
        mock_verify.return_value = MagicMock(
            verified=True,
            agent_id="agent_01",
            agent_name="my-agent",
            action_type="data:read:*",
            algorithm="ml-dsa-65",
            signed_at=1700000000.0,
        )
        result = verify_scope_token("sig_abc123")

        mock_verify.assert_called_once_with("sig_abc123")
        assert result["verified"] is True
        assert result["agent_id"] == "agent_01"
        assert result["algorithm"] == "ml-dsa-65"

    @patch("asqav.client.verify_signature")
    def test_verify_returns_all_fields(self, mock_verify):
        mock_verify.return_value = MagicMock(
            verified=False,
            agent_id="agent_02",
            agent_name="bad-agent",
            action_type="admin:*",
            algorithm="ml-dsa-87",
            signed_at=1700000001.0,
        )
        result = verify_scope_token("sig_bad")
        assert result["verified"] is False
        assert result["agent_name"] == "bad-agent"
        assert result["signed_at"] == 1700000001.0


# ---------------------------------------------------------------------------
# Module-level imports
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_scope_token_importable_from_asqav(self):
        import asqav

        assert hasattr(asqav, "ScopeToken")
        assert hasattr(asqav, "create_scope_token")
        assert hasattr(asqav, "verify_scope_token")

    def test_scope_token_in_all(self):
        import asqav

        assert "ScopeToken" in asqav.__all__
        assert "create_scope_token" in asqav.__all__
        assert "verify_scope_token" in asqav.__all__
        assert "is_replay" in asqav.__all__


# ---------------------------------------------------------------------------
# Nonce replay protection tests
# ---------------------------------------------------------------------------


class TestNonceReplayProtection:
    @patch("asqav.scope.resolve_pattern", side_effect=lambda p: p)
    def test_nonce_is_present(self, _mock_resolve):
        agent = _make_agent()
        sd_resp = _sd_token_response()
        with patch.object(agent, "issue_sd_token", return_value=sd_resp):
            token = create_scope_token(agent, actions=["data:read:*"])
        assert isinstance(token.nonce, str)
        assert len(token.nonce) > 0

    @patch("asqav.scope.resolve_pattern", side_effect=lambda p: p)
    def test_nonce_is_unique(self, _mock_resolve):
        agent = _make_agent()
        sd_resp = _sd_token_response()
        with patch.object(agent, "issue_sd_token", return_value=sd_resp):
            token1 = create_scope_token(agent, actions=["data:read:*"])
            token2 = create_scope_token(agent, actions=["data:read:*"])
        assert token1.nonce != token2.nonce

    def test_is_replay_detects_reuse(self):
        seen = {"abc123", "def456"}
        assert is_replay("abc123", seen) is True

    def test_is_replay_allows_new(self):
        seen = {"abc123", "def456"}
        assert is_replay("xyz789", seen) is False
