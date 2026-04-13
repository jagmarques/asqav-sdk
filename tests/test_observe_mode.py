"""Tests for observe mode in AsqavAdapter."""

from __future__ import annotations

import logging
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from asqav.client import AsqavError, SignatureResponse
from asqav.extras._base import AsqavAdapter


class _ObserveAdapter(AsqavAdapter):
    """Concrete subclass for testing observe mode."""

    pass


MOCK_SIGN_RESPONSE: dict = {
    "signature": "sig_abc123",
    "signature_id": "sid_abc123",
    "action_id": "act_abc123",
    "timestamp": 1700000001.0,
    "verification_url": "https://api.asqav.com/verify/sid_abc123",
}


@patch("asqav.extras._base.Agent")
def test_observe_mode_returns_none(mock_agent_cls):
    """In observe mode, _sign_action returns None without calling the server."""
    mock_agent = MagicMock()
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _ObserveAdapter(agent_name="test", observe=True)

    result = adapter._sign_action("llm:call", {"model": "gpt-4"})
    assert result is None
    mock_agent.sign.assert_not_called()


@patch("asqav.extras._base.Agent")
def test_observe_mode_does_not_accumulate_signatures(mock_agent_cls):
    """In observe mode, no signatures are accumulated."""
    mock_agent = MagicMock()
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _ObserveAdapter(agent_name="test", observe=True)

    adapter._sign_action("action:one")
    adapter._sign_action("action:two")
    assert len(adapter._signatures) == 0
    mock_agent.sign.assert_not_called()


@patch("asqav.extras._base.Agent")
def test_observe_mode_logs_action(mock_agent_cls, caplog):
    """In observe mode, _sign_action logs the action that would be signed."""
    mock_agent = MagicMock()
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _ObserveAdapter(agent_name="test", observe=True)

    with caplog.at_level(logging.INFO, logger="asqav"):
        adapter._sign_action("llm:call", {"model": "gpt-4"})

    assert len(caplog.records) == 1
    assert "OBSERVE" in caplog.records[0].message
    assert "llm:call" in caplog.records[0].message
    assert "gpt-4" in caplog.records[0].message


@patch("asqav.extras._base.Agent")
def test_observe_mode_logs_none_context(mock_agent_cls, caplog):
    """Observe mode works when context is None."""
    mock_agent = MagicMock()
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _ObserveAdapter(agent_name="test", observe=True)

    with caplog.at_level(logging.INFO, logger="asqav"):
        adapter._sign_action("tool:execute")

    assert "OBSERVE" in caplog.records[0].message
    assert "tool:execute" in caplog.records[0].message


@patch("asqav.extras._base.Agent")
def test_observe_false_calls_server(mock_agent_cls):
    """With observe=False (default), _sign_action calls the server normally."""
    mock_agent = MagicMock()
    mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
    mock_agent.sign.return_value = mock_sig
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _ObserveAdapter(agent_name="test", observe=False)

    result = adapter._sign_action("llm:call", {"model": "gpt-4"})
    assert result is mock_sig
    mock_agent.sign.assert_called_once_with("llm:call", {"model": "gpt-4"})


@patch("asqav.extras._base.Agent")
def test_default_observe_is_false(mock_agent_cls):
    """By default, observe mode is disabled."""
    mock_agent = MagicMock()
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _ObserveAdapter(agent_name="test")

    assert adapter._observe is False
