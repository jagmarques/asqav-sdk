"""Tests for AsqavAdapter base class."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from asqav.client import AsqavError, SignatureResponse
from asqav.extras._base import AsqavAdapter, _class_name_to_agent_name

# === Fixtures ===

MOCK_SIGN_RESPONSE: dict = {
    "signature": "sig_abc123",
    "signature_id": "sid_abc123",
    "action_id": "act_abc123",
    "timestamp": 1700000001.0,
    "verification_url": "https://api.asqav.com/verify/sid_abc123",
}


# === Helper: concrete subclass for testing ===


class _TestAdapter(AsqavAdapter):
    """Concrete subclass to test the base."""

    pass


class AsqavCallbackHandler(AsqavAdapter):
    """Named to test auto-name generation."""

    pass


# === Name generation ===


def test_class_name_to_agent_name():
    assert _class_name_to_agent_name("AsqavCallbackHandler") == "asqav-callback-handler"


def test_class_name_to_agent_name_crew():
    assert _class_name_to_agent_name("AsqavCrewHook") == "asqav-crew-hook"


def test_class_name_to_agent_name_simple():
    assert _class_name_to_agent_name("Adapter") == "adapter"


# === Initialization ===


def test_init_raises_without_asqav_init():
    """AsqavAdapter raises AsqavError when asqav.init() was not called."""
    with patch("asqav.client._api_key", None):
        with pytest.raises(AsqavError, match="Call asqav.init"):
            _TestAdapter()


@patch("asqav.extras._base.Agent")
def test_init_with_agent_name(mock_agent_cls):
    """Passing agent_name calls Agent.create with that name."""
    mock_agent_cls.create.return_value = MagicMock()
    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="my-agent")
    mock_agent_cls.create.assert_called_once_with("my-agent")
    assert adapter._agent is mock_agent_cls.create.return_value


@patch("asqav.extras._base.Agent")
def test_init_with_agent_id(mock_agent_cls):
    """Passing agent_id calls Agent.get with that ID."""
    mock_agent_cls.get.return_value = MagicMock()
    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_id="agent_abc")
    mock_agent_cls.get.assert_called_once_with("agent_abc")
    assert adapter._agent is mock_agent_cls.get.return_value


@patch("asqav.extras._base.Agent")
def test_init_auto_name_from_subclass(mock_agent_cls):
    """Without agent_name or agent_id, auto-generates name from class."""
    mock_agent_cls.create.return_value = MagicMock()
    with patch("asqav.client._api_key", "sk_test"):
        AsqavCallbackHandler()
    mock_agent_cls.create.assert_called_once_with("asqav-callback-handler")


# === _sign_action ===


@patch("asqav.extras._base.Agent")
def test_sign_action_success(mock_agent_cls):
    """_sign_action returns SignatureResponse on success."""
    mock_agent = MagicMock()
    mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
    mock_agent.sign.return_value = mock_sig
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="test")

    result = adapter._sign_action("llm:call", {"model": "gpt-4"})
    assert result is mock_sig
    mock_agent.sign.assert_called_once_with("llm:call", {"model": "gpt-4"})


@patch("asqav.extras._base.Agent")
def test_sign_action_accumulates_signatures(mock_agent_cls):
    """Successful signatures accumulate in _signatures list."""
    mock_agent = MagicMock()
    mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
    mock_agent.sign.return_value = mock_sig
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="test")

    adapter._sign_action("action:one")
    adapter._sign_action("action:two")
    assert len(adapter._signatures) == 2
    assert all(s is mock_sig for s in adapter._signatures)


@patch("asqav.extras._base.Agent")
def test_sign_action_fail_open(mock_agent_cls):
    """_sign_action returns None on AsqavError (fail-open)."""
    mock_agent = MagicMock()
    mock_agent.sign.side_effect = AsqavError("network timeout")
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="test")

    result = adapter._sign_action("llm:call")
    assert result is None
    assert len(adapter._signatures) == 0


# === Session lifecycle ===


@patch("asqav.extras._base.Agent")
def test_start_session(mock_agent_cls):
    """_start_session delegates to agent.start_session."""
    mock_agent = MagicMock()
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="test")

    adapter._start_session()
    mock_agent.start_session.assert_called_once()
    assert adapter._session_id is not None


@patch("asqav.extras._base.Agent")
def test_end_session(mock_agent_cls):
    """_end_session delegates to agent.end_session and clears session_id."""
    mock_agent = MagicMock()
    mock_agent._session_id = "sess_123"
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="test")

    adapter._session_id = "local_sess"
    adapter._end_session()
    mock_agent.end_session.assert_called_once_with("completed")
    assert adapter._session_id is None


@patch("asqav.extras._base.Agent")
def test_end_session_with_status(mock_agent_cls):
    """_end_session passes custom status to agent."""
    mock_agent = MagicMock()
    mock_agent._session_id = "sess_123"
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="test")

    adapter._session_id = "local_sess"
    adapter._end_session(status="error")
    mock_agent.end_session.assert_called_once_with("error")


@patch("asqav.extras._base.Agent")
def test_end_session_fail_open(mock_agent_cls):
    """_end_session does not raise on AsqavError."""
    mock_agent = MagicMock()
    mock_agent._session_id = "sess_123"
    mock_agent.end_session.side_effect = AsqavError("timeout")
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="test")

    adapter._session_id = "local_sess"
    adapter._end_session()  # should not raise
    assert adapter._session_id is None


@patch("asqav.extras._base.Agent")
def test_end_session_skips_when_no_agent_session(mock_agent_cls):
    """_end_session skips agent.end_session when agent has no active session."""
    mock_agent = MagicMock()
    mock_agent._session_id = None
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="test")

    adapter._end_session()
    mock_agent.end_session.assert_not_called()


# === IETF Compliance Receipts profile pass-through ===


@patch("asqav.extras._base.Agent")
def test_sign_action_forwards_compliance_kwargs(mock_agent_cls):
    """_sign_action forwards IETF profile kwargs verbatim to Agent.sign.

    Subclasses (LangChain callback, CrewAI hook, LiteLLM guardrail, etc)
    rely on this to populate `risk_class`, `iteration_id`,
    `compliance_mode` etc on the signed envelope without duplicating
    the kwarg list per integration.
    """
    mock_agent = MagicMock()
    mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
    mock_agent.sign.return_value = mock_sig
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="test")

    sig = adapter._sign_action(
        "api:call",
        {"k": "v"},
        compliance_mode=True,
        receipt_type="protectmcp:decision",
        risk_class="high",
        iteration_id="iter_42",
    )
    assert sig is mock_sig
    mock_agent.sign.assert_called_once_with(
        "api:call",
        {"k": "v"},
        compliance_mode=True,
        receipt_type="protectmcp:decision",
        risk_class="high",
        iteration_id="iter_42",
    )


@patch("asqav.extras._base.Agent")
def test_sign_action_no_compliance_kwargs_unchanged(mock_agent_cls):
    """_sign_action without compliance kwargs calls Agent.sign as before."""
    mock_agent = MagicMock()
    mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
    mock_agent.sign.return_value = mock_sig
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="test")

    adapter._sign_action("api:call", {"k": "v"})
    mock_agent.sign.assert_called_once_with("api:call", {"k": "v"})


# === c3-lessons regressions: async offload, api_key override, repr ===


@patch("asqav.extras._base.Agent")
def test_sign_action_async_offloads_to_thread(mock_agent_cls):
    """_sign_action_async runs the sync sign off the event loop thread."""
    import asyncio
    import threading

    mock_agent = MagicMock()
    sign_thread: list[str] = []

    def _record_thread(*args, **kwargs):
        sign_thread.append(threading.current_thread().name)
        return SignatureResponse(**MOCK_SIGN_RESPONSE)

    mock_agent.sign.side_effect = _record_thread
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="test")

    async def _run():
        loop_thread = threading.current_thread().name
        result = await adapter._sign_action_async("llm:call", {"model": "gpt-4"})
        return loop_thread, result

    loop_thread, result = asyncio.run(_run())
    assert result is not None
    assert sign_thread, "Agent.sign never ran"
    assert sign_thread[0] != loop_thread, "sync sign blocked the event loop thread"


@patch("asqav.extras._base.Agent")
def test_sign_action_async_observe_skips_network(mock_agent_cls):
    """Observe mode returns None without touching Agent.sign."""
    import asyncio

    mock_agent_cls.create.return_value = MagicMock()
    with patch("asqav.client._api_key", "sk_test"):
        adapter = _TestAdapter(agent_name="test", observe=True)

    result = asyncio.run(adapter._sign_action_async("llm:call", {}))
    assert result is None
    adapter._agent.sign.assert_not_called()


@patch("asqav.extras._base._client.init")
@patch("asqav.extras._base.Agent")
def test_api_key_argument_initialises_client(mock_agent_cls, mock_init):
    """An explicit api_key= actually initialises the client (was silently ignored)."""
    mock_agent_cls.create.return_value = MagicMock()
    with patch("asqav.client._api_key", None):
        _TestAdapter(api_key="sk_adapter", agent_name="test")
    mock_init.assert_called_once_with("sk_adapter")


@patch("asqav.extras._base.Agent")
def test_repr_reports_config_and_omits_api_key(mock_agent_cls):
    """__repr__ surfaces agent identity and observe flag, never the key."""
    mock_agent = MagicMock()
    mock_agent.agent_id = "agent_123"
    mock_agent.name = "my-agent"
    mock_agent_cls.create.return_value = mock_agent

    with patch("asqav.client._api_key", "sk_super_secret"):
        adapter = _TestAdapter(agent_name="my-agent")

    text = repr(adapter)
    assert "agent_123" in text
    assert "my-agent" in text
    assert "observe=False" in text
    assert "sk_super_secret" not in text
