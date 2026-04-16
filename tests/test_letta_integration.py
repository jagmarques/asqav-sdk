"""Tests for letta AsqavLettaHook integration.

Uses a mock letta_client module injected into sys.modules so tests run
without letta-client installed.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

# ---------------------------------------------------------------------------
# Inject a fake letta_client module before importing the hook
# ---------------------------------------------------------------------------

_fake_letta_client = types.ModuleType("letta_client")
sys.modules["letta_client"] = _fake_letta_client
sys.modules.pop("asqav.extras.letta", None)

from asqav.extras.letta import AsqavLettaHook  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    retrieve_result: Any = None,
    update_result: Any = None,
) -> MagicMock:
    """Create a minimal mock letta Letta client."""
    client = MagicMock()
    block_obj = MagicMock()
    block_obj.value = "stored memory content"
    client.agents.blocks.retrieve = MagicMock(
        return_value=retrieve_result if retrieve_result is not None else block_obj
    )
    client.agents.blocks.update = MagicMock(
        return_value=update_result if update_result is not None else MagicMock()
    )
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hook() -> AsqavLettaHook:
    """Create an AsqavLettaHook with mocked Agent internals."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        h = AsqavLettaHook(agent_name="test-letta")
    h._sign_action = MagicMock(return_value=None)
    return h


# ---------------------------------------------------------------------------
# wrap_client: memory:read
# ---------------------------------------------------------------------------


def test_wrap_client_signs_memory_read(hook: AsqavLettaHook) -> None:
    """wrap_client signs memory:read with agent_id and block_label."""
    client = _make_client()
    hook.wrap_client(client)
    client.agents.blocks.retrieve(agent_id="agent-123", block_label="human")

    first_call = hook._sign_action.call_args_list[0]
    assert first_call[0][0] == "memory:read"
    ctx = first_call[0][1]
    assert ctx["agent_id"] == "agent-123"
    assert ctx["block_label"] == "human"


def test_wrap_client_signs_memory_read_end(hook: AsqavLettaHook) -> None:
    """wrap_client signs memory:read.end with value_length after retrieve."""
    block = MagicMock()
    block.value = "hello world"
    client = _make_client(retrieve_result=block)
    hook.wrap_client(client)
    client.agents.blocks.retrieve(agent_id="agent-123", block_label="human")

    last_call = hook._sign_action.call_args_list[-1]
    assert last_call[0][0] == "memory:read.end"
    ctx = last_call[0][1]
    assert ctx["value_length"] == len("hello world")


def test_wrap_client_signs_two_events_on_successful_read(hook: AsqavLettaHook) -> None:
    """A successful retrieve produces exactly memory:read + memory:read.end."""
    client = _make_client()
    hook.wrap_client(client)
    client.agents.blocks.retrieve(agent_id="a", block_label="persona")

    assert hook._sign_action.call_count == 2
    assert hook._sign_action.call_args_list[0][0][0] == "memory:read"
    assert hook._sign_action.call_args_list[1][0][0] == "memory:read.end"


def test_wrap_client_retrieve_returns_original_result(hook: AsqavLettaHook) -> None:
    """wrap_client does not alter the retrieve return value."""
    sentinel = object()
    client = _make_client(retrieve_result=sentinel)
    hook.wrap_client(client)
    result = client.agents.blocks.retrieve(agent_id="a", block_label="b")
    assert result is sentinel


# ---------------------------------------------------------------------------
# wrap_client: memory:write
# ---------------------------------------------------------------------------


def test_wrap_client_signs_memory_write(hook: AsqavLettaHook) -> None:
    """wrap_client signs memory:write with agent_id, block_label, value_length, value_preview."""
    client = _make_client()
    hook.wrap_client(client)
    client.agents.blocks.update(
        agent_id="agent-456", block_label="persona", value="I am a helpful assistant."
    )

    first_call = hook._sign_action.call_args_list[0]
    assert first_call[0][0] == "memory:write"
    ctx = first_call[0][1]
    assert ctx["agent_id"] == "agent-456"
    assert ctx["block_label"] == "persona"
    assert ctx["value_length"] == len("I am a helpful assistant.")
    assert ctx["value_preview"] == "I am a helpful assistant."


def test_wrap_client_signs_memory_write_end(hook: AsqavLettaHook) -> None:
    """wrap_client signs memory:write.end after a successful update."""
    client = _make_client()
    hook.wrap_client(client)
    client.agents.blocks.update(agent_id="a", block_label="human", value="new")

    last_call = hook._sign_action.call_args_list[-1]
    assert last_call[0][0] == "memory:write.end"
    ctx = last_call[0][1]
    assert ctx["agent_id"] == "a"
    assert ctx["block_label"] == "human"


def test_wrap_client_signs_two_events_on_successful_write(hook: AsqavLettaHook) -> None:
    """A successful update produces exactly memory:write + memory:write.end."""
    client = _make_client()
    hook.wrap_client(client)
    client.agents.blocks.update(agent_id="a", block_label="b", value="v")

    assert hook._sign_action.call_count == 2
    assert hook._sign_action.call_args_list[0][0][0] == "memory:write"
    assert hook._sign_action.call_args_list[1][0][0] == "memory:write.end"


def test_wrap_client_update_returns_original_result(hook: AsqavLettaHook) -> None:
    """wrap_client does not alter the update return value."""
    sentinel = object()
    client = _make_client(update_result=sentinel)
    hook.wrap_client(client)
    result = client.agents.blocks.update(agent_id="a", block_label="b", value="v")
    assert result is sentinel


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_wrap_client_signs_read_error_on_exception(hook: AsqavLettaHook) -> None:
    """wrap_client signs memory:read.error when retrieve raises."""
    client = _make_client()
    client.agents.blocks.retrieve.side_effect = RuntimeError("network error")
    hook.wrap_client(client)

    with pytest.raises(RuntimeError):
        client.agents.blocks.retrieve(agent_id="a", block_label="b")

    error_call = hook._sign_action.call_args_list[-1]
    assert error_call[0][0] == "memory:read.error"
    ctx = error_call[0][1]
    assert ctx["error_type"] == "RuntimeError"
    assert ctx["error"] == "network error"


def test_wrap_client_reraises_retrieve_exception(hook: AsqavLettaHook) -> None:
    """wrap_client re-raises the original retrieve exception unchanged."""
    client = _make_client()
    client.agents.blocks.retrieve.side_effect = ValueError("bad agent id")
    hook.wrap_client(client)

    with pytest.raises(ValueError, match="bad agent id"):
        client.agents.blocks.retrieve(agent_id="a", block_label="b")


def test_wrap_client_signs_write_error_on_exception(hook: AsqavLettaHook) -> None:
    """wrap_client signs memory:write.error when update raises."""
    client = _make_client()
    client.agents.blocks.update.side_effect = RuntimeError("write failed")
    hook.wrap_client(client)

    with pytest.raises(RuntimeError):
        client.agents.blocks.update(agent_id="a", block_label="b", value="v")

    error_call = hook._sign_action.call_args_list[-1]
    assert error_call[0][0] == "memory:write.error"
    ctx = error_call[0][1]
    assert ctx["error_type"] == "RuntimeError"
    assert ctx["error"] == "write failed"


def test_wrap_client_reraises_update_exception(hook: AsqavLettaHook) -> None:
    """wrap_client re-raises the original update exception unchanged."""
    client = _make_client()
    client.agents.blocks.update.side_effect = PermissionError("read only")
    hook.wrap_client(client)

    with pytest.raises(PermissionError, match="read only"):
        client.agents.blocks.update(agent_id="a", block_label="b", value="v")


def test_wrap_client_signs_read_start_and_error_on_failure(hook: AsqavLettaHook) -> None:
    """A failed retrieve produces exactly memory:read + memory:read.error."""
    client = _make_client()
    client.agents.blocks.retrieve.side_effect = RuntimeError("boom")
    hook.wrap_client(client)

    with pytest.raises(RuntimeError):
        client.agents.blocks.retrieve(agent_id="a", block_label="b")

    assert hook._sign_action.call_count == 2
    assert hook._sign_action.call_args_list[0][0][0] == "memory:read"
    assert hook._sign_action.call_args_list[1][0][0] == "memory:read.error"


def test_wrap_client_signs_write_start_and_error_on_failure(hook: AsqavLettaHook) -> None:
    """A failed update produces exactly memory:write + memory:write.error."""
    client = _make_client()
    client.agents.blocks.update.side_effect = RuntimeError("boom")
    hook.wrap_client(client)

    with pytest.raises(RuntimeError):
        client.agents.blocks.update(agent_id="a", block_label="b", value="v")

    assert hook._sign_action.call_count == 2
    assert hook._sign_action.call_args_list[0][0][0] == "memory:write"
    assert hook._sign_action.call_args_list[1][0][0] == "memory:write.error"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_agent_id_truncated_to_200_chars(hook: AsqavLettaHook) -> None:
    """agent_id longer than 200 chars is truncated in the audit record."""
    client = _make_client()
    hook.wrap_client(client)
    client.agents.blocks.retrieve(agent_id="a" * 500, block_label="human")

    ctx = hook._sign_action.call_args_list[0][0][1]
    assert len(ctx["agent_id"]) == 200


def test_value_preview_truncated_to_200_chars(hook: AsqavLettaHook) -> None:
    """value_preview longer than 200 chars is truncated in the audit record."""
    client = _make_client()
    hook.wrap_client(client)
    client.agents.blocks.update(agent_id="a", block_label="b", value="x" * 500)

    ctx = hook._sign_action.call_args_list[0][0][1]
    assert len(ctx["value_preview"]) == 200
    assert ctx["value_length"] == 500


def test_error_message_truncated_to_200_chars(hook: AsqavLettaHook) -> None:
    """Error messages longer than 200 chars are truncated in the audit record."""
    client = _make_client()
    client.agents.blocks.retrieve.side_effect = RuntimeError("e" * 500)
    hook.wrap_client(client)

    with pytest.raises(RuntimeError):
        client.agents.blocks.retrieve(agent_id="a", block_label="b")

    ctx = hook._sign_action.call_args_list[-1][0][1]
    assert len(ctx["error"]) == 200


# ---------------------------------------------------------------------------
# Fail-open: signing must never block memory operations
# ---------------------------------------------------------------------------


def test_fail_open_memory_read_does_not_block_retrieve(hook: AsqavLettaHook) -> None:
    """If memory:read signing raises, retrieve still executes and returns its result."""
    sentinel = object()
    client = _make_client(retrieve_result=sentinel)
    hook._sign_action.side_effect = Exception("service unavailable")
    hook.wrap_client(client)

    result = client.agents.blocks.retrieve(agent_id="a", block_label="b")
    assert result is sentinel


def test_fail_open_memory_write_does_not_block_update(hook: AsqavLettaHook) -> None:
    """If memory:write signing raises, update still executes and returns its result."""
    sentinel = object()
    client = _make_client(update_result=sentinel)
    hook._sign_action.side_effect = Exception("service unavailable")
    hook.wrap_client(client)

    result = client.agents.blocks.update(agent_id="a", block_label="b", value="v")
    assert result is sentinel


def test_fail_open_memory_write_end_does_not_suppress_result(
    hook: AsqavLettaHook,
) -> None:
    """If memory:write.end signing raises, the update result is still returned."""
    sentinel = object()
    client = _make_client(update_result=sentinel)
    call_count = 0

    def sign_side_effect(action_type: str, ctx: dict) -> None:
        nonlocal call_count
        call_count += 1
        if action_type == "memory:write.end":
            raise Exception("end signing failed")

    hook._sign_action.side_effect = sign_side_effect
    hook.wrap_client(client)

    result = client.agents.blocks.update(agent_id="a", block_label="b", value="v")
    assert result is sentinel


def test_fail_open_read_error_signing_does_not_swallow_exception(
    hook: AsqavLettaHook,
) -> None:
    """If memory:read.error signing raises, the original retrieve exception is still re-raised."""
    client = _make_client()
    client.agents.blocks.retrieve.side_effect = ValueError("retrieve failed")
    hook._sign_action.side_effect = Exception("signing also broke")
    hook.wrap_client(client)

    with pytest.raises(ValueError, match="retrieve failed"):
        client.agents.blocks.retrieve(agent_id="a", block_label="b")


# ---------------------------------------------------------------------------
# Multiple clients tracked independently
# ---------------------------------------------------------------------------


def test_multiple_clients_tracked_independently(hook: AsqavLettaHook) -> None:
    """Each wrapped client is signed independently."""
    client_a = _make_client()
    client_b = _make_client()

    hook.wrap_client(client_a)
    hook.wrap_client(client_b)

    client_a.agents.blocks.retrieve(agent_id="agent-a", block_label="human")
    client_b.agents.blocks.update(agent_id="agent-b", block_label="persona", value="hi")

    calls = hook._sign_action.call_args_list
    assert calls[0][0][1]["agent_id"] == "agent-a"
    assert calls[2][0][1]["agent_id"] == "agent-b"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _cleanup_letta_client_module() -> Any:
    """Remove fake letta_client from sys.modules after all tests."""
    yield
    sys.modules.pop("letta_client", None)
    sys.modules.pop("asqav.extras.letta", None)
