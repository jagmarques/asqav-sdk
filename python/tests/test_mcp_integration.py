"""Tests for the MCP AsqavMCPAdapter / enable_mcp_governance integration.

Uses fake servers that mimic the real dispatch funnels (FastMCP's
``_tool_manager.call_tool`` and the low-level ``Server.call_tool``
decorator) so tests run without ``mcp`` installed. Async dispatch is
driven with ``asyncio.run`` so no pytest-asyncio config is needed.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from asqav.extras.mcp import AsqavMCPAdapter, enable_mcp_governance  # noqa: E402


# === Fake servers ===


class FakeToolManager:
    """Mimics FastMCP's _tool_manager.call_tool dispatch funnel."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._result = result if result is not None else [{"type": "text", "text": "ok"}]
        self._error = error

    async def call_tool(
        self, name: str, arguments: dict, context: Any = None, convert_result: bool = False
    ) -> Any:
        self.calls.append((name, arguments))
        if self._error is not None:
            raise self._error
        return self._result


class FakeFastMCP:
    """Mimics FastMCP: call_tool delegates to _tool_manager.call_tool."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self._tool_manager = FakeToolManager(result=result, error=error)

    async def call_tool(self, name: str, arguments: dict) -> Any:
        return await self._tool_manager.call_tool(name, arguments)


class FakeLowLevelServer:
    """Mimics mcp.server.lowlevel.Server.call_tool decorator registration."""

    def __init__(self) -> None:
        self.handler: Any = None

    def call_tool(self, *, validate_input: bool = True) -> Any:
        def decorator(func: Any) -> Any:
            self.handler = func
            return func

        return decorator


# === Fixtures ===


def _build_adapter(server: Any, *, instrument: bool = True) -> AsqavMCPAdapter:
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        if instrument:
            adapter = enable_mcp_governance(server, agent_name="test-mcp")
        else:
            adapter = AsqavMCPAdapter(agent_name="test-mcp")
    adapter._sign_action = MagicMock(return_value=None)  # type: ignore[method-assign]
    return adapter


# === Allow path ===


def test_tool_call_returns_original_result() -> None:
    """A governed tool call returns the tool's result unchanged."""
    server = FakeFastMCP(result=[{"type": "text", "text": "42"}])
    _build_adapter(server)
    out = asyncio.run(server.call_tool("add", {"a": 1, "b": 2}))
    assert out == [{"type": "text", "text": "42"}]
    assert server._tool_manager.calls == [("add", {"a": 1, "b": 2})]


# === Default-on governance ===


def test_default_on_signs_tool_call() -> None:
    """enable_mcp_governance makes every tool call sign tool:call by default."""
    server = FakeFastMCP()
    adapter = _build_adapter(server)
    asyncio.run(server.call_tool("search", {"q": "x"}))
    action_types = [c[0][0] for c in adapter._sign_action.call_args_list]
    assert "tool:call" in action_types
    ctx = adapter._sign_action.call_args_list[0][0][1]
    assert ctx["tool"] == "search"
    assert ctx["arg_keys"] == ["q"]


def test_non_vacuous_uninstrumented_server_signs_nothing() -> None:
    """Control: without the wrap the same server signs nothing.

    The instrument() wrap is what drives signing, not the fixture or import.
    Break the wiring (skip instrument) and governance must not fire - this is
    the mutation control for the default-on test above.
    """
    server = FakeFastMCP()
    adapter = _build_adapter(server, instrument=False)
    asyncio.run(server.call_tool("noop", {"k": 1}))
    adapter._sign_action.assert_not_called()


# === Tool error path ===


def test_tool_error_is_signed_and_reraised() -> None:
    """When a tool raises, tool:error is signed and the error re-raised."""
    server = FakeFastMCP(error=RuntimeError("tool exploded"))
    adapter = _build_adapter(server)
    with pytest.raises(RuntimeError, match="tool exploded"):
        asyncio.run(server.call_tool("boom", {"x": 1}))
    action_types = [c[0][0] for c in adapter._sign_action.call_args_list]
    assert action_types == ["tool:call", "tool:error"]
    err_ctx = adapter._sign_action.call_args_list[-1][0][1]
    assert err_ctx["error_type"] == "RuntimeError"
    assert err_ctx["error"] == "tool exploded"


# === Fail-open ===


def test_fail_open_signing_error_does_not_block_tool() -> None:
    """A signing failure never blocks or alters the tool call."""
    server = FakeFastMCP(result="done")
    adapter = _build_adapter(server)
    adapter._sign_action.side_effect = Exception("sign service down")
    out = asyncio.run(server.call_tool("add", {"a": 1}))
    assert out == "done"


# === Low-level Server decorator path ===


def test_lowlevel_decorator_handler_signs() -> None:
    """Wrapping the low-level call_tool decorator signs registered handlers."""
    server = FakeLowLevelServer()
    adapter = _build_adapter(server)

    @server.call_tool()
    async def handle(name: str, arguments: dict) -> Any:
        return [{"type": "text", "text": "done"}]

    out = asyncio.run(server.handler("mytool", {"k": 1}))
    assert out == [{"type": "text", "text": "done"}]
    action_types = [c[0][0] for c in adapter._sign_action.call_args_list]
    assert action_types == ["tool:call"]
    assert adapter._sign_action.call_args_list[0][0][1]["tool"] == "mytool"


# === Idempotency ===


def test_enable_is_idempotent() -> None:
    """Enabling twice on one server signs exactly once per tool call."""
    server = FakeFastMCP()
    adapter = _build_adapter(server)
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        enable_mcp_governance(server, agent_name="again")
    asyncio.run(server.call_tool("t", {"a": 1}))
    action_types = [c[0][0] for c in adapter._sign_action.call_args_list]
    assert action_types.count("tool:call") == 1


# === Bad target ===


def test_instrument_rejects_non_mcp_object() -> None:
    """enable_mcp_governance raises TypeError on an object with no call_tool."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent_cls.create.return_value = MagicMock()
        with pytest.raises(TypeError, match="call_tool"):
            enable_mcp_governance(object(), agent_name="x")
