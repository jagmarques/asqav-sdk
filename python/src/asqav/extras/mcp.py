"""MCP (Model Context Protocol) integration for asqav.

Install: pip install asqav[mcp]

Usage::

    from mcp.server.fastmcp import FastMCP
    from asqav.extras.mcp import enable_mcp_governance

    server = FastMCP("my-server")
    enable_mcp_governance(server, agent_name="my-mcp-server")

    @server.tool()
    def add(a: int, b: int) -> int:
        return a + b

One call to ``enable_mcp_governance`` turns governance on for the whole
server: every tool call signs an asqav receipt by default, with no
per-call flag. Signing is fail-open - a signing error is logged and never
blocks or alters the tool call. Importing this module without calling the
function changes nothing, and ``mcp`` is an optional extra (this module
duck-types the server, it never imports ``mcp`` itself).
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from ._base import AsqavAdapter

logger = logging.getLogger("asqav")

__all__ = ["AsqavMCPAdapter", "enable_mcp_governance"]

_MAX_LEN = 200


class AsqavMCPAdapter(AsqavAdapter):
    """Signs an asqav governance receipt on every MCP tool call.

    ``instrument`` wraps the server's tool-dispatch funnel so each call
    records ``tool:call`` (and ``tool:error`` when the tool raises). It
    handles a FastMCP server (wraps the ``_tool_manager`` dispatch that both
    the direct API and real request routing funnel through) and a low-level
    ``Server`` (wraps the ``call_tool`` decorator so registered handlers
    sign). Fail-open throughout.

    Args:
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for an asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing asqav agent (calls ``Agent.get``).
        observe: When True, log intent instead of signing.
    """

    def instrument(self, server: Any) -> Any:
        """Wire signing into ``server``'s tool dispatch; returns ``server``."""
        tool_manager = getattr(server, "_tool_manager", None)
        if tool_manager is not None and inspect.iscoroutinefunction(
            getattr(tool_manager, "call_tool", None)
        ):
            self._wrap_dispatch(tool_manager, "call_tool")
            return server
        if inspect.iscoroutinefunction(getattr(server, "call_tool", None)):
            self._wrap_dispatch(server, "call_tool")
            return server
        if callable(getattr(server, "call_tool", None)):
            self._wrap_lowlevel_decorator(server)
            return server
        raise TypeError(
            "enable_mcp_governance expects a FastMCP server or a low-level mcp "
            f"Server exposing call_tool; got {type(server).__name__}."
        )

    def _wrap_dispatch(self, target: Any, attr: str) -> None:
        # Single async funnel: wrap once so both direct calls and routing sign.
        original = getattr(target, attr)
        if getattr(original, "_asqav_wrapped", False):
            return
        adapter = self

        async def _signed_dispatch(
            name: str, arguments: Any, *args: Any, **kwargs: Any
        ) -> Any:
            await adapter._sign_tool_call(name, arguments)
            try:
                return await original(name, arguments, *args, **kwargs)
            except Exception as exc:
                await adapter._sign_tool_error(name, exc)
                raise

        _signed_dispatch._asqav_wrapped = True  # type: ignore[attr-defined]
        setattr(target, attr, _signed_dispatch)

    def _wrap_lowlevel_decorator(self, server: Any) -> None:
        # Low-level Server.call_tool is a decorator; wrap the handler it registers.
        original_call_tool = server.call_tool
        if getattr(original_call_tool, "_asqav_wrapped", False):
            return
        adapter = self

        def _patched_call_tool(*d_args: Any, **d_kwargs: Any) -> Any:
            real_decorator = original_call_tool(*d_args, **d_kwargs)

            def _wrapping_decorator(func: Any) -> Any:
                async def _signed_handler(
                    name: str, arguments: Any, *a: Any, **k: Any
                ) -> Any:
                    await adapter._sign_tool_call(name, arguments)
                    try:
                        return await func(name, arguments, *a, **k)
                    except Exception as exc:
                        await adapter._sign_tool_error(name, exc)
                        raise

                return real_decorator(_signed_handler)

            return _wrapping_decorator

        _patched_call_tool._asqav_wrapped = True  # type: ignore[attr-defined]
        server.call_tool = _patched_call_tool

    async def _sign_tool_call(self, name: str, arguments: Any) -> None:
        """Sign tool:call for one MCP tool invocation (fail-open)."""
        try:
            arg_keys = sorted(arguments.keys()) if isinstance(arguments, dict) else []
            await self._sign_action_async(
                "tool:call",
                {"tool": str(name), "arg_keys": arg_keys},
            )
        except Exception as exc:
            logger.warning("asqav mcp tool:call signing failed (fail-open): %s", exc)

    async def _sign_tool_error(self, name: str, exc: BaseException) -> None:
        """Sign tool:error when a wrapped tool raises (fail-open)."""
        try:
            await self._sign_action_async(
                "tool:error",
                {
                    "tool": str(name),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:_MAX_LEN],
                },
            )
        except Exception as sign_exc:
            logger.warning(
                "asqav mcp tool:error signing failed (fail-open): %s", sign_exc
            )


def enable_mcp_governance(
    server: Any,
    *,
    api_key: str | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    observe: bool = False,
) -> AsqavMCPAdapter:
    """Turn on default-on asqav governance for an MCP server.

    After this one call, every tool call on ``server`` signs a governance
    receipt. The server is instrumented in place; the returned adapter holds
    the recorded signatures.
    """
    adapter = AsqavMCPAdapter(
        api_key=api_key,
        agent_name=agent_name or "asqav-mcp-server",
        agent_id=agent_id,
        observe=observe,
    )
    adapter.instrument(server)
    return adapter
