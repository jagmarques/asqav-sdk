"""smolagents (Hugging Face) integration for asqav.

Install: pip install asqav[smolagents]

Usage::

    from smolagents import tool, CodeAgent, HfApiModel
    from asqav.extras.smolagents import AsqavSmolagentsHook

    hook = AsqavSmolagentsHook(agent_name="my-smolagent")

    @tool
    def search_web(query: str) -> str:
        \"\"\"Search the web for information.

        Args:
            query: The search query string.
        \"\"\"
        return "results"

    signed_search = hook.wrap_tool(search_web)
    agent = CodeAgent(tools=[signed_search], model=HfApiModel())
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import smolagents  # noqa: F401
except ImportError:
    raise ImportError(
        "smolagents integration requires smolagents. "
        "Install with: pip install asqav[smolagents]"
    )

from ._base import AsqavAdapter

logger = logging.getLogger("asqav")

_MAX_LEN = 200


class AsqavSmolagentsHook(AsqavAdapter):
    """Hook that signs smolagents tool call events.

    Wraps smolagents Tool instances to sign ``tool:start``, ``tool:end``,
    and ``tool:error`` governance events on every tool invocation. All
    signing is fail-open - governance failures are logged but never
    interrupt tool or agent execution.

    Args:
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for a new asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing asqav agent (calls ``Agent.get``).
    """

    def wrap_tool(self, tool: Any) -> Any:
        """Wrap a smolagents Tool instance to sign its calls.

        Patches the ``forward`` method on the tool instance so every
        invocation records signed governance events. The original tool
        object is returned so it can be passed directly to smolagents
        agents unchanged.

        Args:
            tool: A smolagents Tool instance (from the ``@tool`` decorator
                  or a ``Tool`` subclass).

        Returns:
            The same tool instance with signing wired into ``forward``.
        """
        hook = self
        original_forward = tool.forward
        tool_name: str = getattr(tool, "name", type(tool).__name__)

        def _signed_forward(*args: Any, **kwargs: Any) -> Any:
            # Build a concise input preview for the audit record.
            if args:
                input_preview = str(args[0])[:_MAX_LEN]
            elif kwargs:
                input_preview = str(next(iter(kwargs.values())))[:_MAX_LEN]
            else:
                input_preview = ""

            try:
                hook._sign_action(
                    "tool:start",
                    {"tool": tool_name, "input": input_preview},
                )
            except Exception as exc:
                logger.warning("asqav tool:start signing failed (fail-open): %s", exc)

            try:
                result = original_forward(*args, **kwargs)
            except Exception as exc:
                try:
                    hook._sign_action(
                        "tool:error",
                        {
                            "tool": tool_name,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:_MAX_LEN],
                        },
                    )
                except Exception as sign_exc:
                    logger.warning(
                        "asqav tool:error signing failed (fail-open): %s", sign_exc
                    )
                raise

            try:
                hook._sign_action(
                    "tool:end",
                    {
                        "tool": tool_name,
                        "output_type": type(result).__name__,
                        "output_length": len(str(result)),
                    },
                )
            except Exception as exc:
                logger.warning("asqav tool:end signing failed (fail-open): %s", exc)

            return result

        tool.forward = _signed_forward
        return tool
