"""letta (MemGPT) integration for asqav.

Install: pip install asqav[letta]

Usage::

    from letta_client import Letta
    from asqav.extras.letta import AsqavLettaHook

    hook = AsqavLettaHook(agent_name="memory-agent")
    client = hook.wrap_client(Letta(api_key="..."))

    # All memory reads and writes are now signed.
    block = client.agents.blocks.retrieve(agent_id="agent-123", block_label="human")
    client.agents.blocks.update(agent_id="agent-123", block_label="human", value="New content")
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import letta_client  # noqa: F401
except ImportError:
    raise ImportError(
        "letta integration requires letta-client. "
        "Install with: pip install asqav[letta]"
    )

from ._base import AsqavAdapter

logger = logging.getLogger("asqav")

_MAX_LEN = 200


class AsqavLettaHook(AsqavAdapter):
    """Hook that signs letta memory read/write governance events.

    Wraps a letta ``Letta`` client instance to intercept memory block
    operations. Signs ``memory:read``, ``memory:read.end``,
    ``memory:read.error``, ``memory:write``, ``memory:write.end``, and
    ``memory:write.error`` events on every memory block access. All
    signing is fail-open — governance failures are logged but never
    interrupt client or agent execution.

    Args:
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for a new asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing asqav agent (calls ``Agent.get``).
    """

    def wrap_client(self, client: Any) -> Any:
        """Wrap a letta Letta client to sign memory block operations.

        Patches ``client.agents.blocks.retrieve`` and
        ``client.agents.blocks.update`` in-place so every memory read and
        write records signed governance events. The original client object
        is returned so it can be used normally.

        Args:
            client: A ``letta_client.Letta`` (or ``AsyncLetta``) instance.

        Returns:
            The same client instance with signing wired into the blocks API.
        """
        hook = self
        blocks = client.agents.blocks
        original_retrieve = blocks.retrieve
        original_update = blocks.update

        def _signed_retrieve(agent_id: str, block_label: str, **kwargs: Any) -> Any:
            try:
                hook._sign_action(
                    "memory:read",
                    {
                        "agent_id": str(agent_id)[:_MAX_LEN],
                        "block_label": block_label,
                    },
                )
            except Exception as exc:
                logger.warning("asqav memory:read signing failed (fail-open): %s", exc)

            try:
                result = original_retrieve(
                    agent_id=agent_id, block_label=block_label, **kwargs
                )
            except Exception as exc:
                try:
                    hook._sign_action(
                        "memory:read.error",
                        {
                            "agent_id": str(agent_id)[:_MAX_LEN],
                            "block_label": block_label,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:_MAX_LEN],
                        },
                    )
                except Exception as sign_exc:
                    logger.warning(
                        "asqav memory:read.error signing failed (fail-open): %s", sign_exc
                    )
                raise

            try:
                hook._sign_action(
                    "memory:read.end",
                    {
                        "agent_id": str(agent_id)[:_MAX_LEN],
                        "block_label": block_label,
                        "value_length": len(str(getattr(result, "value", result))),
                    },
                )
            except Exception as exc:
                logger.warning(
                    "asqav memory:read.end signing failed (fail-open): %s", exc
                )

            return result

        def _signed_update(
            agent_id: str, block_label: str, value: str, **kwargs: Any
        ) -> Any:
            try:
                hook._sign_action(
                    "memory:write",
                    {
                        "agent_id": str(agent_id)[:_MAX_LEN],
                        "block_label": block_label,
                        "value_length": len(str(value)),
                        "value_preview": str(value)[:_MAX_LEN],
                    },
                )
            except Exception as exc:
                logger.warning("asqav memory:write signing failed (fail-open): %s", exc)

            try:
                result = original_update(
                    agent_id=agent_id, block_label=block_label, value=value, **kwargs
                )
            except Exception as exc:
                try:
                    hook._sign_action(
                        "memory:write.error",
                        {
                            "agent_id": str(agent_id)[:_MAX_LEN],
                            "block_label": block_label,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:_MAX_LEN],
                        },
                    )
                except Exception as sign_exc:
                    logger.warning(
                        "asqav memory:write.error signing failed (fail-open): %s", sign_exc
                    )
                raise

            try:
                hook._sign_action(
                    "memory:write.end",
                    {
                        "agent_id": str(agent_id)[:_MAX_LEN],
                        "block_label": block_label,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "asqav memory:write.end signing failed (fail-open): %s", exc
                )

            return result

        blocks.retrieve = _signed_retrieve
        blocks.update = _signed_update
        return client
