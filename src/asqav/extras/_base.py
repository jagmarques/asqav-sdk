"""Base adapter class for asqav framework integrations.

All framework-specific adapters (LangChain, CrewAI, LiteLLM, Haystack,
OpenAI Agents) extend AsqavAdapter. This isolates signing logic from
framework callback/hook APIs so framework breaking changes only affect
the thin framework-facing layer.
"""

from __future__ import annotations

import logging
import re
import uuid

from .. import client as _client
from ..client import Agent, AsqavError, SignatureResponse

logger = logging.getLogger("asqav")


def _class_name_to_agent_name(cls_name: str) -> str:
    """Convert CamelCase class name to kebab-case agent name.

    Examples:
        AsqavCallbackHandler -> asqav-callback-handler
        AsqavCrewHook -> asqav-crew-hook
    """
    # Insert hyphen before uppercase letters, then lowercase
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"-\1", cls_name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", s)
    return s.lower()


class AsqavAdapter:
    """Base class for asqav framework integrations.

    Provides shared helpers for agent initialization, action signing,
    and optional session grouping. Subclasses implement framework-specific
    interfaces (callbacks, hooks, guardrails, components) and call
    ``_sign_action`` to record governance events.

    Args:
        api_key: Optional API key override (uses asqav.init() default).
        agent_name: Name for a new agent (calls Agent.create).
        agent_id: ID of an existing agent (calls Agent.get).

    If neither ``agent_name`` nor ``agent_id`` is provided, the agent name
    is auto-generated from the subclass class name.

    Raises:
        AsqavError: If ``asqav.init()`` has not been called.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        if _client._api_key is None and api_key is None:
            raise AsqavError("Call asqav.init() first")

        if agent_id is not None:
            self._agent: Agent = Agent.get(agent_id)
        else:
            name = agent_name or _class_name_to_agent_name(type(self).__name__)
            self._agent = Agent.create(name)

        self._signatures: list[SignatureResponse] = []
        self._session_id: str | None = None

    def _sign_action(
        self,
        action_type: str,
        context: dict | None = None,
    ) -> SignatureResponse | None:
        """Sign an action via the agent. Returns None on failure (fail-open).

        Governance failures are logged but never raise - the user's AI
        pipeline must not break because of asqav availability issues.
        """
        try:
            sig = self._agent.sign(action_type, context)
            self._signatures.append(sig)
            return sig
        except AsqavError as exc:
            logger.warning("asqav signing failed (fail-open): %s", exc)
            return None

    def _start_session(self) -> None:
        """Start a session to group related signatures."""
        self._session_id = uuid.uuid4().hex
        self._agent.start_session()

    def _end_session(self, status: str = "completed") -> None:
        """End the current session."""
        if self._agent._session_id is not None:
            try:
                self._agent.end_session(status)
            except AsqavError as exc:
                logger.warning("asqav end_session failed: %s", exc)
        self._session_id = None
