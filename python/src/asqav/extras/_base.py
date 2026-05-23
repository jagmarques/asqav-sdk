"""Base adapter class for Asqav framework integrations.

Isolates signing logic from framework callback/hook APIs so breaking changes
affect only the thin framework-facing layer.
"""

from __future__ import annotations

import logging
import re
import uuid

from .. import client as _client
from ..client import Agent, AsqavError, SignatureResponse

logger = logging.getLogger("asqav")


def _class_name_to_agent_name(cls_name: str) -> str:
    """Convert CamelCase class name to kebab-case (e.g. AsqavCrewHook -> asqav-crew-hook)."""
    # Insert hyphen before uppercase letters, then lowercase
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"-\1", cls_name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", s)
    return s.lower()


class AsqavAdapter:
    """Base class for Asqav framework integrations.

    Subclasses call ``_sign_action`` to record governance events. Agent name
    defaults to a kebab-cased subclass class name when neither ``agent_name``
    nor ``agent_id`` is provided.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        observe: bool = False,
    ) -> None:
        self._observe = observe

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
        **compliance_kwargs,
    ) -> SignatureResponse | None:
        """Sign an action via the agent; returns ``None`` on failure (fail-open).

        ``compliance_kwargs`` forwards IETF Compliance Receipts fields to
        ``Agent.sign``.
        """
        if self._observe:
            logger.info(
                "OBSERVE: would sign %s with context %s", action_type, context
            )
            return None
        try:
            sig = self._agent.sign(action_type, context, **compliance_kwargs)
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
