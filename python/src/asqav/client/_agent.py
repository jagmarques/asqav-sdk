"""The Agent dataclass: create, get and delegate."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import _transport
from ._agent_session import _AgentSession
from ._agent_sign import _AgentSign
from ._agent_signing import _AgentSigning
from ._agent_tokens import _AgentTokens
from ._common import _parse_timestamp
from ._errors import require_field


@dataclass
class Agent(_AgentTokens, _AgentSign, _AgentSigning, _AgentSession):
    """Agent representation from Asqav Cloud.

    All ML-DSA cryptography happens server-side.
    This is a thin client that wraps API calls.
    """

    agent_id: str
    name: str
    public_key: str
    key_id: str
    algorithm: str
    capabilities: list[str]
    created_at: float
    _session_id: str | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        name: str,
        algorithm: str = "ml-dsa-65",
        capabilities: list[str] | None = None,
    ) -> Agent:
        """Create a new agent via Asqav Cloud.

        The server generates the keypair. The private key never leaves
        the server.

        Args:
            name: Human-readable name for the agent.
            algorithm: Cloud agent-signing algorithm. The cloud accepts
                `ml-dsa-44`, `ml-dsa-65` (FIPS 204; default), and
                `ml-dsa-87`. `ed25519` and `es256` are local-keygen only
                (see `generate_local_keypair`) and 400 here.
            capabilities: List of capabilities/permissions.

        Returns:
            An Agent instance.

        Raises:
            AuthenticationError: If API key is missing or invalid.
            APIError: If the request fails.
        """
        data = _transport._post(
            "/agents/create",
            {
                "name": name,
                "algorithm": algorithm,
                "capabilities": capabilities or [],
            },
        )

        return cls(
            agent_id=require_field(data, "agent_id"),
            name=require_field(data, "name"),
            public_key=require_field(data, "public_key"),
            key_id=require_field(data, "key_id"),
            algorithm=require_field(data, "algorithm"),
            capabilities=require_field(data, "capabilities"),
            created_at=_parse_timestamp(require_field(data, "created_at")),
        )

    @classmethod
    def get(cls, agent_id: str) -> Agent:
        """Get an existing agent by ID.

        Args:
            agent_id: The agent ID to retrieve.

        Returns:
            An Agent instance.
        """
        data = _transport._get(f"/agents/{agent_id}")

        return cls(
            agent_id=require_field(data, "agent_id"),
            name=require_field(data, "name"),
            public_key=require_field(data, "public_key"),
            key_id=require_field(data, "key_id"),
            algorithm=require_field(data, "algorithm"),
            capabilities=require_field(data, "capabilities"),
            created_at=_parse_timestamp(require_field(data, "created_at")),
        )

    def delegate(
        self,
        name: str,
        scope: list[str] | None = None,
        ttl: int = 86400,
    ) -> "Agent":
        """Create a delegated child agent with limited scope.

        Args:
            name: Name for the child agent.
            scope: List of capabilities to delegate.
            ttl: Time-to-live in seconds (default 24h, max 7 days).

        Returns:
            Agent: The delegated child agent.
        """
        data = _transport._post(
            f"/agents/{self.agent_id}/delegate",
            {
                "name": name,
                "scope": scope or [],
                "ttl": ttl,
            },
        )

        return Agent(
            agent_id=data["child_id"],
            name=data["child_name"],
            public_key="",  # Use Agent.get() to fetch full details
            key_id="",
            algorithm=self.algorithm,
            capabilities=data["scope"],
            created_at=_parse_timestamp(data["created_at"]),
        )
