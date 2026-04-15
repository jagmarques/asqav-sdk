"""Portable scope tokens for cross-org agent verification.

Thin wrapper around the existing SD-JWT token API. Scope tokens let an
agent carry proof of what it can do into external services, so the
receiving side can verify permissions without calling back to your org.

All cryptography happens server-side via issue_sd_token(). This module
just adds a convenient ScopeToken envelope with helpers for attaching
the token to HTTP requests, checking expiry, and selecting which scopes
to reveal.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .patterns import resolve_pattern

if TYPE_CHECKING:
    from .client import Agent, SDTokenResponse


@dataclass
class ScopeToken:
    """A portable proof of agent permissions.

    Wraps an SDTokenResponse with convenience methods for carrying the
    token across service boundaries.
    """

    token: str
    jwt: str
    disclosures: dict[str, str]
    expires_at: float
    actions: list[str]
    agent_id: str
    nonce: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- helpers --------------------------------------------------------------

    def to_header(self, disclose: list[str] | None = None) -> dict[str, str]:
        """Return an Authorization header dict for HTTP requests.

        Args:
            disclose: Specific scope claims to reveal. When None the full
                token (all disclosures) is used.

        Returns:
            Dict suitable for ``requests.get(url, headers=token.to_header())``.
        """
        if disclose is not None:
            parts = [self.jwt]
            for name in disclose:
                if name in self.disclosures:
                    parts.append(self.disclosures[name])
            value = "~".join(parts) + "~"
        else:
            value = self.token
        return {"Authorization": f"Bearer {value}"}

    def present(self, disclose: list[str]) -> str:
        """Create a selective-disclosure presentation.

        Delegates to the same logic as SDTokenResponse.present().

        Args:
            disclose: Claim names to reveal.

        Returns:
            SD-JWT string with only the listed disclosures.
        """
        parts = [self.jwt]
        for name in disclose:
            if name in self.disclosures:
                parts.append(self.disclosures[name])
        return "~".join(parts) + "~"

    def is_expired(self) -> bool:
        """Check if the token has expired."""
        return time.time() >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for storage or transport."""
        return {
            "token": self.token,
            "jwt": self.jwt,
            "disclosures": self.disclosures,
            "expires_at": self.expires_at,
            "actions": self.actions,
            "agent_id": self.agent_id,
            "nonce": self.nonce,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScopeToken:
        """Reconstruct a ScopeToken from a dict (e.g. after deserialization)."""
        return cls(
            token=data["token"],
            jwt=data["jwt"],
            disclosures=data["disclosures"],
            expires_at=data["expires_at"],
            actions=data["actions"],
            agent_id=data["agent_id"],
            nonce=data.get("nonce", ""),
            metadata=data.get("metadata", {}),
        )


# -- factory ----------------------------------------------------------------


def create_scope_token(
    agent: Agent,
    actions: list[str],
    ttl: int = 3600,
    metadata: dict[str, Any] | None = None,
) -> ScopeToken:
    """Issue a portable scope token for an agent.

    Resolves semantic action patterns, builds the SD-JWT claims via the
    existing ``agent.issue_sd_token()`` call, and wraps the response in
    a ScopeToken.

    Args:
        agent: The Agent instance to issue the token for.
        actions: Action patterns (semantic names or raw globs).
        ttl: Token time-to-live in seconds.
        metadata: Optional extra claims to embed.

    Returns:
        A ScopeToken ready to attach to outbound requests.
    """
    resolved = [resolve_pattern(a) for a in actions]
    nonce = uuid.uuid4().hex

    claims: dict[str, Any] = {
        "scope_actions": resolved,
        "agent_id": agent.agent_id,
        "agent_name": agent.name,
        "nonce": nonce,
    }
    if metadata:
        claims["metadata"] = metadata

    disclosable = ["scope_actions", "agent_id", "agent_name"]
    if metadata:
        disclosable.append("metadata")

    sd_resp: SDTokenResponse = agent.issue_sd_token(
        claims=claims,
        disclosable=disclosable,
        ttl=ttl,
    )

    return ScopeToken(
        token=sd_resp.token,
        jwt=sd_resp.jwt,
        disclosures=sd_resp.disclosures,
        expires_at=sd_resp.expires_at,
        actions=resolved,
        agent_id=agent.agent_id,
        nonce=nonce,
        metadata=metadata or {},
    )


def verify_scope_token(signature_id: str) -> dict[str, Any]:
    """Verify a scope token via the public verification endpoint.

    Delegates to the existing ``verify_signature()`` function - no new
    crypto, just a convenience alias that returns a dict.

    Args:
        signature_id: The signature or token ID to verify.

    Returns:
        Dict with verification result including ``verified`` bool.
    """
    from .client import verify_signature

    resp = verify_signature(signature_id)
    return {
        "verified": resp.verified,
        "agent_id": resp.agent_id,
        "agent_name": resp.agent_name,
        "action_type": resp.action_type,
        "algorithm": resp.algorithm,
        "signed_at": resp.signed_at,
    }


def is_replay(nonce: str, seen_nonces: set[str]) -> bool:
    """Check if a nonce was already used. Returns True if replay detected."""
    return nonce in seen_nonces
