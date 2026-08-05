"""Portable scope tokens for cross-org agent verification.

Thin wrapper around ``issue_sd_token`` adding an envelope with helpers for
HTTP attachment, expiry, and selective-disclosure presentation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .patterns import resolve_pattern

if TYPE_CHECKING:
    from .client import Agent, SDTokenResponse


    # A portable proof of agent permissions; wraps an SDTokenResponse.
@dataclass
class ScopeToken:

    token: str
    jwt: str
    disclosures: dict[str, str]
    expires_at: float
    actions: list[str]
    agent_id: str
    nonce: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- helpers --------------------------------------------------------------

        # Return an ``Authorization`` header dict; ``disclose=None`` includes all disclosures.
    def to_header(self, disclose: list[str] | None = None) -> dict[str, str]:
        if disclose is not None:
            parts = [self.jwt]
            for name in disclose:
                if name in self.disclosures:
                    parts.append(self.disclosures[name])
            value = "~".join(parts) + "~"
        else:
            value = self.token
        return {"Authorization": f"Bearer {value}"}

        # Return an SD-JWT string with only the listed disclosures revealed.
    def present(self, disclose: list[str]) -> str:
        parts = [self.jwt]
        for name in disclose:
            if name in self.disclosures:
                parts.append(self.disclosures[name])
        return "~".join(parts) + "~"

        # Check if the token has expired.
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

        # Serialize to a plain dict for storage or transport.
    def to_dict(self) -> dict[str, Any]:
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

        # Reconstruct a :class:`ScopeToken` from a dict.
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScopeToken:
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


    # Issue a portable :class:`ScopeToken` for an agent over the given action patterns.
def create_scope_token(
    agent: Agent,
    actions: list[str],
    ttl: int = 3600,
    metadata: dict[str, Any] | None = None,
) -> ScopeToken:
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


    # Verify a scope token via :func:`verify_signature`; returns a result dict.
def verify_scope_token(signature_id: str) -> dict[str, Any]:
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


    # Check if a nonce was already used. Returns True if replay detected.
def is_replay(nonce: str, seen_nonces: set[str]) -> bool:
    return nonce in seen_nonces
