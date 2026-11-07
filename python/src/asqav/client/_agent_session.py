"""Agent sessions, status changes, preflight and scope tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..scope import ScopeToken

from .._sql_match import matches_pattern as _matches_pattern
from ..patterns import resolve_pattern
from . import _transport
from ._common import _parse_timestamp
from ._errors import AsqavError, require_field
from ._models import CertificateResponse, PreflightResult, SessionResponse, SignedActionResponse


class _AgentSession:
    def start_session(self) -> SessionResponse:
        """Start a new session.

        Returns:
            SessionResponse with session details.
        """
        data = _transport._post("/sessions/", {"agent_id": self.agent_id})

        self._session_id = data["session_id"]

        return SessionResponse(
            session_id=data["session_id"],
            agent_id=data["agent_id"],
            status=data["status"],
            started_at=data["started_at"],
        )

    def end_session(self, status: str = "completed") -> SessionResponse:
        """End the current session.

        Args:
            status: Final status (completed, error, timeout).

        Returns:
            SessionResponse with final details.
        """
        if not self._session_id:
            raise AsqavError("No active session")

        data = _transport._patch(
            f"/sessions/{self._session_id}",
            {"status": status},
        )

        session_id = self._session_id
        self._session_id = None

        return SessionResponse(
            session_id=session_id,
            agent_id=data["agent_id"],
            status=data["status"],
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
        )

    def get_session_signatures(self) -> list[SignedActionResponse]:
        """Get signed actions for the current session.

        Returns:
            List of SignedActionResponse with signature details.
        """
        if not self._session_id:
            raise AsqavError("No active session")

        return get_session_signatures(self._session_id)

    def revoke(self, reason: str = "manual") -> None:
        """Revoke this agent's credentials permanently.

        Revocation propagates globally across the asqav network.
        Use suspend() for temporary disable that can be reversed.

        Args:
            reason: Reason for revocation.
        """
        _transport._post(
            f"/agents/{self.agent_id}/revoke",
            {"reason": reason},
        )

    def suspend(
        self,
        reason: str = "manual",
        note: str | None = None,
    ) -> dict[str, Any]:
        """Temporarily suspend this agent.

        Suspended agents cannot sign, issue tokens, or delegate.
        Use unsuspend() to restore access. For permanent revocation,
        use revoke() instead.

        Args:
            reason: Reason for suspension (investigation, maintenance,
                    policy_violation, manual, anomaly_detected).
            note: Optional note about the suspension.

        Returns:
            Dict with suspension details.
        """
        payload: dict[str, Any] = {"reason": reason}
        if note:
            payload["note"] = note
        return _transport._post(f"/agents/{self.agent_id}/suspend", payload)

        # Remove suspension from this agent.
    def unsuspend(self) -> dict[str, Any]:
        return _transport._post(f"/agents/{self.agent_id}/unsuspend", {})

        # Permanently retire this agent (keys archived, no further activity).
    def decommission(self, reason: str = "manual") -> dict[str, Any]:
        return _transport._post(
            f"/agents/{self.agent_id}/decommission",
            {"reason": reason},
        )

        # Block signing and token issuance, keep read + verify. Enterprise.
    def quarantine(self) -> dict[str, Any]:
        return _transport._post(f"/agents/{self.agent_id}/quarantine", {})

        # Restore normal signing and token issuance after quarantine.
    def unquarantine(self) -> dict[str, Any]:
        return _transport._post(f"/agents/{self.agent_id}/unquarantine", {})

        # Check if this agent is revoked.
    @property
    def is_revoked(self) -> bool:
        data = _transport._get(f"/agents/{self.agent_id}/status")
        return bool(data.get("revoked", False))

        # Check if this agent is suspended.
    @property
    def is_suspended(self) -> bool:
        data = _transport._get(f"/agents/{self.agent_id}/status")
        return bool(data.get("suspended", False))

    def preflight(self, action_type: str) -> PreflightResult:
        """Pre-flight check combining revocation, suspension, and policy status.

        Returns a single result indicating whether the agent is cleared to
        perform the given action type. Combines:
        - Agent status (not revoked, not suspended)
        - Policy check (action allowed by org policies)

        Use before executing sensitive actions to catch issues early.

        Fail-closed on error: if a status or policy fetch errors out, the
        check could not complete, so cleared is False and checks_complete is
        False. A transient outage never silently clears a revoked or blocked
        agent. Inspect checks_complete to tell a real verdict from an
        unfinished one.

        Args:
            action_type: The action to check (e.g., "data:read", "api:call").
                Accepts semantic pattern names like "sql-read" which
                auto-expand to their glob equivalents.

        Returns:
            PreflightResult with combined check outcome.
        """
        action_type = resolve_pattern(action_type)
        agent_active = True
        policy_allowed = True
        checks_complete = True
        reasons: list[str] = []

        # Check agent status (revoked / suspended).
        try:
            status_data = _transport._get(f"/agents/{self.agent_id}/status")
            if status_data.get("revoked", False):
                agent_active = False
                reasons.append("agent is revoked")
            if status_data.get("suspended", False):
                agent_active = False
                reasons.append("agent is suspended")
        except Exception as exc:
            checks_complete = False
            reasons.append(f"status check failed ({exc}) - could not verify")

        # Check organization policies.
        try:
            policies = _transport._get("/policies")
            if not isinstance(policies, list):
                # A non-list response is anomalous, so fail closed instead of
                # silently clearing on an empty iteration.
                checks_complete = False
                reasons.append("policy check failed (unexpected response) - could not verify")
            else:
                for p in policies:
                    if not p.get("is_active"):
                        continue
                    pattern = p.get("action_pattern", "")
                    if _matches_pattern(pattern, action_type):
                        if p.get("action") in ("block", "block_and_alert"):
                            policy_allowed = False
                            reasons.append(f"blocked by policy: {p.get('name', 'unknown')}")
        except Exception as exc:
            checks_complete = False
            reasons.append(f"policy check failed ({exc}) - could not verify")

        # A failed fetch leaves the verdict unknown, so do not clear on it.
        cleared = agent_active and policy_allowed and checks_complete

        # Build human-readable explanation from check results.
        if cleared:
            explanation = "Allowed: agent is active and action is permitted by policy"
        elif not checks_complete:
            explanation = "Blocked: could not verify (" + "; ".join(reasons) + ")"
        elif not agent_active and "agent is revoked" in reasons:
            explanation = "Blocked: agent has been revoked"
        elif not agent_active and "agent is suspended" in reasons:
            suspend_reason = ""
            for r in reasons:
                if r.startswith("agent is suspended"):
                    suspend_reason = r
                    break
            explanation = f"Blocked: agent is suspended ({suspend_reason})"
        elif not policy_allowed:
            explanation = "Blocked: action not permitted by current policy rules"
        else:
            explanation = "Blocked: " + "; ".join(reasons) if reasons else "Blocked"

        return PreflightResult(
            cleared=cleared,
            agent_active=agent_active,
            policy_allowed=policy_allowed,
            reasons=reasons,
            explanation=explanation,
            checks_complete=checks_complete,
        )

    def get_certificate(self) -> CertificateResponse:
        """Get the agent's identity certificate (ML-DSA public key in PEM).
        Returns: CertificateResponse."""
        data = _transport._get(f"/agents/{self.agent_id}/certificate")

        return CertificateResponse(
            agent_id=data["agent_id"],
            agent_name=data["agent_name"],
            algorithm=data["algorithm"],
            public_key_pem=data["public_key_pem"],
            key_id=data["key_id"],
            created_at=_parse_timestamp(data["created_at"]),
            is_revoked=data["is_revoked"],
        )

    def create_scope_token(
        self,
        actions: list[str],
        ttl: int = 3600,
        metadata: dict[str, Any] | None = None,
    ) -> "ScopeToken":
        """Issue a portable scope token for cross-org verification.

        Wraps issue_sd_token() with resolved action patterns and returns
        a ScopeToken that can be attached to outbound HTTP requests.

        Args:
            actions: Action patterns (semantic names or raw globs).
            ttl: Token time-to-live in seconds.
            metadata: Optional extra claims to embed.

        Returns:
            ScopeToken ready for to_header() or present().

        Example:
            token = agent.create_scope_token(["sql-read", "http-external"])
            requests.get(url, headers=token.to_header())
        """
        from ..scope import create_scope_token as _create

        return _create(self, actions, ttl=ttl, metadata=metadata)


def get_session_signatures(session_id: str) -> list[SignedActionResponse]:
    """Get signed actions for a session.

    Args:
        session_id: The session ID.

    Returns:
        List of SignedActionResponse with signature details.
    """
    data = _transport._get(f"/sessions/{session_id}/signatures")

    return [
        SignedActionResponse(
            signature_id=require_field(sig, "signature_id"),
            agent_id=require_field(sig, "agent_id"),
            action_id=require_field(sig, "action_id"),
            action_type=require_field(sig, "action_type"),
            payload=sig.get("payload"),
            algorithm=require_field(sig, "algorithm"),
            signed_at=_parse_timestamp(require_field(sig, "signed_at")),
            signature_preview=require_field(sig, "signature_preview"),
            verification_url=require_field(sig, "verification_url"),
        )
        for sig in data
    ]
