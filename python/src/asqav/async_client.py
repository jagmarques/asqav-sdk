"""Async API client; :class:`AsyncAgent` mirrors :class:`Agent` over ``httpx.AsyncClient``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._sql_match import matches_pattern as _matches_pattern
from ._useragent import USER_AGENT
from .client import (
    APIError,
    AsqavError,
    AuthenticationError,
    PreflightResult,
    RateLimitError,
    SessionResponse,
    SignatureResponse,
    VerificationResponse,
    _api_base,
    _api_key,
    _parse_bitcoin_anchor,
    _parse_timestamp,
    _resolve_action_ref,
    _resolve_previous_receipt_hash,
)
from .patterns import resolve_pattern
from .retry import with_async_retry

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore


def _ensure_httpx() -> None:
    """Ensure httpx is available for async operations."""
    if not _HTTPX_AVAILABLE:
        raise AsqavError(
            "httpx is required for async operations. Install with: pip install asqav[httpx]"
        )


def _ensure_initialized() -> None:
    """Ensure the SDK is initialized."""

    if not _api_key:
        raise AuthenticationError("Call asqav.init() first. Get your API key at asqav.com")


def _get_config() -> tuple[str, str]:
    """Get the current ``(api_base, api_key)`` tuple from the client module."""

    if not _api_key:
        raise AuthenticationError("Call asqav.init() first. Get your API key at asqav.com")
    return _api_base, _api_key


async def _handle_response(response: Any) -> None:
    """Handle API response errors (async version)."""
    if response.status_code == 401:
        raise AuthenticationError("Invalid API key")
    elif response.status_code == 429:
        raise RateLimitError("Rate limit exceeded. Upgrade at asqav.com/pricing")
    elif response.status_code >= 400:
        try:
            error = response.json().get("error", "Unknown error")
        except Exception:
            error = response.text
        raise APIError(error, response.status_code)


@with_async_retry()
async def _async_get(path: str) -> dict[str, Any]:
    """Make an async GET request to the API."""
    _ensure_httpx()
    _ensure_initialized()
    api_base, api_key = _get_config()

    async with httpx.AsyncClient(
        base_url=api_base,
        headers={"X-API-Key": api_key, "User-Agent": USER_AGENT},
        timeout=30.0,
    ) as client:
        response = await client.get(path)
        await _handle_response(response)
        result: dict[str, Any] = response.json()
        return result


@with_async_retry()
async def _async_post(path: str, data: dict[str, Any]) -> dict[str, Any]:
    """Make an async POST request to the API."""
    _ensure_httpx()
    _ensure_initialized()
    api_base, api_key = _get_config()

    async with httpx.AsyncClient(
        base_url=api_base,
        headers={"X-API-Key": api_key, "User-Agent": USER_AGENT},
        timeout=30.0,
    ) as client:
        response = await client.post(path, json=data)
        await _handle_response(response)
        result: dict[str, Any] = response.json()
        return result


@with_async_retry()
async def _async_patch(path: str, data: dict[str, Any]) -> dict[str, Any]:
    """Make an async PATCH request to the API."""
    _ensure_httpx()
    _ensure_initialized()
    api_base, api_key = _get_config()

    async with httpx.AsyncClient(
        base_url=api_base,
        headers={"X-API-Key": api_key, "User-Agent": USER_AGENT},
        timeout=30.0,
    ) as client:
        response = await client.patch(path, json=data)
        await _handle_response(response)
        result: dict[str, Any] = response.json()
        return result


@dataclass
class AsyncAgent:
    """Async agent representation; mirrors the sync :class:`Agent` over async HTTP."""

    agent_id: str
    name: str
    public_key: str
    key_id: str
    algorithm: str
    capabilities: list[str]
    created_at: float
    _session_id: str | None = field(default=None, repr=False)

    @classmethod
    async def create(
        cls,
        name: str,
        algorithm: str = "ml-dsa-65",
        capabilities: list[str] | None = None,
    ) -> AsyncAgent:
        """Create a new agent via Asqav Cloud; ``algorithm`` is ml-dsa-44/65/87."""
        data = await _async_post(
            "/agents/create",
            {
                "name": name,
                "algorithm": algorithm,
                "capabilities": capabilities or [],
            },
        )

        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            public_key=data["public_key"],
            key_id=data["key_id"],
            algorithm=data["algorithm"],
            capabilities=data["capabilities"],
            created_at=_parse_timestamp(data["created_at"]),
        )

    @classmethod
    async def get(cls, agent_id: str) -> AsyncAgent:
        """Get an existing agent by ID."""
        data = await _async_get(f"/agents/{agent_id}")

        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            public_key=data["public_key"],
            key_id=data["key_id"],
            algorithm=data["algorithm"],
            capabilities=data["capabilities"],
            created_at=_parse_timestamp(data["created_at"]),
        )

    async def sign(
        self,
        action_type: str,
        context: dict[str, Any] | None = None,
        tool_name: str | None = None,
        model_name: str | None = None,
        parent_id: str | None = None,
        *,
        co_signers: list[str] | None = None,
        user_intent: dict[str, Any] | None = None,
    ) -> SignatureResponse:
        """Sign an action cryptographically (async); ``co_signers`` lists countersigner agents."""
        from .client import _build_sign_body

        if tool_name or model_name or parent_id:
            context = dict(context) if context else {}
            if tool_name:
                context["_tool_name"] = tool_name
            if model_name:
                context["_model_name"] = model_name
            if parent_id:
                context["_parent_id"] = parent_id

        body = _build_sign_body(
            action_type=action_type,
            context=context,
            session_id=self._session_id,
            agent_id=self.agent_id,
            user_intent=user_intent,
        )
        if co_signers:
            body["co_signers"] = list(co_signers)
        data = await _async_post(f"/agents/{self.agent_id}/sign", body)

        return SignatureResponse(
            signature=data["signature"],
            signature_id=data["signature_id"],
            action_id=data["action_id"],
            timestamp=data["timestamp"],
            verification_url=data["verification_url"],
            algorithm=data.get("algorithm"),
            chain_hash=data.get("chain_hash") or data.get("record_hash"),
            rfc3161_tsa=(data.get("rfc3161_timestamp") or {}).get("tsa"),
            rfc3161_serial=(data.get("rfc3161_timestamp") or {}).get("serial_number"),
            scan_result=data.get("scan_result"),
            policy_digest=data.get("policy_digest"),
            policy_decision=data.get("policy_decision", "permit"),
            authorization_ref=data.get("authorization_ref"),
            bitcoin_anchor=_parse_bitcoin_anchor(data.get("bitcoin_anchor")),
            required_co_signers=data.get("required_co_signers"),
            co_signatures=data.get("co_signatures"),
            countersign_url=data.get("countersign_url"),
            user_intent_verified=data.get("user_intent_verified"),
            action_ref=_resolve_action_ref(data),
            previous_receipt_hash=_resolve_previous_receipt_hash(data),
        )

    async def countersign(self, signature_id: str) -> SignatureResponse:
        """Countersign an existing signature record (async); appends to ``co_signatures``."""
        data = await _async_post(
            f"/agents/{self.agent_id}/countersign/{signature_id}",
            {},
        )
        return SignatureResponse(
            signature=data["signature"],
            signature_id=data["signature_id"],
            action_id=data["action_id"],
            timestamp=data["timestamp"],
            verification_url=data["verification_url"],
            algorithm=data.get("algorithm"),
            chain_hash=data.get("chain_hash") or data.get("record_hash"),
            rfc3161_tsa=(data.get("rfc3161_timestamp") or {}).get("tsa"),
            rfc3161_serial=(data.get("rfc3161_timestamp") or {}).get("serial_number"),
            policy_digest=data.get("policy_digest"),
            policy_decision=data.get("policy_decision", "permit"),
            authorization_ref=data.get("authorization_ref"),
            bitcoin_anchor=_parse_bitcoin_anchor(data.get("bitcoin_anchor")),
            required_co_signers=data.get("required_co_signers"),
            co_signatures=data.get("co_signatures"),
            countersign_url=data.get("countersign_url"),
            user_intent_verified=data.get("user_intent_verified"),
        )

    async def start_session(self) -> SessionResponse:
        """Start a new session (async)."""
        data = await _async_post("/sessions/", {"agent_id": self.agent_id})

        self._session_id = data["session_id"]

        return SessionResponse(
            session_id=data["session_id"],
            agent_id=data["agent_id"],
            status=data["status"],
            started_at=data["started_at"],
        )

    async def end_session(self, status: str = "completed") -> SessionResponse:
        """End the current session (async); ``status`` is completed/error/timeout."""
        if not self._session_id:
            raise AsqavError("No active session")

        data = await _async_patch(
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

    async def preflight(self, action_type: str) -> PreflightResult:
        """Pre-flight check (async): revocation + suspension + policy status.

        Fail-closed on error: if a status or policy fetch errors out, the
        check could not complete, so cleared is False and checks_complete is
        False. A transient outage never silently clears a revoked or blocked
        agent. Inspect checks_complete to tell a real verdict from an
        unfinished one. Mirrors the sync Agent.preflight semantics.
        """
        action_type = resolve_pattern(action_type)
        agent_active = True
        policy_allowed = True
        checks_complete = True
        reasons: list[str] = []

        # Check agent status (revoked / suspended).
        try:
            status_data = await _async_get(f"/agents/{self.agent_id}/status")
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
            policies = await _async_get("/policies")
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

    async def verify(self, signature_id: str) -> VerificationResponse:
        """Publicly verify a signature by ID (async)."""
        _ensure_httpx()
        api_base, _ = _get_config()
        url = f"{api_base}/verify/{signature_id}"

        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
            response = await client.get(url)
            if response.status_code == 404:
                raise APIError("Signature not found", 404)
            if response.status_code >= 400:
                raise APIError(response.text, response.status_code)
            data: dict[str, Any] = response.json()

        return VerificationResponse(
            signature_id=data["signature_id"],
            agent_id=data["agent_id"],
            agent_name=data["agent_name"],
            action_id=data["action_id"],
            action_type=data["action_type"],
            payload=data.get("payload"),
            signature=data["signature"],
            algorithm=data["algorithm"],
            signed_at=_parse_timestamp(data["signed_at"]),
            verified=data["verified"],
            verification_url=data["verification_url"],
        )
