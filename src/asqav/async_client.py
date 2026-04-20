"""Async API Client for asqav SDK.

Provides AsyncAgent, an async mirror of the sync Agent class.
Uses httpx.AsyncClient for non-blocking HTTP calls.

Example:
    import asqav
    from asqav.async_client import AsyncAgent

    asqav.init(api_key="sk_...")

    agent = await AsyncAgent.create("my-agent")
    sig = await agent.sign("api:call", {"model": "gpt-4"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    _parse_timestamp,
)
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
    """Get current API configuration from client module.

    Returns:
        Tuple of (api_base, api_key).
    """

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
        headers={"X-API-Key": api_key},
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
        headers={"X-API-Key": api_key},
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
        headers={"X-API-Key": api_key},
        timeout=30.0,
    ) as client:
        response = await client.patch(path, json=data)
        await _handle_response(response)
        result: dict[str, Any] = response.json()
        return result


@dataclass
class AsyncAgent:
    """Async agent representation from asqav Cloud.

    Mirrors the sync Agent class but uses async HTTP calls.
    All ML-DSA cryptography happens server-side.

    Example:
        import asqav
        from asqav.async_client import AsyncAgent

        asqav.init(api_key="sk_...")

        agent = await AsyncAgent.create("my-agent")
        sig = await agent.sign("api:call", {"model": "gpt-4"})

        session = await agent.start_session()
        await agent.sign("read:data", {"file": "config.json"})
        await agent.end_session()
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
    async def create(
        cls,
        name: str,
        algorithm: str = "ml-dsa-65",
        capabilities: list[str] | None = None,
    ) -> AsyncAgent:
        """Create a new agent via asqav Cloud.

        Args:
            name: Human-readable name for the agent.
            algorithm: ML-DSA level (ml-dsa-44, ml-dsa-65, ml-dsa-87).
            capabilities: List of capabilities/permissions.

        Returns:
            An AsyncAgent instance.
        """
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
        """Get an existing agent by ID.

        Args:
            agent_id: The agent ID to retrieve.

        Returns:
            An AsyncAgent instance.
        """
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
    ) -> SignatureResponse:
        """Sign an action cryptographically (async).

        Args:
            action_type: Type of action (e.g., "read:data", "api:call").
            context: Additional context for the action.

        Returns:
            SignatureResponse with the signature.
        """
        data = await _async_post(
            f"/agents/{self.agent_id}/sign",
            {
                "action_type": action_type,
                "context": context or {},
                "session_id": self._session_id,
            },
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
            scan_result=data.get("scan_result"),
            policy_digest=data.get("policy_digest"),
            policy_decision=data.get("policy_decision", "permit"),
            authorization_ref=data.get("authorization_ref"),
        )

    async def start_session(self) -> SessionResponse:
        """Start a new session (async).

        Returns:
            SessionResponse with session details.
        """
        data = await _async_post("/sessions/", {"agent_id": self.agent_id})

        self._session_id = data["session_id"]

        return SessionResponse(
            session_id=data["session_id"],
            agent_id=data["agent_id"],
            status=data["status"],
            started_at=data["started_at"],
        )

    async def end_session(self, status: str = "completed") -> SessionResponse:
        """End the current session (async).

        Args:
            status: Final status (completed, error, timeout).

        Returns:
            SessionResponse with final details.
        """
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
        """Pre-flight check combining revocation, suspension, and policy status (async).

        Returns a single result indicating whether the agent is cleared to
        perform the given action type. Combines:
        - Agent status (not revoked, not suspended)
        - Policy check (action allowed by org policies)

        Fail-open: if any individual check errors out, the agent is not
        blocked. A warning is added to reasons instead.

        Args:
            action_type: The action to check (e.g., "data:read", "api:call").

        Returns:
            PreflightResult with combined check outcome.
        """
        agent_active = True
        policy_allowed = True
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
            reasons.append(f"status check failed ({exc}) - skipped")

        # Check organization policies.
        try:
            policies = await _async_get("/policies")
            for p in policies if isinstance(policies, list) else []:
                if not p.get("is_active"):
                    continue
                pattern = p.get("action_pattern", "")
                if pattern == "*" or action_type.startswith(pattern.rstrip("*")):
                    if p.get("action") in ("block", "block_and_alert"):
                        policy_allowed = False
                        reasons.append(f"blocked by policy: {p.get('name', 'unknown')}")
        except Exception as exc:
            reasons.append(f"policy check failed ({exc}) - skipped")

        cleared = agent_active and policy_allowed
        return PreflightResult(
            cleared=cleared,
            agent_active=agent_active,
            policy_allowed=policy_allowed,
            reasons=reasons,
        )

    async def verify(self, signature_id: str) -> VerificationResponse:
        """Publicly verify a signature by ID (async).

        Args:
            signature_id: The signature ID to verify.

        Returns:
            VerificationResponse with verification details.
        """
        _ensure_httpx()
        api_base, _ = _get_config()
        url = f"{api_base}/verify/{signature_id}"

        async with httpx.AsyncClient(timeout=30.0) as client:
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
