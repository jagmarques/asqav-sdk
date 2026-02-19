"""
asqav API Client - Thin SDK that connects to asqav.com.

This module provides the API client for connecting to asqav Cloud.
All ML-DSA cryptography happens server-side; this SDK handles
API communication and response parsing.

Example:
    import asqav

    # Initialize with API key
    asqav.init(api_key="sk_...")

    # Create an agent (server generates identity)
    agent = asqav.Agent.create("my-agent")

    # Sign an action (server signs with ML-DSA)
    signature = agent.sign("read:data", {"file": "config.json"})
"""

from __future__ import annotations

import functools
import os
import sys
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar
from urllib.parse import urljoin

F = TypeVar("F", bound=Callable[..., Any])


def _parse_timestamp(value: Any) -> float:
    """Parse a timestamp from API response (ISO string or float)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        from datetime import datetime

        # Parse ISO format datetime string
        try:
            # Handle both with and without timezone
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            dt = datetime.fromisoformat(value)
            return dt.timestamp()
        except ValueError:
            return 0.0
    return 0.0


try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore

# API configuration
_api_key: str | None = None
_api_base: str = os.environ.get("ASQAV_API_URL", "https://api.asqav.com/api/v1")
_client: Any = None


class AsqavError(Exception):
    """Base exception for asqav errors."""

    pass


class AuthenticationError(AsqavError):
    """Raised when API key is missing or invalid."""

    pass


class RateLimitError(AsqavError):
    """Raised when rate limit is exceeded."""

    pass


class APIError(AsqavError):
    """Raised for general API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class AgentResponse:
    """Response from agent creation."""

    agent_id: str
    name: str
    public_key: str
    key_id: str
    algorithm: str
    capabilities: list[str]
    created_at: float


@dataclass
class TokenResponse:
    """Response from token issuance."""

    token: str
    expires_at: float
    algorithm: str


@dataclass
class SignatureResponse:
    """Response from signing operation."""

    signature: str
    signature_id: str
    action_id: str
    timestamp: float
    verification_url: str


@dataclass
class SessionResponse:
    """Response from session operations."""

    session_id: str
    agent_id: str
    status: str
    started_at: str  # ISO datetime string
    ended_at: str | None = None


@dataclass
class SDTokenResponse:
    """Response from SD-JWT token issuance (Business tier).

    SD-JWT tokens allow selective disclosure of claims to external services.
    Use present() to create a proof with only specific claims revealed.
    """

    token: str  # Full SD-JWT with all disclosures
    jwt: str  # Just the signed JWT part
    disclosures: dict[str, str]  # claim_name -> encoded disclosure
    expires_at: float

    def present(self, disclose: list[str]) -> str:
        """Create a presentation with only specified claims disclosed.

        Args:
            disclose: List of claim names to reveal.

        Returns:
            SD-JWT string with only specified disclosures.

        Example:
            # Full token has: tier, org, capabilities
            # Present only tier to partner:
            proof = sd_token.present(["tier"])
        """
        parts = [self.jwt]
        for claim_name in disclose:
            if claim_name in self.disclosures:
                parts.append(self.disclosures[claim_name])
        return "~".join(parts) + "~"

    def full(self) -> str:
        """Return full SD-JWT with all disclosures."""
        return self.token


@dataclass
class DelegationResponse:
    """Response from agent delegation."""

    delegation_id: str
    parent_id: str
    child_id: str
    child_name: str
    scope: list[str]
    expires_at: float
    created_at: float


@dataclass
class CertificateResponse:
    """Agent identity certificate."""

    agent_id: str
    agent_name: str
    algorithm: str
    public_key_pem: str
    key_id: str
    created_at: float
    is_revoked: bool


@dataclass
class VerificationResponse:
    """Public verification response."""

    signature_id: str
    agent_id: str
    agent_name: str
    action_id: str
    action_type: str
    payload: dict[str, Any] | None
    signature: str
    algorithm: str
    signed_at: float
    verified: bool
    verification_url: str


@dataclass
class SignedActionResponse:
    """Signed action from a session."""

    signature_id: str
    agent_id: str
    action_id: str
    action_type: str
    payload: dict[str, Any] | None
    algorithm: str
    signed_at: float
    signature_preview: str
    verification_url: str


@dataclass
class ThresholdSignatureDetail:
    """Details of a single entity signature within a threshold session."""

    entity_id: str
    entity_name: str
    entity_class: str
    signed_at: str


@dataclass
class ThresholdSessionResponse:
    """Response from threshold signing session operations."""

    session_id: str
    config_id: str
    agent_id: str
    action_type: str
    action_params: dict[str, Any] | None
    threshold_required: int
    signatures_collected: int
    status: str
    signatures: list[ThresholdSignatureDetail]
    policy_attestation_hash: str | None
    created_at: str
    expires_at: str
    resolved_at: str | None = None


@dataclass
class ThresholdApproveResponse:
    """Response from threshold approval operation."""

    session_id: str
    entity_id: str
    signatures_collected: int
    threshold_required: int
    status: str
    threshold_met: bool


@dataclass
class ThresholdConfigResponse:
    """Response from threshold config operations."""

    id: str
    agent_id: str
    threshold: int
    total_shares: int
    is_active: bool
    created_at: str
    updated_at: str | None = None


@dataclass
class SigningEntityResponse:
    """Response from signing entity operations."""

    id: str
    config_id: str
    entity_class: str
    label: str
    is_active: bool
    created_at: str


@dataclass
class ThresholdKeypairResponse:
    """Response from threshold keypair operations."""

    id: str
    config_id: str
    public_key_hex: str
    threshold: int
    total_shares: int
    created_at: str
    status: str = "active"


@dataclass
class ThresholdSignResponse:
    """Response from threshold signing operation."""

    signature_hex: str
    message_hex: str
    keypair_id: str
    verified: bool
    audit_record_id: str | None = None


@dataclass
class RiskRuleResponse:
    """Response from risk rule operations."""

    id: str
    name: str
    action_pattern: str
    risk_level: str
    threshold_override: int | None
    priority: int
    entity_weights: dict[str, float] | None = None
    time_schedule: dict[str, Any] | None = None
    created_at: str = ""


@dataclass
class ThresholdDelegationResponse:
    """Response from threshold delegation operations."""

    id: str
    config_id: str
    delegator_entity_id: str
    delegate_entity_id: str
    expires_at: str
    is_active: bool
    created_at: str


@dataclass
class KeyRefreshResponse:
    """Response from key share refresh operation."""

    keypair_id: str
    refreshed_at: str
    delegations_invalidated: int


@dataclass
class ShareRecoveryResponse:
    """Response from key share recovery operation."""

    keypair_id: str
    recovered_entity_id: str
    recovered_at: str


@dataclass
class Span:
    """A single traced operation.

    Spans track the duration and context of operations. When ended,
    they are signed server-side with ML-DSA for cryptographic proof.
    """

    span_id: str
    name: str
    start_time: float
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    end_time: float | None = None
    status: str = "ok"
    signature: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Add an attribute to the span."""
        self.attributes[key] = value

    def set_status(self, status: str) -> None:
        """Set span status (ok, error)."""
        self.status = status


# Global tracer state
_current_span: Span | None = None
_span_stack: list[Span] = []
_completed_spans: list[Span] = []
_otel_endpoint: str | None = None


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Span, None, None]:
    """Create a traced span with automatic signing.

    Example:
        with asqav.span("api:openai", {"model": "gpt-4"}) as s:
            response = openai.chat.completions.create(...)
            s.set_attribute("tokens", response.usage.total_tokens)
    """
    global _current_span, _span_stack

    span_obj = Span(
        span_id=str(uuid.uuid4()),
        name=name,
        start_time=time.time(),
        attributes=attributes or {},
        parent_id=_current_span.span_id if _current_span else None,
    )

    _span_stack.append(span_obj)
    _current_span = span_obj

    try:
        yield span_obj
        span_obj.status = "ok"
    except Exception as e:
        span_obj.status = "error"
        span_obj.set_attribute("error.message", str(e))
        span_obj.set_attribute("error.type", type(e).__name__)
        raise
    finally:
        span_obj.end_time = time.time()
        _span_stack.pop()
        _current_span = _span_stack[-1] if _span_stack else None

        # Sign the span via API
        try:
            agent = get_agent()
            result = agent.sign(
                action_type=f"span:{name}",
                context={
                    "span_id": span_obj.span_id,
                    "parent_id": span_obj.parent_id,
                    "start_time": span_obj.start_time,
                    "end_time": span_obj.end_time,
                    "duration_ms": (span_obj.end_time - span_obj.start_time) * 1000,
                    "status": span_obj.status,
                    "attributes": span_obj.attributes,
                },
            )
            span_obj.signature = result.signature
        except Exception:
            pass  # Don't fail user code if signing fails

        # Add to completed spans for OTEL export
        _completed_spans.append(span_obj)


def get_current_span() -> Span | None:
    """Get the currently active span, if any."""
    return _current_span


def configure_otel(endpoint: str | None = None) -> None:
    """Configure OpenTelemetry export.

    Args:
        endpoint: OTEL collector endpoint (e.g., "http://localhost:4318/v1/traces").
                  Set to None to disable export.

    Example:
        asqav.configure_otel("http://localhost:4318/v1/traces")
    """
    global _otel_endpoint
    _otel_endpoint = endpoint


def span_to_otel(s: Span) -> dict[str, Any]:
    """Convert a Span to OTEL format."""
    return {
        "traceId": s.span_id.replace("-", "")[:32].ljust(32, "0"),
        "spanId": s.span_id.replace("-", "")[:16],
        "parentSpanId": s.parent_id.replace("-", "")[:16] if s.parent_id else None,
        "name": s.name,
        "kind": 1,  # INTERNAL
        "startTimeUnixNano": int(s.start_time * 1_000_000_000),
        "endTimeUnixNano": int((s.end_time or s.start_time) * 1_000_000_000),
        "attributes": [
            {"key": k, "value": {"stringValue": str(v)}} for k, v in s.attributes.items()
        ]
        + (
            [{"key": "asqav.signature", "value": {"stringValue": s.signature}}]
            if s.signature
            else []
        ),
        "status": {"code": 1 if s.status == "ok" else 2},
    }


def export_spans() -> list[dict[str, Any]]:
    """Export completed spans in OTEL format.

    Returns:
        List of spans in OTEL format.
    """
    global _completed_spans
    spans = [span_to_otel(s) for s in _completed_spans]
    _completed_spans = []
    return spans


def flush_spans() -> None:
    """Flush spans to configured OTEL endpoint."""
    global _otel_endpoint, _completed_spans

    if not _otel_endpoint or not _completed_spans:
        return

    spans = export_spans()
    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "asqav"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "asqav", "version": "0.2.6"},
                        "spans": spans,
                    }
                ],
            }
        ]
    }

    try:
        import json
        import urllib.request

        req = urllib.request.Request(
            _otel_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # Don't fail on export errors


@dataclass
class Agent:
    """Agent representation from asqav Cloud.

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
        """Create a new agent via asqav Cloud.

        The server generates the ML-DSA keypair. The private key
        never leaves the server.

        Args:
            name: Human-readable name for the agent.
            algorithm: ML-DSA level (ml-dsa-44, ml-dsa-65, ml-dsa-87).
            capabilities: List of capabilities/permissions.

        Returns:
            An Agent instance.

        Raises:
            AuthenticationError: If API key is missing or invalid.
            APIError: If the request fails.
        """
        data = _post(
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
    def get(cls, agent_id: str) -> Agent:
        """Get an existing agent by ID.

        Args:
            agent_id: The agent ID to retrieve.

        Returns:
            An Agent instance.
        """
        data = _get(f"/agents/{agent_id}")

        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            public_key=data["public_key"],
            key_id=data["key_id"],
            algorithm=data["algorithm"],
            capabilities=data["capabilities"],
            created_at=_parse_timestamp(data["created_at"]),
        )

    def issue_token(
        self,
        scope: list[str] | None = None,
        ttl: int = 3600,
    ) -> TokenResponse:
        """Issue a PQC-JWT token for this agent.

        The token is signed server-side with ML-DSA.

        Args:
            scope: Capabilities to include (default: all).
            ttl: Token time-to-live in seconds.

        Returns:
            TokenResponse with the signed token.
        """
        data = _post(
            f"/agents/{self.agent_id}/tokens",
            {
                "scope": scope or self.capabilities,
                "ttl": ttl,
            },
        )

        return TokenResponse(
            token=data["token"],
            expires_at=_parse_timestamp(data["expires_at"]),
            algorithm=data["algorithm"],
        )

    def issue_sd_token(
        self,
        claims: dict[str, Any] | None = None,
        disclosable: list[str] | None = None,
        ttl: int = 3600,
    ) -> SDTokenResponse:
        """Issue a PQC-SD-JWT token with selective disclosure (Business tier).

        SD-JWT tokens allow agents to selectively reveal claims when
        presenting the token to external services, maintaining privacy.

        Args:
            claims: Claims to include in the token.
            disclosable: List of claim names that can be selectively disclosed.
            ttl: Token time-to-live in seconds.

        Returns:
            SDTokenResponse with the token and disclosures.

        Example:
            sd_token = agent.issue_sd_token(
                claims={"tier": "pro", "org": "acme"},
                disclosable=["tier", "org"]
            )

            # Present to partner - only show tier
            proof = sd_token.present(["tier"])
        """
        data = _post(
            f"/agents/{self.agent_id}/tokens/sd",
            {
                "claims": claims or {},
                "disclosable": disclosable or [],
                "ttl": ttl,
            },
        )

        return SDTokenResponse(
            token=data["token"],
            jwt=data["jwt"],
            disclosures=data["disclosures"],
            expires_at=_parse_timestamp(data["expires_at"]),
        )

    def sign(
        self,
        action_type: str,
        context: dict[str, Any] | None = None,
    ) -> SignatureResponse:
        """Sign an action cryptographically.

        The signature is created server-side with ML-DSA.

        Args:
            action_type: Type of action (e.g., "read:data", "api:call").
            context: Additional context for the action.

        Returns:
            SignatureResponse with the signature.
        """
        data = _post(
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
        )

    def start_session(self) -> SessionResponse:
        """Start a new session.

        Returns:
            SessionResponse with session details.
        """
        data = _post("/sessions/", {"agent_id": self.agent_id})

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

        data = _patch(
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
        _post(
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
        return _post(f"/agents/{self.agent_id}/suspend", payload)

    def unsuspend(self) -> dict[str, Any]:
        """Remove suspension from this agent.

        Restores the agent to active status.

        Returns:
            Dict with updated agent status.
        """
        return _post(f"/agents/{self.agent_id}/unsuspend", {})

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
        data = _post(
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

    @property
    def is_revoked(self) -> bool:
        """Check if this agent is revoked."""
        data = _get(f"/agents/{self.agent_id}/status")
        return bool(data.get("revoked", False))

    @property
    def is_suspended(self) -> bool:
        """Check if this agent is suspended."""
        data = _get(f"/agents/{self.agent_id}/status")
        return bool(data.get("suspended", False))

    def get_certificate(self) -> CertificateResponse:
        """Get the agent's identity certificate.

        The certificate contains the ML-DSA public key in PEM format
        for independent verification.

        Returns:
            CertificateResponse with certificate details.
        """
        data = _get(f"/agents/{self.agent_id}/certificate")

        return CertificateResponse(
            agent_id=data["agent_id"],
            agent_name=data["agent_name"],
            algorithm=data["algorithm"],
            public_key_pem=data["public_key_pem"],
            key_id=data["key_id"],
            created_at=_parse_timestamp(data["created_at"]),
            is_revoked=data["is_revoked"],
        )


def init(
    api_key: str | None = None,
    base_url: str | None = None,
) -> None:
    """Initialize the asqav SDK.

    Args:
        api_key: Your asqav API key. Can also be set via ASQAV_API_KEY env var.
        base_url: Override API base URL (for testing).

    Raises:
        AuthenticationError: If no API key is provided.

    Example:
        import asqav

        # Using parameter
        asqav.init(api_key="sk_...")

        # Using environment variable
        os.environ["ASQAV_API_KEY"] = "sk_..."
        asqav.init()
    """
    global _api_key, _api_base, _client

    _api_key = api_key or os.environ.get("ASQAV_API_KEY")

    if not _api_key:
        raise AuthenticationError(
            "API key required. Set ASQAV_API_KEY or pass api_key to init(). Get yours at asqav.com"
        )

    if base_url:
        _api_base = base_url

    # Initialize HTTP client
    if _HTTPX_AVAILABLE:
        _client = httpx.Client(
            base_url=_api_base,
            headers={"X-API-Key": _api_key},
            timeout=30.0,
        )


def _get(path: str) -> dict[str, Any]:
    """Make a GET request to the API."""
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        response = _client.get(path)
        _handle_response(response)
        result: dict[str, Any] = response.json()
        return result
    else:
        # Use stdlib urllib if httpx not installed
        return _urllib_request("GET", path)


def _post(path: str, data: dict[str, Any]) -> dict[str, Any]:
    """Make a POST request to the API."""
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        response = _client.post(path, json=data)
        _handle_response(response)
        result: dict[str, Any] = response.json()
        return result
    else:
        return _urllib_request("POST", path, data)


def _patch(path: str, data: dict[str, Any]) -> dict[str, Any]:
    """Make a PATCH request to the API."""
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        response = _client.patch(path, json=data)
        _handle_response(response)
        result: dict[str, Any] = response.json()
        return result
    else:
        return _urllib_request("PATCH", path, data)


def _put(path: str, data: dict[str, Any]) -> dict[str, Any]:
    """Make a PUT request to the API."""
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        response = _client.put(path, json=data)
        _handle_response(response)
        result: dict[str, Any] = response.json()
        return result
    else:
        return _urllib_request("PUT", path, data)


def _delete(path: str) -> dict[str, Any]:
    """Make a DELETE request to the API."""
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        response = _client.delete(path)
        _handle_response(response)
        result: dict[str, Any] = response.json()
        return result
    else:
        return _urllib_request("DELETE", path)


def _ensure_initialized() -> None:
    """Ensure the SDK is initialized."""
    if not _api_key:
        raise AuthenticationError("Call asqav.init() first. Get your API key at asqav.com")


def _handle_response(response: Any) -> None:
    """Handle API response errors."""
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


def _urllib_request(
    method: str,
    path: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """HTTP client using stdlib urllib (used when httpx not installed)."""
    import json
    import urllib.error
    import urllib.request

    url = urljoin(_api_base, path)
    headers = {
        "X-API-Key": _api_key,
        "Content-Type": "application/json",
    }

    body = json.dumps(data).encode("utf-8") if data else None

    request = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return result
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise AuthenticationError("Invalid API key") from e
        elif e.code == 429:
            raise RateLimitError("Rate limit exceeded") from e
        else:
            raise APIError(str(e), e.code) from e


# Global agent for decorators
_global_agent: Agent | None = None


def get_agent() -> Agent:
    """Get the global agent for decorator use.

    Returns:
        The global Agent instance.

    Raises:
        AsqavError: If no agent is configured.
    """
    global _global_agent
    if _global_agent is None:
        # Auto-create an agent with default name
        name = _auto_generate_name()
        _global_agent = Agent.create(name)
    return _global_agent


def _auto_generate_name() -> str:
    """Generate an agent name from environment."""
    env_name = os.environ.get("ASQAV_AGENT_NAME")
    if env_name:
        return env_name

    if hasattr(sys, "argv") and sys.argv:
        import pathlib

        script_path = pathlib.Path(sys.argv[0])
        return f"agent-{script_path.stem}"

    return "asqav-agent"


def secure(func: F) -> F:
    """Decorator to secure function calls with cryptographic signing.

    Wraps the function in an asqav session, signing the call as an action.
    All ML-DSA cryptography happens server-side.

    Args:
        func: The function to secure.

    Returns:
        The wrapped function.

    Example:
        import asqav

        asqav.init(api_key="sk_...")

        @asqav.secure
        def process_data(data: dict) -> dict:
            # This call is signed with cryptographic proof
            return {"processed": True}
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        agent = get_agent()

        # Start session if not active
        if agent._session_id is None:
            agent.start_session()

        # Log the function call
        agent.sign(
            action_type="function:call",
            context={
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            },
        )

        try:
            result = func(*args, **kwargs)

            # Log success
            agent.sign(
                action_type="function:result",
                context={
                    "function": func.__name__,
                    "success": True,
                },
            )

            return result

        except Exception as e:
            # Log error
            agent.sign(
                action_type="function:error",
                context={
                    "function": func.__name__,
                    "error": str(e),
                },
            )
            raise

    return wrapper  # type: ignore


def secure_async(func: F) -> F:
    """Async version of the @secure decorator.

    Args:
        func: The async function to secure.

    Returns:
        The wrapped async function.

    Example:
        import asqav

        asqav.init(api_key="sk_...")

        @asqav.secure_async
        async def fetch_data(url: str) -> dict:
            # This call is signed with cryptographic proof
            return {"data": "..."}
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        agent = get_agent()

        if agent._session_id is None:
            agent.start_session()

        agent.sign(
            action_type="function:call",
            context={
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            },
        )

        try:
            result = await func(*args, **kwargs)

            agent.sign(
                action_type="function:result",
                context={
                    "function": func.__name__,
                    "success": True,
                },
            )

            return result

        except Exception as e:
            agent.sign(
                action_type="function:error",
                context={
                    "function": func.__name__,
                    "error": str(e),
                },
            )
            raise

    return wrapper  # type: ignore


def get_session_signatures(session_id: str) -> list[SignedActionResponse]:
    """Get signed actions for a session.

    Args:
        session_id: The session ID.

    Returns:
        List of SignedActionResponse with signature details.
    """
    data = _get(f"/sessions/{session_id}/signatures")

    return [
        SignedActionResponse(
            signature_id=sig["signature_id"],
            agent_id=sig["agent_id"],
            action_id=sig["action_id"],
            action_type=sig["action_type"],
            payload=sig.get("payload"),
            algorithm=sig["algorithm"],
            signed_at=_parse_timestamp(sig["signed_at"]),
            signature_preview=sig["signature_preview"],
            verification_url=sig["verification_url"],
        )
        for sig in data
    ]


def verify_signature(signature_id: str) -> VerificationResponse:
    """Publicly verify a signature by ID.

    This endpoint requires no authentication. Anyone with the signature_id
    can verify that the signature is valid and was created by the agent.

    Args:
        signature_id: The signature ID to verify.

    Returns:
        VerificationResponse with verification details.

    Example:
        result = asqav.verify_signature("sig_abc123")
        if result.verified:
            print(f"Signature valid for agent {result.agent_name}")
    """
    url = f"{_api_base}/verify/{signature_id}"

    # Use httpx if available (better SSL handling)
    if _HTTPX_AVAILABLE:
        response = httpx.get(url, timeout=30.0)
        if response.status_code == 404:
            raise APIError("Signature not found", 404)
        if response.status_code >= 400:
            raise APIError(response.text, response.status_code)
        data: dict[str, Any] = response.json()
    else:
        # Fallback to urllib
        import json
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise APIError("Signature not found", 404) from e
            raise APIError(str(e), e.code) from e

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


def request_action(
    agent_id: str,
    action_type: str,
    params: dict[str, Any] | None = None,
) -> ThresholdSessionResponse:
    """Request a threshold-authorized action for an agent.

    Creates a signing session that requires multiple entity approvals
    before the action is authorized (multi-key authorization).

    Args:
        agent_id: The agent requesting the action.
        action_type: Type of action (e.g., "deploy:production").
        params: Optional parameters for the action.

    Returns:
        ThresholdSessionResponse with session details and approval status.

    Raises:
        AuthenticationError: If API key is missing or invalid.
        APIError: If the request fails (e.g., no threshold config for agent).

    Example:
        session = asqav.request_action(
            agent_id="agent_abc",
            action_type="deploy:production",
            params={"target": "us-east-1"},
        )
        print(f"Session {session.session_id}: needs {session.threshold_required} approvals")
    """
    body: dict[str, Any] = {
        "agent_id": agent_id,
        "action_type": action_type,
    }
    if params is not None:
        body["params"] = params

    data = _post("/threshold/sessions", body)
    return _parse_threshold_session(data)


def approve_action(
    session_id: str,
    entity_id: str,
) -> ThresholdApproveResponse:
    """Approve a pending threshold action session.

    Adds an entity signature to the session. When enough entities
    have approved (meeting the threshold), the action is authorized.

    Args:
        session_id: The threshold session to approve.
        entity_id: The signing entity providing approval.

    Returns:
        ThresholdApproveResponse with approval status and threshold progress.

    Raises:
        AuthenticationError: If API key is missing or invalid.
        APIError: If the request fails (e.g., session expired, entity already signed).

    Example:
        result = asqav.approve_action(
            session_id="thr_abc123",
            entity_id="ent_xyz789",
        )
        if result.threshold_met:
            print("Action authorized!")
    """
    data = _post(
        f"/threshold/sessions/{session_id}/approve",
        {"entity_id": entity_id},
    )
    return ThresholdApproveResponse(
        session_id=data["session_id"],
        entity_id=data["entity_id"],
        signatures_collected=data["signatures_collected"],
        threshold_required=data["threshold_required"],
        status=data["status"],
        threshold_met=data["threshold_met"],
    )


def get_action_status(session_id: str) -> ThresholdSessionResponse:
    """Get the current status of a threshold action session.

    Args:
        session_id: The threshold session to check.

    Returns:
        ThresholdSessionResponse with full session details.

    Raises:
        AuthenticationError: If API key is missing or invalid.
        APIError: If the request fails (e.g., session not found).

    Example:
        status = asqav.get_action_status("thr_abc123")
        print(f"{status.signatures_collected}/{status.threshold_required} approvals")
    """
    data = _get(f"/threshold/sessions/{session_id}")
    return _parse_threshold_session(data)


def _parse_threshold_session(data: dict[str, Any]) -> ThresholdSessionResponse:
    """Parse API response into ThresholdSessionResponse."""
    signatures = [
        ThresholdSignatureDetail(
            entity_id=s["entity_id"],
            entity_name=s["entity_name"],
            entity_class=s["entity_class"],
            signed_at=s["signed_at"],
        )
        for s in data.get("signatures", [])
    ]
    return ThresholdSessionResponse(
        session_id=data["session_id"],
        config_id=data["config_id"],
        agent_id=data["agent_id"],
        action_type=data["action_type"],
        action_params=data.get("action_params"),
        threshold_required=data["threshold_required"],
        signatures_collected=data["signatures_collected"],
        status=data["status"],
        signatures=signatures,
        policy_attestation_hash=data.get("policy_attestation_hash"),
        created_at=data["created_at"],
        expires_at=data["expires_at"],
        resolved_at=data.get("resolved_at"),
    )


# --- Threshold Configuration ---

def create_threshold_config(
    agent_id: str,
    threshold: int,
    total_shares: int,
) -> ThresholdConfigResponse:
    """Create a threshold config for an agent.

    Args:
        agent_id: The agent to configure threshold signing for.
        threshold: Minimum signatures required (t in t-of-n).
        total_shares: Total key shares to generate (n in t-of-n).

    Returns:
        ThresholdConfigResponse with config details.

    Example:
        config = asqav.create_threshold_config("agt_xxx", threshold=2, total_shares=3)
    """
    data = _post("/threshold/configs", {
        "agent_id": agent_id,
        "threshold": threshold,
        "total_shares": total_shares,
    })
    return _parse_threshold_config(data)


def get_threshold_config(agent_id: str) -> ThresholdConfigResponse:
    """Get the active threshold config for an agent.

    Args:
        agent_id: The agent ID.

    Returns:
        ThresholdConfigResponse with config details.
    """
    data = _get(f"/threshold/configs/{agent_id}")
    return _parse_threshold_config(data)


def update_threshold_config(
    config_id: str,
    threshold: int | None = None,
    is_active: bool | None = None,
) -> ThresholdConfigResponse:
    """Update a threshold config.

    Args:
        config_id: The config to update.
        threshold: New threshold value.
        is_active: Activate or deactivate the config.

    Returns:
        ThresholdConfigResponse with updated config.
    """
    body: dict[str, Any] = {}
    if threshold is not None:
        body["threshold"] = threshold
    if is_active is not None:
        body["is_active"] = is_active
    data = _put(f"/threshold/configs/{config_id}", body)
    return _parse_threshold_config(data)


def _parse_threshold_config(data: dict[str, Any]) -> ThresholdConfigResponse:
    """Parse API response into ThresholdConfigResponse."""
    return ThresholdConfigResponse(
        id=data["id"],
        agent_id=data["agent_id"],
        threshold=data["threshold"],
        total_shares=data["total_shares"],
        is_active=data["is_active"],
        created_at=data["created_at"],
        updated_at=data.get("updated_at"),
    )


# --- Signing Entities ---

def add_entity(
    config_id: str,
    entity_class: str,
    label: str,
) -> SigningEntityResponse:
    """Add a signing entity to a threshold config.

    Entity classes: A (Agent), B (Human Operator), C (Policy Engine),
    D (Compliance Verifier), E (Organizational Authority).

    Args:
        config_id: The threshold config to add the entity to.
        entity_class: Entity class (A, B, C, D, or E).
        label: Human-readable label for the entity.

    Returns:
        SigningEntityResponse with entity details.

    Example:
        entity = asqav.add_entity("cfg_xxx", entity_class="B", label="operator-1")
    """
    data = _post(f"/threshold/configs/{config_id}/entities", {
        "entity_class": entity_class,
        "label": label,
    })
    return SigningEntityResponse(
        id=data["id"],
        config_id=data["config_id"],
        entity_class=data["entity_class"],
        label=data["label"],
        is_active=data["is_active"],
        created_at=data["created_at"],
    )


def list_entities(config_id: str) -> list[SigningEntityResponse]:
    """List signing entities for a threshold config.

    Args:
        config_id: The threshold config.

    Returns:
        List of SigningEntityResponse.
    """
    data = _get(f"/threshold/configs/{config_id}/entities")
    return [
        SigningEntityResponse(
            id=e["id"],
            config_id=e["config_id"],
            entity_class=e["entity_class"],
            label=e["label"],
            is_active=e["is_active"],
            created_at=e["created_at"],
        )
        for e in data
    ]


def remove_entity(entity_id: str) -> dict[str, Any]:
    """Remove a signing entity (soft delete).

    Args:
        entity_id: The entity to remove.

    Returns:
        Confirmation dict.
    """
    return _delete(f"/threshold/entities/{entity_id}")


# --- Threshold Keypairs ---

def generate_keypair(config_id: str) -> ThresholdKeypairResponse:
    """Generate a threshold ML-DSA keypair with Shamir secret sharing.

    Creates a keypair where the private key is split into shares
    distributed to the signing entities in the config.

    Args:
        config_id: The threshold config to generate a keypair for.

    Returns:
        ThresholdKeypairResponse with public key and metadata.

    Example:
        keypair = asqav.generate_keypair("cfg_xxx")
        print(f"Public key: {keypair.public_key_hex[:32]}...")
    """
    data = _post("/threshold/keypairs", {"config_id": config_id})
    return _parse_threshold_keypair(data)


def get_keypair(keypair_id: str) -> ThresholdKeypairResponse:
    """Get threshold keypair details (no raw shares).

    Args:
        keypair_id: The keypair ID.

    Returns:
        ThresholdKeypairResponse with keypair details.
    """
    data = _get(f"/threshold/keypairs/{keypair_id}")
    return _parse_threshold_keypair(data)


def threshold_sign(
    keypair_id: str,
    message_hex: str,
) -> ThresholdSignResponse:
    """Perform threshold ML-DSA signing.

    Requires that the threshold of entity approvals has been met.
    Output is a standard FIPS 204 ML-DSA-65 signature.

    Args:
        keypair_id: The threshold keypair to sign with.
        message_hex: The message to sign (hex-encoded).

    Returns:
        ThresholdSignResponse with signature and verification status.

    Example:
        result = asqav.threshold_sign("kp_xxx", message_hex="deadbeef")
        assert result.verified  # Standard ML-DSA verifier works
    """
    data = _post(f"/threshold/keypairs/{keypair_id}/sign", {
        "message_hex": message_hex,
    })
    return ThresholdSignResponse(
        signature_hex=data["signature_hex"],
        message_hex=data["message_hex"],
        keypair_id=data["keypair_id"],
        verified=data["verified"],
        audit_record_id=data.get("audit_record_id"),
    )


def refresh_keypair(keypair_id: str) -> KeyRefreshResponse:
    """Refresh key shares without changing the public key.

    Generates new Shamir shares while preserving the same public key.
    Invalidates all active delegations.

    Args:
        keypair_id: The keypair to refresh.

    Returns:
        KeyRefreshResponse with refresh details.
    """
    data = _post(f"/threshold/keypairs/{keypair_id}/refresh", {})
    return KeyRefreshResponse(
        keypair_id=data["keypair_id"],
        refreshed_at=data["refreshed_at"],
        delegations_invalidated=data["delegations_invalidated"],
    )


def recover_share(
    keypair_id: str,
    entity_id: str,
    contributing_entity_ids: list[str],
) -> ShareRecoveryResponse:
    """Recover a lost key share for an entity.

    Requires exactly t contributing entities to reconstruct the share.

    Args:
        keypair_id: The keypair containing the lost share.
        entity_id: The entity whose share is being recovered.
        contributing_entity_ids: Entity IDs providing their shares for recovery.

    Returns:
        ShareRecoveryResponse with recovery details.
    """
    data = _post(f"/threshold/keypairs/{keypair_id}/recover", {
        "entity_id": entity_id,
        "contributing_entity_ids": contributing_entity_ids,
    })
    return ShareRecoveryResponse(
        keypair_id=data["keypair_id"],
        recovered_entity_id=data["recovered_entity_id"],
        recovered_at=data["recovered_at"],
    )


def _parse_threshold_keypair(data: dict[str, Any]) -> ThresholdKeypairResponse:
    """Parse API response into ThresholdKeypairResponse."""
    return ThresholdKeypairResponse(
        id=data["id"],
        config_id=data["config_id"],
        public_key_hex=data["public_key_hex"],
        threshold=data["threshold"],
        total_shares=data["total_shares"],
        created_at=data["created_at"],
        status=data.get("status", "active"),
    )


# --- Risk Rules ---

def create_risk_rule(
    name: str,
    action_pattern: str,
    risk_level: str,
    threshold_override: int | None = None,
    priority: int = 0,
    entity_weights: dict[str, float] | None = None,
    time_schedule: dict[str, Any] | None = None,
) -> RiskRuleResponse:
    """Create a risk classification rule.

    Risk rules dynamically adjust thresholds based on action patterns.

    Args:
        name: Rule name.
        action_pattern: Glob pattern to match action types.
        risk_level: Risk level (low, medium, high, critical).
        threshold_override: Override the default threshold for matching actions.
        priority: Rule priority (higher = evaluated first).
        entity_weights: Per-class weight multipliers (e.g., {"D": 2.0}).
        time_schedule: Time-dependent threshold adjustment.

    Returns:
        RiskRuleResponse with rule details.

    Example:
        rule = asqav.create_risk_rule(
            name="High-risk finance",
            action_pattern="finance.*",
            risk_level="critical",
            threshold_override=3,
            priority=100,
        )
    """
    body: dict[str, Any] = {
        "name": name,
        "action_pattern": action_pattern,
        "risk_level": risk_level,
        "priority": priority,
    }
    if threshold_override is not None:
        body["threshold_override"] = threshold_override
    if entity_weights is not None:
        body["entity_weights"] = entity_weights
    if time_schedule is not None:
        body["time_schedule"] = time_schedule
    data = _post("/risk-rules", body)
    return _parse_risk_rule(data)


def list_risk_rules() -> list[RiskRuleResponse]:
    """List all risk rules for the organization.

    Returns:
        List of RiskRuleResponse.
    """
    data = _get("/risk-rules")
    return [_parse_risk_rule(r) for r in data]


def get_risk_rule(rule_id: str) -> RiskRuleResponse:
    """Get a risk rule by ID.

    Args:
        rule_id: The rule ID.

    Returns:
        RiskRuleResponse with rule details.
    """
    data = _get(f"/risk-rules/{rule_id}")
    return _parse_risk_rule(data)


def update_risk_rule(
    rule_id: str,
    **kwargs: Any,
) -> RiskRuleResponse:
    """Update a risk rule.

    Args:
        rule_id: The rule to update.
        **kwargs: Fields to update (name, action_pattern, risk_level,
                  threshold_override, priority, entity_weights, time_schedule).

    Returns:
        RiskRuleResponse with updated rule.
    """
    data = _put(f"/risk-rules/{rule_id}", kwargs)
    return _parse_risk_rule(data)


def delete_risk_rule(rule_id: str) -> dict[str, Any]:
    """Delete a risk rule.

    Args:
        rule_id: The rule to delete.

    Returns:
        Confirmation dict.
    """
    return _delete(f"/risk-rules/{rule_id}")


def _parse_risk_rule(data: dict[str, Any]) -> RiskRuleResponse:
    """Parse API response into RiskRuleResponse."""
    return RiskRuleResponse(
        id=data["id"],
        name=data["name"],
        action_pattern=data["action_pattern"],
        risk_level=data["risk_level"],
        threshold_override=data.get("threshold_override"),
        priority=data["priority"],
        entity_weights=data.get("entity_weights"),
        time_schedule=data.get("time_schedule"),
        created_at=data.get("created_at", ""),
    )


# --- Threshold Delegations ---

def create_delegation(
    config_id: str,
    delegator_entity_id: str,
    delegate_entity_id: str,
    expires_in: int = 86400,
) -> ThresholdDelegationResponse:
    """Create a temporary delegation between Class B entities.

    Allows one entity to delegate their signing authority to another
    with cryptographic expiry.

    Args:
        config_id: The threshold config.
        delegator_entity_id: Entity delegating their authority.
        delegate_entity_id: Entity receiving delegated authority.
        expires_in: Delegation duration in seconds (default 24h).

    Returns:
        ThresholdDelegationResponse with delegation details.

    Example:
        delegation = asqav.create_delegation(
            config_id="cfg_xxx",
            delegator_entity_id="ent_alice",
            delegate_entity_id="ent_bob",
            expires_in=3600,  # 1 hour
        )
    """
    data = _post("/threshold/delegations", {
        "config_id": config_id,
        "delegator_entity_id": delegator_entity_id,
        "delegate_entity_id": delegate_entity_id,
        "expires_in": expires_in,
    })
    return _parse_delegation(data)


def list_delegations() -> list[ThresholdDelegationResponse]:
    """List all active delegations.

    Returns:
        List of ThresholdDelegationResponse.
    """
    data = _get("/threshold/delegations")
    return [_parse_delegation(d) for d in data]


def get_delegation(delegation_id: str) -> ThresholdDelegationResponse:
    """Get a delegation by ID.

    Args:
        delegation_id: The delegation ID.

    Returns:
        ThresholdDelegationResponse.
    """
    data = _get(f"/threshold/delegations/{delegation_id}")
    return _parse_delegation(data)


def revoke_delegation(delegation_id: str) -> dict[str, Any]:
    """Revoke an active delegation.

    Args:
        delegation_id: The delegation to revoke.

    Returns:
        Confirmation dict.
    """
    return _delete(f"/threshold/delegations/{delegation_id}")


def _parse_delegation(data: dict[str, Any]) -> ThresholdDelegationResponse:
    """Parse API response into ThresholdDelegationResponse."""
    return ThresholdDelegationResponse(
        id=data["id"],
        config_id=data["config_id"],
        delegator_entity_id=data["delegator_entity_id"],
        delegate_entity_id=data["delegate_entity_id"],
        expires_at=data["expires_at"],
        is_active=data["is_active"],
        created_at=data["created_at"],
    )


# --- List Threshold Sessions ---

def list_sessions(
    status: str | None = None,
    agent_id: str | None = None,
    limit: int = 100,
) -> list[ThresholdSessionResponse]:
    """List threshold signing sessions.

    Args:
        status: Filter by status (pending, approved, denied, expired).
        agent_id: Filter by agent ID.
        limit: Max results (default 100).

    Returns:
        List of ThresholdSessionResponse.
    """
    params = [f"limit={limit}"]
    if status:
        params.append(f"status={status}")
    if agent_id:
        params.append(f"agent_id={agent_id}")
    path = "/threshold/sessions?" + "&".join(params)
    data = _get(path)
    return [_parse_threshold_session(s) for s in data]


def export_audit_json(
    start_date: str | None = None,
    end_date: str | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """Export signed actions as JSON for audit trail export (Pro+ tier).

    Args:
        start_date: Filter by start date (ISO format).
        end_date: Filter by end date (ISO format).
        agent_id: Filter by agent ID.

    Returns:
        Dict with export data including signatures and verification URLs.
    """
    params = []
    if start_date:
        params.append(f"start_date={start_date}")
    if end_date:
        params.append(f"end_date={end_date}")
    if agent_id:
        params.append(f"agent_id={agent_id}")

    path = "/export/json"
    if params:
        path += "?" + "&".join(params)

    return _get(path)


def export_audit_csv(
    start_date: str | None = None,
    end_date: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Export signed actions as CSV for audit trail export (Pro+ tier).

    Args:
        start_date: Filter by start date (ISO format).
        end_date: Filter by end date (ISO format).
        agent_id: Filter by agent ID.

    Returns:
        CSV string with signed actions.
    """
    _ensure_initialized()

    params = []
    if start_date:
        params.append(f"start_date={start_date}")
    if end_date:
        params.append(f"end_date={end_date}")
    if agent_id:
        params.append(f"agent_id={agent_id}")

    path = "/export/csv"
    if params:
        path += "?" + "&".join(params)

    if _HTTPX_AVAILABLE and _client:
        response = _client.get(path)
        _handle_response(response)
        return response.text
    else:
        import urllib.error
        import urllib.request

        url = urljoin(_api_base, path)
        headers = {
            "Authorization": f"Bearer {_api_key}",
        }
        request = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise AuthenticationError("Invalid API key") from e
            elif e.code == 429:
                raise RateLimitError("Rate limit exceeded") from e
            else:
                raise APIError(str(e), e.code) from e
