"""
asqav API Client - Thin SDK that connects to asqav.com.

This module provides the API client for connecting to asqav Cloud.
All cryptographic operations happen server-side; this SDK handles
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
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections.abc import Callable, Generator
from typing import Any, TypeVar
from urllib.parse import urljoin

F = TypeVar("F", bound=Callable[..., Any])

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore

# API configuration
_api_key: str | None = None
_api_base: str = os.environ.get("ASQAV_API_URL", "https://api.asqav.com/v1")
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
    action_id: str
    timestamp: float


@dataclass
class SessionResponse:
    """Response from session operations."""

    session_id: str
    agent_id: str
    status: str
    started_at: float
    ended_at: float | None = None
    action_count: int = 0


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
            {"key": k, "value": {"stringValue": str(v)}}
            for k, v in s.attributes.items()
        ] + ([
            {"key": "asqav.signature", "value": {"stringValue": s.signature}}
        ] if s.signature else []),
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
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "asqav"}},
                ]
            },
            "scopeSpans": [{
                "scope": {"name": "asqav", "version": "0.1.0"},
                "spans": spans,
            }]
        }]
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

    All cryptographic operations are performed server-side.
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
            created_at=data["created_at"],
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
            created_at=data["created_at"],
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
            expires_at=data["expires_at"],
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
            expires_at=data["expires_at"],
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
            action_id=data["action_id"],
            timestamp=data["timestamp"],
        )

    def start_session(self) -> SessionResponse:
        """Start a new monitoring session.

        Returns:
            SessionResponse with session details.
        """
        data = _post(f"/agents/{self.agent_id}/sessions", {})

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
            status: Final status (completed, error).

        Returns:
            SessionResponse with final details.
        """
        if not self._session_id:
            raise AsqavError("No active session")

        data = _post(
            f"/agents/{self.agent_id}/sessions/{self._session_id}/end",
            {"status": status},
        )

        session_id = self._session_id
        self._session_id = None

        return SessionResponse(
            session_id=session_id,
            agent_id=data["agent_id"],
            status=data["status"],
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            action_count=data["action_count"],
        )

    def revoke(self, reason: str = "manual") -> None:
        """Revoke this agent's credentials.

        Revocation propagates globally across the asqav network.

        Args:
            reason: Reason for revocation.
        """
        _post(
            f"/agents/{self.agent_id}/revoke",
            {"reason": reason},
        )

    @property
    def is_revoked(self) -> bool:
        """Check if this agent is revoked."""
        data = _get(f"/agents/{self.agent_id}/status")
        return bool(data.get("revoked", False))


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
            "API key required. Set ASQAV_API_KEY or pass api_key to init(). "
            "Get yours at asqav.com"
        )

    if base_url:
        _api_base = base_url

    # Initialize HTTP client
    if _HTTPX_AVAILABLE:
        _client = httpx.Client(
            base_url=_api_base,
            headers={"Authorization": f"Bearer {_api_key}"},
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


def _ensure_initialized() -> None:
    """Ensure the SDK is initialized."""
    if not _api_key:
        raise AuthenticationError(
            "Call asqav.init() first. Get your API key at asqav.com"
        )


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
        "Authorization": f"Bearer {_api_key}",
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
    """Decorator to secure a function with asqav session logging.

    Wraps the function in an asqav session, logging the call as an action.
    All logging is done server-side with ML-DSA signatures.

    Args:
        func: The function to secure.

    Returns:
        The wrapped function.

    Example:
        import asqav

        asqav.init(api_key="sk_...")

        @asqav.secure
        def process_data(data: dict) -> dict:
            # This call is logged with cryptographic proof
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
            # This call is logged with cryptographic proof
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
