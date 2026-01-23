"""
asqav - Quantum-safe control for AI agents.

Thin SDK that connects to asqav.com. All ML-DSA cryptography happens server-side.

Quick Start:
    import asqav

    asqav.init(api_key="sk_...")
    agent = asqav.Agent.create("my-agent")
    sig = agent.sign("api:call", {"model": "gpt-4"})

With Tracing:
    import asqav

    asqav.init()

    with asqav.span("api:openai", {"model": "gpt-4"}) as s:
        response = client.chat.completions.create(...)
        s.set_attribute("tokens", response.usage.total_tokens)

With Decorators:
    @asqav.secure
    def my_agent_function():
        return "Cryptographically signed"

Get your API key at asqav.com
"""

from .client import (
    Agent,
    AgentResponse,
    APIError,
    AsqavError,
    AuthenticationError,
    CertificateResponse,
    DelegationResponse,
    RateLimitError,
    SDTokenResponse,
    SessionResponse,
    SignatureResponse,
    SignedActionResponse,
    Span,
    TokenResponse,
    VerificationResponse,
    configure_otel,
    export_audit_csv,
    export_audit_json,
    export_spans,
    flush_spans,
    get_agent,
    get_current_span,
    get_session_signatures,
    init,
    secure,
    secure_async,
    span,
    verify_signature,
)

__version__ = "0.2.5"
__all__ = [
    # Initialization
    "init",
    # Agent
    "Agent",
    "AgentResponse",
    "get_agent",
    # Responses
    "TokenResponse",
    "SDTokenResponse",
    "SignatureResponse",
    "SignedActionResponse",
    "SessionResponse",
    "DelegationResponse",
    "CertificateResponse",
    "VerificationResponse",
    # Verification
    "verify_signature",
    # Sessions
    "get_session_signatures",
    # Export
    "export_audit_json",
    "export_audit_csv",
    # Tracing
    "Span",
    "span",
    "get_current_span",
    # OTEL Export
    "configure_otel",
    "export_spans",
    "flush_spans",
    # Decorators
    "secure",
    "secure_async",
    # Exceptions
    "AsqavError",
    "AuthenticationError",
    "RateLimitError",
    "APIError",
]
