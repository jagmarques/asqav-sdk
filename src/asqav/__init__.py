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
    ApprovalResponse,
    AsqavError,
    AuthenticationError,
    CertificateResponse,
    DelegationResponse,
    GroupKeypairResponse,
    GroupSignResponse,
    KeyRefreshResponse,
    RateLimitError,
    RiskRuleResponse,
    SDTokenResponse,
    SessionResponse,
    ShareRecoveryResponse,
    SignatureDetail,
    SignatureResponse,
    SignedActionResponse,
    SigningEntityResponse,
    SigningGroupResponse,
    SigningSessionResponse,
    Span,
    TokenResponse,
    VerificationResponse,
    add_entity,
    approve_action,
    configure_otel,
    create_delegation,
    create_risk_rule,
    create_signing_group,
    delete_risk_rule,
    export_audit_csv,
    export_audit_json,
    export_spans,
    flush_spans,
    generate_keypair,
    get_action_status,
    get_agent,
    get_current_span,
    get_delegation,
    get_keypair,
    get_risk_rule,
    get_session_signatures,
    get_signing_group,
    group_sign,
    health_check,
    init,
    list_delegations,
    list_entities,
    list_risk_rules,
    list_sessions,
    recover_share,
    refresh_keypair,
    remove_entity,
    request_action,
    revoke_delegation,
    secure,
    secure_async,
    span,
    update_risk_rule,
    update_signing_group,
    verify_signature,
)

from .async_client import AsyncAgent
from .decorators import async_session, session, sign
from .local import LocalQueue, local_sign
from .retry import with_async_retry, with_retry

__version__ = "0.2.7"
__all__ = [
    # Initialization
    "init",
    "health_check",
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
    "CertificateResponse",
    "VerificationResponse",
    # Multi-Party Signing
    "SigningSessionResponse",
    "ApprovalResponse",
    "SignatureDetail",
    "request_action",
    "approve_action",
    "get_action_status",
    # Signing Groups
    "SigningGroupResponse",
    "create_signing_group",
    "get_signing_group",
    "update_signing_group",
    # Signing Entities
    "SigningEntityResponse",
    "add_entity",
    "list_entities",
    "remove_entity",
    # Group Keypairs
    "GroupKeypairResponse",
    "GroupSignResponse",
    "generate_keypair",
    "get_keypair",
    "group_sign",
    "KeyRefreshResponse",
    "refresh_keypair",
    "ShareRecoveryResponse",
    "recover_share",
    # Risk Rules
    "RiskRuleResponse",
    "create_risk_rule",
    "list_risk_rules",
    "get_risk_rule",
    "update_risk_rule",
    "delete_risk_rule",
    # Delegations
    "DelegationResponse",
    "create_delegation",
    "list_delegations",
    "get_delegation",
    "revoke_delegation",
    # Session Listing
    "list_sessions",
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
    "sign",
    "session",
    "async_session",
    # Async
    "AsyncAgent",
    # Local Queue
    "LocalQueue",
    "local_sign",
    # Retry
    "with_retry",
    "with_async_retry",
    # Exceptions
    "AsqavError",
    "AuthenticationError",
    "RateLimitError",
    "APIError",
]
