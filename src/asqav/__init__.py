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
    KeyRefreshResponse,
    RateLimitError,
    RiskRuleResponse,
    SDTokenResponse,
    SessionResponse,
    ShareRecoveryResponse,
    SignatureResponse,
    SignedActionResponse,
    SigningEntityResponse,
    Span,
    ThresholdApproveResponse,
    ThresholdConfigResponse,
    ThresholdDelegationResponse,
    ThresholdKeypairResponse,
    ThresholdSessionResponse,
    ThresholdSignatureDetail,
    ThresholdSignResponse,
    TokenResponse,
    VerificationResponse,
    add_entity,
    approve_action,
    configure_otel,
    create_delegation,
    create_risk_rule,
    create_threshold_config,
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
    get_threshold_config,
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
    threshold_sign,
    update_risk_rule,
    update_threshold_config,
    verify_signature,
)

__version__ = "0.2.6"
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
    # Threshold (Multi-Key Authorization)
    "ThresholdSessionResponse",
    "ThresholdApproveResponse",
    "ThresholdSignatureDetail",
    "request_action",
    "approve_action",
    "get_action_status",
    # Threshold (v2.0)
    # Threshold Config
    "ThresholdConfigResponse",
    "create_threshold_config",
    "get_threshold_config",
    "update_threshold_config",
    # Signing Entities
    "SigningEntityResponse",
    "add_entity",
    "list_entities",
    "remove_entity",
    # Threshold Keypairs
    "ThresholdKeypairResponse",
    "ThresholdSignResponse",
    "generate_keypair",
    "get_keypair",
    "threshold_sign",
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
    "ThresholdDelegationResponse",
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
    # Exceptions
    "AsqavError",
    "AuthenticationError",
    "RateLimitError",
    "APIError",
]
