"""
asqav - AI agent governance: audit trails, policy enforcement, compliance.

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

from ._jcs import canonical_json
from .async_client import AsyncAgent
from .canonicalize import canonicalize, canonicalize_tool_args, hash_action
from .client import (
    CAPTURE_TOPOLOGY_NAMESPACE,
    DORA_INCIDENT_CLASS_NAMESPACE,
    RECEIPT_TYPE_NAMESPACE,
    SKEW_BOUND_SECONDS,
    Agent,
    AgentResponse,
    APIError,
    ApprovalResponse,
    AsqavError,
    AuthenticationError,
    BitcoinAnchorStatus,
    BudgetCheckResult,
    BudgetTracker,
    CertificateResponse,
    ComplianceReceiptVerification,
    DelegationResponse,
    GroupKeypairResponse,
    GroupSignResponse,
    KeyRefreshResponse,
    PreflightResult,
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
    VerificationDetail,
    VerificationResponse,
    add_entity,
    approve_action,
    configure_otel,
    create_delegation,
    create_risk_rule,
    create_signing_group,
    delete_risk_rule,
    emergency_halt,
    export_audit_csv,
    export_audit_json,
    export_spans,
    flush_spans,
    generate_attestation,
    generate_keypair,
    generate_trace_id,
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
    list_agents,
    list_delegations,
    list_entities,
    list_rejected_attempts,
    list_risk_rules,
    list_sessions,
    post_applied_attestation,
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
    verify_attestation,
    verify_compliance_receipt,
    verify_output,
    verify_signature,
)
from .compliance import ComplianceBundle, export_bundle, fetch_audit_pack
from .counterparty import (
    ACKNOWLEDGMENT_RECEIPT_TYPE,
    CounterpartyBinding,
    CounterpartyBindingVerification,
    compute_counterparty_binding,
    verify_counterparty_binding,
)
from .decorators import async_session, session, sign
from .hooks import clear_hooks, register_after, register_before
from .keys import (
    ALGORITHM_ED25519,
    ALGORITHM_ES256,
    ALGORITHM_ML_DSA_65,
    SUPPORTED_ALGORITHMS,
    LocalKeypair,
    generate_local_keypair,
)
from .local import LocalQueue, local_sign
from .patterns import PATTERNS, list_patterns, resolve_pattern
from .phases import PhaseChain, sign_with_phases
from .reasoning import ReasoningReceipt, sign_reasoning
from .replay import ReplayStep, ReplayTimeline, replay, replay_from_bundle
from .retry import with_async_retry, with_retry
from .scope import ScopeToken, create_scope_token, is_replay, verify_scope_token

__version__ = "0.4.5"
__all__ = [
    # Counterparty acknowledgment binding (IETF -04 counterparty_binding extension)
    "ACKNOWLEDGMENT_RECEIPT_TYPE",
    "ALGORITHM_ED25519",
    "ALGORITHM_ES256",
    # Algorithm agility
    "ALGORITHM_ML_DSA_65",
    # IETF -04 capture-topologies appendix vocabulary
    "CAPTURE_TOPOLOGY_NAMESPACE",
    # DORA RTS JC 2024-33 Annex II vocabulary
    "DORA_INCIDENT_CLASS_NAMESPACE",
    # Patterns
    "PATTERNS",
    # IETF Compliance Receipts profile
    "RECEIPT_TYPE_NAMESPACE",
    "SKEW_BOUND_SECONDS",
    "SUPPORTED_ALGORITHMS",
    "APIError",
    # Agent
    "Agent",
    "AgentResponse",
    "ApprovalResponse",
    # Exceptions
    "AsqavError",
    # Async
    "AsyncAgent",
    "AuthenticationError",
    "BitcoinAnchorStatus",
    "BudgetCheckResult",
    # Budget Tracking
    "BudgetTracker",
    "CertificateResponse",
    # Compliance
    "ComplianceBundle",
    "ComplianceReceiptVerification",
    "CounterpartyBinding",
    "CounterpartyBindingVerification",
    # Delegations
    "DelegationResponse",
    # Group Keypairs
    "GroupKeypairResponse",
    "GroupSignResponse",
    "KeyRefreshResponse",
    "LocalKeypair",
    # Local Queue
    "LocalQueue",
    # Three-Phase Signing
    "PhaseChain",
    "PreflightResult",
    "RateLimitError",
    # Reasoning Trace
    "ReasoningReceipt",
    "ReplayStep",
    "ReplayTimeline",
    # Risk Rules
    "RiskRuleResponse",
    "SDTokenResponse",
    # Scope Tokens
    "ScopeToken",
    "SessionResponse",
    "ShareRecoveryResponse",
    "SignatureDetail",
    "SignatureResponse",
    "SignedActionResponse",
    # Signing Entities
    "SigningEntityResponse",
    # Signing Groups
    "SigningGroupResponse",
    # Multi-Party Signing
    "SigningSessionResponse",
    # Tracing
    "Span",
    # Responses
    "TokenResponse",
    "VerificationDetail",
    "VerificationResponse",
    "add_entity",
    "approve_action",
    "async_session",
    "canonical_json",
    # Fingerprint helpers
    "canonicalize",
    "canonicalize_tool_args",
    "clear_hooks",
    "compute_counterparty_binding",
    # OTEL Export
    "configure_otel",
    "create_delegation",
    "create_risk_rule",
    "create_scope_token",
    "create_signing_group",
    "delete_risk_rule",
    # Emergency Halt
    "emergency_halt",
    "export_audit_csv",
    # Export
    "export_audit_json",
    "export_bundle",
    "export_spans",
    "fetch_audit_pack",
    "flush_spans",
    # Trust Signal Export
    "generate_attestation",
    "generate_keypair",
    "generate_local_keypair",
    # Trace Correlation
    "generate_trace_id",
    "get_action_status",
    "get_agent",
    "get_current_span",
    "get_delegation",
    "get_keypair",
    "get_risk_rule",
    # Sessions
    "get_session_signatures",
    "get_signing_group",
    "group_sign",
    "hash_action",
    "health_check",
    # Initialization
    "init",
    "is_replay",
    # Agents
    "list_agents",
    "list_delegations",
    "list_entities",
    "list_patterns",
    "list_rejected_attempts",
    "list_risk_rules",
    # Session Listing
    "list_sessions",
    "local_sign",
    # May 2026 release helpers
    "post_applied_attestation",
    "recover_share",
    "refresh_keypair",
    "register_after",
    # Hooks
    "register_before",
    "remove_entity",
    # Replay
    "replay",
    "replay_from_bundle",
    "request_action",
    "resolve_pattern",
    "revoke_delegation",
    # Decorators
    "secure",
    "secure_async",
    "session",
    "sign",
    "sign_reasoning",
    "sign_with_phases",
    "span",
    "update_risk_rule",
    "update_signing_group",
    "verify_attestation",
    "verify_compliance_receipt",
    "verify_counterparty_binding",
    "verify_output",
    "verify_scope_token",
    # Verification
    "verify_signature",
    "with_async_retry",
    # Retry
    "with_retry",
]
