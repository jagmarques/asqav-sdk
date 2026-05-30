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

__version__ = "0.5.5"
__all__ = [
    # Initialization
    "init",
    "health_check",
    # Fingerprint helpers
    "canonicalize",
    "canonicalize_tool_args",
    "canonical_json",
    "hash_action",
    # Agent
    "Agent",
    "AgentResponse",
    "get_agent",
    "PreflightResult",
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
    # Agents
    "list_agents",
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
    "verify_output",
    "VerificationDetail",
    "BitcoinAnchorStatus",
    # May 2026 release helpers
    "post_applied_attestation",
    "list_rejected_attempts",
    # Trust Signal Export
    "generate_attestation",
    "verify_attestation",
    # Sessions
    "get_session_signatures",
    # Export
    "export_audit_json",
    "export_audit_csv",
    # Trace Correlation
    "generate_trace_id",
    # Emergency Halt
    "emergency_halt",
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
    # Budget Tracking
    "BudgetTracker",
    "BudgetCheckResult",
    # Compliance
    "ComplianceBundle",
    "export_bundle",
    "fetch_audit_pack",
    # Replay
    "replay",
    "replay_from_bundle",
    "ReplayTimeline",
    "ReplayStep",
    # Patterns
    "PATTERNS",
    "resolve_pattern",
    "list_patterns",
    # Scope Tokens
    "ScopeToken",
    "create_scope_token",
    "verify_scope_token",
    "is_replay",
    # Three-Phase Signing
    "PhaseChain",
    "sign_with_phases",
    # Reasoning Trace
    "ReasoningReceipt",
    "sign_reasoning",
    # Hooks
    "register_before",
    "register_after",
    "clear_hooks",
    # Exceptions
    "AsqavError",
    "AuthenticationError",
    "RateLimitError",
    "APIError",
    # IETF Compliance Receipts profile
    "RECEIPT_TYPE_NAMESPACE",
    "SKEW_BOUND_SECONDS",
    "ComplianceReceiptVerification",
    "verify_compliance_receipt",
    # Counterparty acknowledgment binding (IETF -04 counterparty_binding extension)
    "ACKNOWLEDGMENT_RECEIPT_TYPE",
    "CounterpartyBinding",
    "CounterpartyBindingVerification",
    "compute_counterparty_binding",
    "verify_counterparty_binding",
    # DORA RTS JC 2024-33 Annex II vocabulary
    "DORA_INCIDENT_CLASS_NAMESPACE",
    # IETF -04 capture-topologies appendix vocabulary
    "CAPTURE_TOPOLOGY_NAMESPACE",
    # Algorithm agility
    "ALGORITHM_ML_DSA_65",
    "ALGORITHM_ED25519",
    "ALGORITHM_ES256",
    "SUPPORTED_ALGORITHMS",
    "LocalKeypair",
    "generate_local_keypair",
]
