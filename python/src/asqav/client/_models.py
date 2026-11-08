"""Response objects returned by the client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


    # Response from signing group operations.
@dataclass
class SigningGroupResponse:

    id: str
    agent_id: str
    min_approvals: int
    total_shares: int
    is_active: bool
    created_at: str
    updated_at: str | None = None


    # Response from signing entity operations.
@dataclass
class SigningEntityResponse:

    id: str
    config_id: str
    entity_class: str
    label: str
    is_active: bool
    created_at: str


    # Response from group keypair operations.
@dataclass
class GroupKeypairResponse:

    id: str
    config_id: str
    public_key_hex: str
    min_approvals: int
    total_shares: int
    created_at: str
    status: str = "active"


    # Response from group signing operation.
@dataclass
class GroupSignResponse:

    signature_hex: str
    message_hex: str
    keypair_id: str
    verified: bool
    audit_record_id: str | None = None


    # Response from signing approval operation.
@dataclass
class ApprovalResponse:

    session_id: str
    entity_id: str
    signatures_collected: int
    approvals_required: int
    status: str
    approved: bool


    # Response from risk rule operations.
@dataclass
class RiskRuleResponse:

    id: str
    name: str
    action_pattern: str
    risk_level: str
    approval_override: int | None
    priority: int
    entity_weights: dict[str, float] | None = None
    time_schedule: dict[str, Any] | None = None
    created_at: str = ""


    # Response from delegation operations.
@dataclass
class DelegationResponse:

    id: str
    config_id: str
    delegator_entity_id: str
    delegate_entity_id: str
    expires_at: str
    is_active: bool
    created_at: str


    # Response from key share refresh operation.
@dataclass
class KeyRefreshResponse:

    keypair_id: str
    refreshed_at: str
    delegations_invalidated: int


    # Response from key share recovery operation.
@dataclass
class ShareRecoveryResponse:

    keypair_id: str
    recovered_entity_id: str
    recovered_at: str


@dataclass
class VerificationDetail:
    """Granular per-axis verification outcome, optional on the response.
    `validation_label` is the cloud verdict; IETF sub-axes need compliance."""

    signer_key_match: bool
    signature_valid: bool
    algorithm_match: bool
    agent_active: bool
    validation_label: str
    signed_at_skew_seconds: float | None = None
    chain_valid: bool | None = None
    anchor_valid_ots: bool | None = None
    anchor_valid_rfc3161: bool | None = None
    anchor_status_ots: str | None = None
    anchor_status_rfc3161: str | None = None
    anchor_independent_rfc3161: str | None = None
    missing_fields: list[str] | None = None
    policy_digest_resolved: bool | None = None
    duplicate_emission_candidate: bool | None = None
    regimes_satisfied: list[str] = field(default_factory=list)


@dataclass
class ExecutionEvidence:
    """Execution-evidence projection on a verification response.
    `label`: result_bound, digest_present, none_by_design or absent."""

    present: bool
    label: str
    result_digest: str | None = None
    result_bound: bool = False
    none_by_design: bool = False


@dataclass
class BitcoinAnchorStatus:
    """Anchor attestation state on a verification response.
    `status`: none, pending, confirmed or failed."""

    status: str
    anchor_tx_ref: str | None = None
    anchor_block_height: int | None = None


@dataclass
class VerificationResponse:
    """Public verification response; `verified` is signature-only.
    Anchor outcome is on `bitcoin_anchor`; IETF fields need compliance mode."""

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
    verification_detail: VerificationDetail | None = None
    execution_evidence: ExecutionEvidence | None = None
    type: str = "signature"
    bitcoin_anchor: BitcoinAnchorStatus | None = None
    signature_envelope: dict[str, str] | None = None
    anchors: list[dict[str, Any]] | None = None
    algorithm_registry_version: str | None = None


    # Signed action from a session.
@dataclass
class SignedActionResponse:

    signature_id: str
    agent_id: str
    action_id: str
    action_type: str
    payload: dict[str, Any] | None
    algorithm: str
    signed_at: float
    signature_preview: str
    verification_url: str


    # Details of a single entity signature within a signing session.
@dataclass
class SignatureDetail:

    entity_id: str
    entity_name: str
    entity_class: str
    signed_at: str


    # Response from signing session operations.
@dataclass
class SigningSessionResponse:

    session_id: str
    config_id: str
    agent_id: str
    action_type: str
    action_params: dict[str, Any] | None
    approvals_required: int
    signatures_collected: int
    status: str
    signatures: list[SignatureDetail]
    policy_attestation_hash: str | None
    created_at: str
    expires_at: str
    resolved_at: str | None = None
