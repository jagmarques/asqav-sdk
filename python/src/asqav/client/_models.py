"""Response objects returned by the client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


    # Response from agent creation.
@dataclass
class AgentResponse:

    agent_id: str
    name: str
    public_key: str
    key_id: str
    algorithm: str
    capabilities: list[str]
    created_at: float


    # Response from token issuance.
@dataclass
class TokenResponse:

    token: str
    expires_at: float
    algorithm: str


    # Turn the JSON bitcoin_anchor block from the API into a typed value.
def _parse_bitcoin_anchor(raw: dict[str, Any] | None) -> "BitcoinAnchor | None":
    if not raw:
        return None
    return BitcoinAnchor(
        status=raw.get("status") or "none",
        anchor_tx_ref=raw.get("anchor_tx_ref"),
        anchor_block_height=raw.get("anchor_block_height"),
    )


@dataclass
class BitcoinAnchor:
    """Anchor attestation state of a signed action; branch on ``status``.
    ``none``, ``pending``, ``confirmed`` or ``failed`` (treat as unanchored)."""

    status: str
    anchor_tx_ref: str | None = None
    anchor_block_height: int | None = None


@dataclass
class SignatureResponse:
    """Response from signing operation.

    The ML-DSA ``signature`` is always present. ``bitcoin_anchor`` carries
    the anchoring state separately so callers can distinguish a
    cryptographically signed action from one that is also Bitcoin-anchored.
    """

    # Polymorphic: str (b64) for non-compliance, {alg, kid, sig} dict under
    # compliance_mode. Use signature_envelope() for the dict form.
    signature: str | dict[str, str]
    signature_id: str
    action_id: str
    timestamp: float
    verification_url: str
    algorithm: str | None = None
    chain_hash: str | None = None
    rfc3161_tsa: str | None = None
    rfc3161_serial: str | None = None
    scan_result: dict[str, Any] | None = None
    policy_digest: str | None = None
    policy_decision: str = "permit"
    authorization_ref: str | None = None
    bitcoin_anchor: BitcoinAnchor | None = None
    # Multi-party countersigning: peer agent_ids expected to countersign.
    required_co_signers: list[str] | None = None
    co_signatures: list[dict[str, Any]] | None = None
    countersign_url: str | None = None
    # True when the server validated a user_intent envelope on this record.
    user_intent_verified: bool | None = None
    # Compliance Receipts profile fields, set by the cloud in compliance_mode.
    compliance_mode: bool = False
    receipt_type: str | None = None
    action_ref: str | None = None
    # payload_digest accepts both shapes: the flat
    # "sha256:<hex>" string or the wire object form {"hash", "size"}.
    payload_digest: str | dict[str, Any] | None = None
    issuer_id: str | None = None
    iteration_id: str | None = None
    sandbox_state: str | None = None
    risk_class: str | None = None
    # str or list of strings for multi-regime cases.
    incident_class: str | list[str] | None = None
    reason: str | None = None
    previous_receipt_hash: str | None = None
    # Spec `decision` token (allow | deny | rate_limit); None off compliance.
    decision: str | None = None
    # Three-key Compliance Receipts envelope (payload + anchors); else None.
    payload: dict[str, Any] | None = None
    anchors: list[dict[str, Any]] | None = None

    def signature_envelope(self) -> dict[str, str] | None:
        """Return the spec-shape signature envelope `{alg, kid, sig}`.

        Returns the cloud-supplied object form when present (compliance
        mode); None otherwise.
        """
        if isinstance(self.signature, dict):
            keys = set(self.signature.keys())
            if keys >= {"alg", "kid", "sig"}:
                return self.signature
        return None

    def receipt(self) -> dict[str, Any] | None:
        """The receipt to hand a peer agent: ``{signature_id, payload, signature}``.

        ``payload`` and ``signature`` are the signed envelope a peer hashes and
        verifies; ``signature_id`` is the locator it acknowledges against. None
        outside compliance mode, where there is no signed envelope to share.
        """
        signature = self.signature_envelope()
        if signature is None or not isinstance(self.payload, dict):
            return None
        return {"signature_id": self.signature_id, "payload": self.payload, "signature": signature}

    def anchors_array(self) -> list[dict[str, Any]] | None:
        """Return the cloud-supplied `anchors[]` array. None on
        non-compliance receipts; [] under compliance_mode before
        anchors land.
        """
        return self.anchors


    # Response from session operations.
@dataclass
class SessionResponse:

    session_id: str
    agent_id: str
    status: str
    started_at: str  # ISO datetime string
    ended_at: str | None = None


@dataclass
class SDTokenResponse:
    """Response from SD-JWT token issuance (Enterprise tier).

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
            # Full token has: role, org, capabilities
            # Present only role to partner:
            proof = sd_token.present(["role"])
        """
        parts = [self.jwt]
        for claim_name in disclose:
            if claim_name in self.disclosures:
                parts.append(self.disclosures[claim_name])
        return "~".join(parts) + "~"

        # Return full SD-JWT with all disclosures.
    def full(self) -> str:
        return self.token


    # Agent identity certificate.
@dataclass
class CertificateResponse:

    agent_id: str
    agent_name: str
    algorithm: str
    public_key_pem: str
    key_id: str
    created_at: float
    is_revoked: bool


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


    # Response from signing approval operation.
@dataclass
class ApprovalResponse:

    session_id: str
    entity_id: str
    signatures_collected: int
    approvals_required: int
    status: str
    approved: bool


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
class PreflightResult:
    """Result of a pre-flight check of agent status and policy.
    ``cleared`` lets the agent proceed; ``checks_complete`` False: an error."""

    cleared: bool
    agent_active: bool
    policy_allowed: bool
    reasons: list[str]
    explanation: str = ""
    checks_complete: bool = True
