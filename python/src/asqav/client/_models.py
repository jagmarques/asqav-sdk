"""Response objects returned by the client."""

from __future__ import annotations

from dataclasses import dataclass
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
