"""Response objects returned by the client."""

from __future__ import annotations

from dataclasses import dataclass


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
