"""Signing groups, entities and threshold keypairs."""

from __future__ import annotations

from typing import Any

from . import _transport
from ._models import (
    GroupKeypairResponse,
    GroupSignResponse,
    KeyRefreshResponse,
    ShareRecoveryResponse,
    SigningEntityResponse,
    SigningGroupResponse,
)

# --- Signing Groups ---


def create_signing_group(
    agent_id: str,
    min_approvals: int,
    total_shares: int,
) -> SigningGroupResponse:
    """Create a signing group for an agent.

    Args:
        agent_id: The agent to create a signing group for.
        min_approvals: Minimum approvals required (t in t-of-n).
        total_shares: Total key shares to generate (n in t-of-n).

    Returns:
        SigningGroupResponse with config details.

    Example:
        config = asqav.create_signing_group("agt_xxx", min_approvals=2, total_shares=3)
    """
    data = _transport._post(
        "/signing-groups/configs",
        {
            "agent_id": agent_id,
            "min_approvals": min_approvals,
            "total_shares": total_shares,
        },
    )
    return _parse_signing_group(data)


def get_signing_group(agent_id: str) -> SigningGroupResponse:
    """Get the active signing group for an agent.

    Args:
        agent_id: The agent ID.

    Returns:
        SigningGroupResponse with config details.
    """
    data = _transport._get(f"/signing-groups/configs/{agent_id}")
    return _parse_signing_group(data)


def update_signing_group(
    config_id: str,
    min_approvals: int | None = None,
    is_active: bool | None = None,
) -> SigningGroupResponse:
    """Update a signing group.

    Args:
        config_id: The config to update.
        min_approvals: New minimum approvals value.
        is_active: Activate or deactivate the config.

    Returns:
        SigningGroupResponse with updated config.
    """
    body: dict[str, Any] = {}
    if min_approvals is not None:
        body["min_approvals"] = min_approvals
    if is_active is not None:
        body["is_active"] = is_active
    data = _transport._put(f"/signing-groups/configs/{config_id}", body)
    return _parse_signing_group(data)


    # Parse API response into a SigningGroupResponse.
def _parse_signing_group(data: dict[str, Any]) -> SigningGroupResponse:
    return SigningGroupResponse(
        id=data["id"],
        agent_id=data["agent_id"],
        min_approvals=data["min_approvals"],
        total_shares=data["total_shares"],
        is_active=data["is_active"],
        created_at=data["created_at"],
        updated_at=data.get("updated_at"),
    )


# --- Signing Entities ---


def add_entity(
    config_id: str,
    entity_class: str,
    label: str,
) -> SigningEntityResponse:
    """Add a signing entity to a signing group.

    Entity classes: A (Agent), B (Human Operator), C (Policy Engine),
    D (Compliance Verifier), E (Organizational Authority).

    Args:
        config_id: The signing group to add the entity to.
        entity_class: Entity class (A, B, C, D, or E).
        label: Human-readable label for the entity.

    Returns:
        SigningEntityResponse with entity details.

    Example:
        entity = asqav.add_entity("cfg_xxx", entity_class="B", label="operator-1")
    """
    data = _transport._post(
        f"/signing-groups/configs/{config_id}/entities",
        {
            "entity_class": entity_class,
            "label": label,
        },
    )
    return SigningEntityResponse(
        id=data["id"],
        config_id=data["config_id"],
        entity_class=data["entity_class"],
        label=data["label"],
        is_active=data["is_active"],
        created_at=data["created_at"],
    )


def list_entities(config_id: str) -> list[SigningEntityResponse]:
    """List signing entities for a signing group.

    Args:
        config_id: The signing group.

    Returns:
        List of SigningEntityResponse.
    """
    data = _transport._get(f"/signing-groups/configs/{config_id}/entities")
    return [
        SigningEntityResponse(
            id=e["id"],
            config_id=e["config_id"],
            entity_class=e["entity_class"],
            label=e["label"],
            is_active=e["is_active"],
            created_at=e["created_at"],
        )
        for e in data
    ]


def remove_entity(entity_id: str) -> dict[str, Any]:
    """Remove a signing entity (soft delete).

    Args:
        entity_id: The entity to remove.

    Returns:
        Confirmation dict.
    """
    return _transport._delete(f"/signing-groups/entities/{entity_id}")


# --- Group Keypairs ---


def generate_keypair(config_id: str) -> GroupKeypairResponse:
    """Generate a multi-party ML-DSA keypair with Shamir secret sharing.

    Creates a keypair where the private key is split into shares
    distributed to the signing entities in the config.

    Args:
        config_id: The signing group to generate a keypair for.

    Returns:
        GroupKeypairResponse with public key and metadata.

    Example:
        keypair = asqav.generate_keypair("cfg_xxx")
        print(f"Public key: {keypair.public_key_hex[:32]}...")
    """
    data = _transport._post("/signing-groups/keypairs", {"config_id": config_id})
    return _parse_group_keypair(data)


def get_keypair(keypair_id: str) -> GroupKeypairResponse:
    """Get group keypair details (no raw shares).

    Args:
        keypair_id: The keypair ID.

    Returns:
        GroupKeypairResponse with keypair details.
    """
    data = _transport._get(f"/signing-groups/keypairs/{keypair_id}")
    return _parse_group_keypair(data)


def group_sign(
    keypair_id: str,
    message_hex: str,
) -> GroupSignResponse:
    """Perform multi-party ML-DSA signing.

    Requires that the minimum number of entity approvals has been met.
    Output is a standard FIPS 204 ML-DSA-65 signature.

    Args:
        keypair_id: The group keypair to sign with.
        message_hex: The message to sign (hex-encoded).

    Returns:
        GroupSignResponse with signature and verification status.

    Example:
        result = asqav.group_sign("kp_xxx", message_hex="deadbeef")
        assert result.verified  # Standard ML-DSA verifier works
    """
    data = _transport._post(
        f"/signing-groups/keypairs/{keypair_id}/sign",
        {
            "message_hex": message_hex,
        },
    )
    return GroupSignResponse(
        signature_hex=data["signature_hex"],
        message_hex=data["message_hex"],
        keypair_id=data["keypair_id"],
        verified=data["verified"],
        audit_record_id=data.get("audit_record_id"),
    )


def refresh_keypair(keypair_id: str) -> KeyRefreshResponse:
    """Refresh key shares without changing the public key.

    Generates new Shamir shares while preserving the same public key.
    Invalidates all active delegations.

    Args:
        keypair_id: The keypair to refresh.

    Returns:
        KeyRefreshResponse with refresh details.
    """
    data = _transport._post(f"/signing-groups/keypairs/{keypair_id}/refresh", {})
    return KeyRefreshResponse(
        keypair_id=data["keypair_id"],
        refreshed_at=data["refreshed_at"],
        delegations_invalidated=data["delegations_invalidated"],
    )


def recover_share(
    keypair_id: str,
    entity_id: str,
    contributing_entity_ids: list[str],
) -> ShareRecoveryResponse:
    """Recover a lost key share for an entity.

    Requires exactly t contributing entities to reconstruct the share.

    Args:
        keypair_id: The keypair containing the lost share.
        entity_id: The entity whose share is being recovered.
        contributing_entity_ids: Entity IDs providing their shares for recovery.

    Returns:
        ShareRecoveryResponse with recovery details.
    """
    data = _transport._post(
        f"/signing-groups/keypairs/{keypair_id}/recover",
        {
            "entity_id": entity_id,
            "contributing_entity_ids": contributing_entity_ids,
        },
    )
    return ShareRecoveryResponse(
        keypair_id=data["keypair_id"],
        recovered_entity_id=data["recovered_entity_id"],
        recovered_at=data["recovered_at"],
    )


    # Parse API response into a GroupKeypairResponse.
def _parse_group_keypair(data: dict[str, Any]) -> GroupKeypairResponse:
    return GroupKeypairResponse(
        id=data["id"],
        config_id=data["config_id"],
        public_key_hex=data["public_key_hex"],
        min_approvals=data["min_approvals"],
        total_shares=data["total_shares"],
        created_at=data["created_at"],
        status=data.get("status", "active"),
    )
