"""Delegation management."""

from __future__ import annotations

from typing import Any

from . import _transport
from ._models import DelegationResponse

# --- Delegations ---


def create_delegation(
    config_id: str,
    delegator_entity_id: str,
    delegate_entity_id: str,
    expires_in: int = 86400,
) -> DelegationResponse:
    """Create a temporary delegation between Class B entities.

    Allows one entity to delegate their signing authority to another
    with cryptographic expiry.

    Args:
        config_id: The signing group.
        delegator_entity_id: Entity delegating their authority.
        delegate_entity_id: Entity receiving delegated authority.
        expires_in: Delegation duration in seconds (default 24h).

    Returns:
        DelegationResponse with delegation details.

    Example:
        delegation = asqav.create_delegation(
            config_id="cfg_xxx",
            delegator_entity_id="ent_alice",
            delegate_entity_id="ent_bob",
            expires_in=3600,  # 1 hour
        )
    """
    data = _transport._post(
        "/signing-groups/delegations",
        {
            "config_id": config_id,
            "delegator_entity_id": delegator_entity_id,
            "delegate_entity_id": delegate_entity_id,
            "expires_in": expires_in,
        },
    )
    return _parse_delegation(data)


def list_delegations() -> list[DelegationResponse]:
    """List all active delegations.

    Returns:
        List of DelegationResponse.
    """
    data = _transport._get("/signing-groups/delegations")
    return [_parse_delegation(d) for d in data]


def get_delegation(delegation_id: str) -> DelegationResponse:
    """Get a delegation by ID.

    Args:
        delegation_id: The delegation ID.

    Returns:
        DelegationResponse.
    """
    data = _transport._get(f"/signing-groups/delegations/{delegation_id}")
    return _parse_delegation(data)


def revoke_delegation(delegation_id: str) -> dict[str, Any]:
    """Revoke an active delegation.

    Args:
        delegation_id: The delegation to revoke.

    Returns:
        Confirmation dict.
    """
    return _transport._delete(f"/signing-groups/delegations/{delegation_id}")


    # Parse API response into a DelegationResponse.
def _parse_delegation(data: dict[str, Any]) -> DelegationResponse:
    return DelegationResponse(
        id=data["id"],
        config_id=data["config_id"],
        delegator_entity_id=data["delegator_entity_id"],
        delegate_entity_id=data["delegate_entity_id"],
        expires_at=data["expires_at"],
        is_active=data["is_active"],
        created_at=data["created_at"],
    )
