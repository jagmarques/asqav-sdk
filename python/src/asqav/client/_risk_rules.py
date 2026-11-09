"""Risk rule management."""

from __future__ import annotations

from typing import Any

from . import _transport
from ._models import RiskRuleResponse

# --- Risk Rules ---


def create_risk_rule(
    name: str,
    action_pattern: str,
    risk_level: str,
    approval_override: int | None = None,
    priority: int = 0,
    entity_weights: dict[str, float] | None = None,
    time_schedule: dict[str, Any] | None = None,
) -> RiskRuleResponse:
    """Create a risk classification rule.

    Risk rules dynamically adjust approval requirements based on action patterns.

    Args:
        name: Rule name.
        action_pattern: Glob pattern to match action types.
        risk_level: Risk level (low, medium, high, critical).
        approval_override: Override the default approval count for matching actions.
        priority: Rule priority (higher = evaluated first).
        entity_weights: Per-class weight multipliers (e.g., {"D": 2.0}).
        time_schedule: Time-dependent approval adjustment.

    Returns:
        RiskRuleResponse with rule details.

    Example:
        rule = asqav.create_risk_rule(
            name="High-risk finance",
            action_pattern="finance.*",
            risk_level="critical",
            approval_override=3,
            priority=100,
        )
    """
    body: dict[str, Any] = {
        "name": name,
        "action_pattern": action_pattern,
        "risk_level": risk_level,
        "priority": priority,
    }
    if approval_override is not None:
        body["approval_override"] = approval_override
    if entity_weights is not None:
        body["entity_weights"] = entity_weights
    if time_schedule is not None:
        body["time_schedule"] = time_schedule
    data = _transport._post("/risk-rules", body)
    return _parse_risk_rule(data)


def list_risk_rules() -> list[RiskRuleResponse]:
    """List all risk rules for the organization.

    Returns:
        List of RiskRuleResponse.
    """
    data = _transport._get("/risk-rules")
    return [_parse_risk_rule(r) for r in data]


def get_risk_rule(rule_id: str) -> RiskRuleResponse:
    """Get a risk rule by ID.

    Args:
        rule_id: The rule ID.

    Returns:
        RiskRuleResponse with rule details.
    """
    data = _transport._get(f"/risk-rules/{rule_id}")
    return _parse_risk_rule(data)


def update_risk_rule(
    rule_id: str,
    **kwargs: Any,
) -> RiskRuleResponse:
    """Update a risk rule.

    Args:
        rule_id: The rule to update.
        **kwargs: Fields to update (name, action_pattern, risk_level,
                  approval_override, priority, entity_weights, time_schedule).

    Returns:
        RiskRuleResponse with updated rule.
    """
    data = _transport._put(f"/risk-rules/{rule_id}", kwargs)
    return _parse_risk_rule(data)


def delete_risk_rule(rule_id: str) -> dict[str, Any]:
    """Delete a risk rule.

    Args:
        rule_id: The rule to delete.

    Returns:
        Confirmation dict.
    """
    return _transport._delete(f"/risk-rules/{rule_id}")


    # Parse API response into RiskRuleResponse.
def _parse_risk_rule(data: dict[str, Any]) -> RiskRuleResponse:
    return RiskRuleResponse(
        id=data["id"],
        name=data["name"],
        action_pattern=data["action_pattern"],
        risk_level=data["risk_level"],
        approval_override=data.get("approval_override"),
        priority=data["priority"],
        entity_weights=data.get("entity_weights"),
        time_schedule=data.get("time_schedule"),
        created_at=data.get("created_at", ""),
    )
