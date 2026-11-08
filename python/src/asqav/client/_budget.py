"""Client-side spend budget tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import Agent, SignatureResponse


    # Result of a budget pre-check for an agent action.
@dataclass
class BudgetCheckResult:

    allowed: bool
    current_spend: float
    limit: float
    remaining: float
    reason: str | None = None


class BudgetTracker:
    """Client-side spend tracker over signed `agent.sign()` records.
    Tamper-evident and replayable; fails closed; the caller enforces."""

    def __init__(self, agent: Agent, limit: float, currency: str = "USD"):
        if limit < 0:
            raise ValueError("budget limit must be non-negative")
        self.agent = agent
        self.limit = float(limit)
        self.currency = currency
        self._spend: float = 0.0
        self._records: list[str] = []  # signature_ids

    def record(
        self,
        action_type: str,
        actual_cost: float,
        context: dict[str, Any] | None = None,
    ) -> SignatureResponse:
        """Record an action cost by signing it. Updates cumulative spend.

        The signature payload includes the cost, currency, resulting
        cumulative spend, and the configured limit, so the trail can
        be independently replayed and verified.

        Args:
            action_type: Type of action being recorded (e.g. "api:openai").
            actual_cost: Realized cost of the action.
            context: Extra context to include in the signed payload.

        Returns:
            SignatureResponse from the underlying sign() call.
        """
        cost = float(actual_cost)
        if cost != cost or cost in (float("inf"), float("-inf")) or cost < 0:
            raise ValueError("actual_cost must be a finite non-negative number")

        self._spend += cost

        payload: dict[str, Any] = {
            "cost": cost,
            "currency": self.currency,
            "cumulative_spend": self._spend,
            "limit": self.limit,
        }
        if context:
            payload.update(context)

        sig = self.agent.sign(f"budget:{action_type}", payload, result=dict(payload))
        self._records.append(sig.signature_id)
        return sig
