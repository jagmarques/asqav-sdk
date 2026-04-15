"""Three-phase signing: intent, decision, execution.

Instead of a single post-action receipt, sign at each phase of an action
for a complete decision trail. Each receipt links to the previous via parent_id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class PhaseChain:
    """Three linked receipts covering an action's full lifecycle."""

    intent: Any  # SignatureResponse from intent phase
    decision: Any  # PreflightResult from decision phase
    execution: Any  # SignatureResponse from execution phase
    trace_id: str

    @property
    def approved(self) -> bool:
        """Whether the decision phase approved the action."""
        return self.decision.cleared if self.decision else False

    def to_dict(self) -> dict:
        """Serialize the chain to a plain dict."""

        def _sig_dict(sig: Any) -> dict | None:
            if sig is None:
                return None
            return {
                "signature_id": sig.signature_id,
                "action_id": sig.action_id,
                "timestamp": sig.timestamp,
                "verification_url": sig.verification_url,
            }

        def _decision_dict(dec: Any) -> dict | None:
            if dec is None:
                return None
            return {
                "cleared": dec.cleared,
                "agent_active": dec.agent_active,
                "policy_allowed": dec.policy_allowed,
                "reasons": list(dec.reasons),
                "explanation": dec.explanation,
            }

        return {
            "trace_id": self.trace_id,
            "intent": _sig_dict(self.intent),
            "decision": _decision_dict(self.decision),
            "execution": _sig_dict(self.execution),
            "approved": self.approved,
        }

    def to_json(self) -> str:
        """Serialize the chain to a JSON string."""
        return json.dumps(self.to_dict())


def sign_with_phases(
    agent: Any,
    action_type: str,
    context: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> PhaseChain:
    """Execute the full three-phase signing flow.

    Phase 1 (Intent): Sign what the agent wants to do
    Phase 2 (Decision): Preflight check - policy evaluates
    Phase 3 (Execution): Sign what actually happened (only if approved)

    Each phase produces a receipt linked to the previous one.
    """
    from .client import generate_trace_id

    tid = trace_id or generate_trace_id()

    # Phase 1: Intent
    intent_sig = agent.sign(
        f"{action_type}:intent",
        context,
        trace_id=tid,
    )

    # Phase 2: Decision
    decision = agent.preflight(action_type)

    # Phase 3: Execution (only if approved)
    exec_sig = None
    if decision.cleared:
        exec_sig = agent.sign(
            f"{action_type}:execution",
            context,
            trace_id=tid,
            parent_id=intent_sig.signature_id if intent_sig else None,
        )

    return PhaseChain(
        intent=intent_sig,
        decision=decision,
        execution=exec_sig,
        trace_id=tid,
    )
