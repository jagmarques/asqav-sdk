"""Pause high-risk agent actions for human review and approval.

Some calls are too risky to execute automatically: data deletion, customer
emails, payment processing. The pattern: request a signing session, wait for
human approval, then proceed only when the session is fully signed.

Run:
    export ASQAV_API_KEY=sk_...
    python examples/human_approval.py

Same flow at the terminal:
    asqav approve <session_id> <entity_id>     # human signs off

Closes #64.
"""

from __future__ import annotations

import time

import asqav
from asqav.client import get_action_status, request_action

HIGH_RISK_PATTERNS = {
    "sql-destructive",
    "payment:process",
    "email:send-customer",
}


def needs_approval(action_type: str) -> bool:
    """Trivial classifier. Real systems read this from a policy table."""
    return action_type in HIGH_RISK_PATTERNS


def wait_for_approval(session_id: str, timeout_s: int = 300) -> bool:
    """Poll the session until approved or timed out. Approval happens out-of-band
    (a reviewer calls asqav.approve_action(session_id, entity_id) or runs
    `asqav approve <session_id> <entity_id>`)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        session = get_action_status(session_id)
        if session.status == "approved":
            return True
        if session.status in {"denied", "expired"}:
            return False
        time.sleep(2)
    return False


def main() -> None:
    asqav.init()

    agent = asqav.Agent.create("review-required-agent")
    action_type = "payment:process"
    params = {"amount_usd": 4500, "vendor": "vendor_42", "reference": "PO-7711"}

    if needs_approval(action_type):
        print(f"{action_type} is high-risk, requesting human approval")
        session = request_action(
            agent_id=agent.agent_id,
            action_type=action_type,
            params=params,
        )
        print(f"  session_id: {session.session_id}")
        print(f"  needs {session.approvals_required} approvals")
        print(
            f"  share the session id with the reviewer; they sign with:\n"
            f"    asqav approve {session.session_id} <entity_id>"
        )

        if not wait_for_approval(session.session_id):
            print("approval not granted, aborting")
            return

        print("approved, proceeding")
        sig = agent.sign(
            action_type,
            params,
            counterparty={"endpoint": "vendor_42.example.com"},
        )
        # The signing session id ties the executed signature back to the
        # human approval, which is what an auditor reads later.
        print(f"signed {sig.signature_id} (authorization_ref={sig.authorization_ref})")
    else:
        sig = agent.sign(action_type, params)
        print(f"low-risk, signed inline as {sig.signature_id}")


if __name__ == "__main__":
    main()
