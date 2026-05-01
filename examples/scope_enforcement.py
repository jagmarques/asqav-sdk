"""Catch out-of-scope agent actions before they execute.

Define an agent's allowed actions, then run a preflight check against each
action it's about to perform. If the policy blocks the call, the example
records the violation as a signed audit entry so the attempt is permanent.

Run:
    export ASQAV_API_KEY=sk_...
    python examples/scope_enforcement.py

Same flow at the terminal:
    asqav preflight <agent_id> data:read       # exits 0 if cleared
    asqav preflight <agent_id> sql-destructive # exits 1 with reasons listed

Closes #62.
"""

from __future__ import annotations

import asqav

# Pretend an external orchestrator just told the agent to do these.
ATTEMPTED_ACTIONS = [
    "data:read",
    "api:call",
    "sql-destructive",     # likely blocked by the org's destructive-sql policy
    "filesystem:write:*",  # likely outside the agent's scope
]


def main() -> None:
    asqav.init()

    agent = asqav.Agent.create("scope-demo-agent")

    for action_type in ATTEMPTED_ACTIONS:
        result = agent.preflight(action_type)
        if result.cleared:
            print(f"CLEARED  {action_type:24s}  {result.explanation}")
            sig = agent.sign(action_type, {"source": "scope-demo"})
            print(f"         executed and signed as {sig.signature_id}")
            continue

        print(f"BLOCKED  {action_type:24s}")
        for reason in result.reasons:
            print(f"           - {reason}")

        # Sign the violation itself so the attempt is permanent. The
        # action_type is namespaced under 'audit:' so the org's policy
        # engine doesn't conflate it with the original blocked call.
        violation_sig = agent.sign(
            "audit:scope_violation",
            {
                "attempted": action_type,
                "reasons": result.reasons,
                "policy_allowed": result.policy_allowed,
                "agent_active": result.agent_active,
            },
        )
        print(f"           audit record signed as {violation_sig.signature_id}")


if __name__ == "__main__":
    main()
