"""Multi-provider budget enforcement with signed spend records.

A pipeline that calls GPT-4 for reasoning and Claude for code generation
shares a single ceiling. Every spend is signed via agent.sign() so the
budget trail is independently verifiable.

Run:
    export ASQAV_API_KEY=sk_...
    python examples/budget_tracking.py

Or do the same thing from the shell:
    asqav budget check --agent-id <id> --limit 10 --estimated-cost 0.25 --current-spend 4.20
    asqav budget record --agent-id <id> --action api:openai --actual-cost 0.23 --limit 10 --current-spend 4.20

Closes #61.
"""

from __future__ import annotations

import asqav

# Per-call cost lookups for the example. Real pipelines pull these from the
# provider response (e.g. response.usage.total_tokens * price_per_token).
PRICE_PER_CALL = {
    "openai/gpt-4-reason": 0.30,
    "anthropic/claude-codegen": 0.18,
    "google/gemini-fastpath": 0.04,
}


def main() -> None:
    asqav.init()

    agent = asqav.Agent.create("multi-provider-pipeline")
    budget = asqav.BudgetTracker(agent, limit=10.00, currency="USD")

    plan = [
        ("openai/gpt-4-reason", "draft a strategy"),
        ("anthropic/claude-codegen", "implement step 1"),
        ("anthropic/claude-codegen", "implement step 2"),
        ("google/gemini-fastpath", "summarize the run"),
    ]

    for action_type, description in plan:
        estimated = PRICE_PER_CALL[action_type]
        decision = budget.check(estimated)
        if not decision.allowed:
            print(f"DENIED {action_type}: {decision.reason}")
            print(
                f"  current_spend={decision.current_spend:.2f} "
                f"limit={decision.limit:.2f}"
            )
            break

        # ... actually execute the call here ...
        actual = estimated  # for the example, assume estimate matches reality

        sig = budget.record(
            action_type,
            actual_cost=actual,
            context={"description": description},
        )
        print(
            f"OK    {action_type:32s} ${actual:.2f}  "
            f"spend=${budget._spend:.2f}/{budget.limit:.2f}  sig={sig.signature_id}"
        )

    print(f"\nFinal spend: ${budget._spend:.2f} of ${budget.limit:.2f}")
    print("Every entry is a signed record on asqav, replay-verifiable from the audit trail.")


if __name__ == "__main__":
    main()
