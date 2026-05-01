"""End-to-end quarterly audit: bundle, replay, verify, summarize.

Auditors get a self-contained file (no API access required) that contains
every signed action over the period plus a Merkle root. Replay reconstructs
the timeline; chain verification proves the records were not tampered with
after the fact.

Run:
    export ASQAV_API_KEY=sk_...
    python examples/quarterly_audit.py

Same flow at the terminal:
    asqav compliance export --session sess_q3 --framework eu_ai_act_art12 --output q3-bundle.json
    asqav replay --bundle q3-bundle.json

Closes #63.
"""

from __future__ import annotations

from asqav.client import get_session_signatures
from asqav.compliance import export_bundle
from asqav.replay import replay_from_bundle


def main() -> None:
    import asqav
    asqav.init()

    # Replace these with the actual identifiers for the audit period.
    session_id = "sess_q3_2026"
    framework = "eu_ai_act_art12"
    bundle_path = "q3-bundle.json"

    print(f"Step 1. Fetch signatures for {session_id}")
    sigs = get_session_signatures(session_id)
    print(f"        pulled {len(sigs)} signed actions")

    print(f"\nStep 2. Export {framework} bundle to {bundle_path}")
    bundle = export_bundle(sigs, framework=framework)
    bundle.to_file(bundle_path)
    print(f"        receipts={bundle.receipt_count}")
    print(f"        merkle_root={bundle.merkle_root}")

    print("\nStep 3. Replay the timeline from the bundle (offline, no API)")
    timeline = replay_from_bundle(bundle)
    print(timeline.summary())

    print("\nStep 4. Re-verify the SHA-256 chain")
    chain_ok = timeline.verify_chain()
    if chain_ok:
        print("        chain verified, the audit trail has not been tampered with")
    else:
        print("        chain BROKEN, this bundle cannot be trusted")
        for step in timeline.steps:
            if not step.chain_valid:
                print(f"        - step {step.index} ({step.signature_id}) failed")

    print("\nStep 5. Summary line for the audit report")
    print(
        f"  Period: {session_id} | Framework: {framework} | "
        f"Actions: {bundle.receipt_count} | "
        f"Chain integrity: {'OK' if chain_ok else 'BROKEN'} | "
        f"Merkle root: {bundle.merkle_root}"
    )


if __name__ == "__main__":
    main()
