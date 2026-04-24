"""Example: sign prompt + reasoning trace + output for an AI agent decision.

Only SHA-256 hashes of the prompt, reasoning trace, and output are posted
to the Asqav API by default. Pass ``store_raw=True`` with ``retention_days``
to additionally persist the raw payloads on Asqav for the given retention
window.

Usage:
    ASQAV_API_KEY=sk_... python examples/reasoning_trace_example.py
"""

from __future__ import annotations

import os

import asqav
from asqav import sign_reasoning


def main() -> None:
    api_key = os.environ.get("ASQAV_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set the ASQAV_API_KEY environment variable before running this example.\n"
            "Get your key at https://asqav.com"
        )

    asqav.init(api_key=api_key)
    agent = asqav.Agent.create("research-crawler")

    prompt = "Summarise the most recent quarterly risk report."
    trace = (
        "Step 1: identified the quarterly risk report for Q1 2026.\n"
        "Step 2: extracted top five risks.\n"
        "Step 3: drafted the summary."
    )
    output = "Top risks: concentration, liquidity, operational, cyber, regulatory."

    receipt = sign_reasoning(
        agent,
        prompt=prompt,
        trace=trace,
        output=output,
    )

    print("signed reasoning receipt:")
    print(f"  signature_id: {receipt.signature_id}")
    print(f"  verification_url: {receipt.verification_url}")
    print(f"  prompt_hash: {receipt.prompt_hash}")
    print(f"  trace_hash: {receipt.trace_hash}")
    print(f"  output_hash: {receipt.output_hash}")
    print(f"  ts: {receipt.signature.timestamp}")
    print(f"  raw_stored: {receipt.raw_stored}")


if __name__ == "__main__":
    main()
