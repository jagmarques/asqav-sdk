"""Manifest + Asqav two-SDK example.

Demonstrates a minimal flow where Manifest routes the LLM call and asqav
signs the resulting agent action with ML-DSA-65 (Compliance Receipt):

    prompt
      -> Manifest /v1/chat/completions (picks the model)
      -> agent decides on an action
      -> asqav signs the action
      -> public verify URL anyone can audit

Requirements:
    pip install asqav openai

Usage:
    ASQAV_API_KEY=sk_... python examples/manifest_example.py \\
        "Wire 1500 EUR to vendor INV-2026-04-12"
"""

from __future__ import annotations

import os
import sys

import asqav

try:
    from openai import OpenAI
except ImportError as err:
    raise SystemExit(
        "openai is required for this example.\n"
        "Install with: pip install openai"
    ) from err


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "noop"

    api_key = os.environ.get("ASQAV_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "Set the ASQAV_API_KEY environment variable before running.\n"
            "Get your key at https://asqav.com"
        )

    manifest = OpenAI(
        base_url=os.environ.get("MANIFEST_BASE_URL", "http://localhost:2099/v1"),
        api_key=os.environ.get("MANIFEST_API_KEY", "dev-api-key-12345"),
    )

    asqav.init(api_key=api_key)
    agent = asqav.Agent.create("finance-bot", capabilities=["wire_transfer"])

    resp = manifest.chat.completions.create(
        model="auto",
        messages=[
            {
                "role": "system",
                "content": "Extract structured wire-transfer details from the user request.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    model = resp.model
    text = resp.choices[0].message.content or ""
    tokens = resp.usage.total_tokens if resp.usage else 0
    cost = float(tokens) * 0.000002  # placeholder unit price

    sig = agent.sign(
        "wire_transfer",
        context={
            "prompt": prompt,
            "extracted": text,
            "model": model,
            "cost_usd": cost,
        },
        receipt_type="protectmcp:decision",
        risk_class="high",
        model_name=model,
    )

    print(f"Manifest routed to: {model}  cost: ${cost:.4f}")
    print(f"Asqav signature:    {sig.signature_id}")
    print(f"Verify:             {sig.verification_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
