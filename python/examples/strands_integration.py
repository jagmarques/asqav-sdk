"""Strands Agents integration example for asqav.

Demonstrates how to attach asqav governance hooks to a Strands ``Agent`` so
every model call is signed and recorded in the audit trail.

Requirements:
    pip install asqav[strands]

Usage:
    ASQAV_API_KEY=sk_... python examples/strands_integration.py
"""

from __future__ import annotations

import os

import asqav
from asqav.extras.strands import AsqavStrandsHooks

try:
    from strands import Agent
except ImportError as err:
    raise SystemExit(
        "strands-agents is required for this example.\n"
        "Install with: pip install asqav[strands]"
    ) from err

# === Initialise asqav ===

api_key = os.environ.get("ASQAV_API_KEY", "")
if not api_key:
    raise SystemExit(
        "Set the ASQAV_API_KEY environment variable before running this example.\n"
        "Get your key at https://asqav.com"
    )

asqav.init(api_key=api_key)

# === Build the governance hook and attach it to a Strands Agent ===

# AsqavStrandsHooks implements Strands' HookProvider protocol: its BeforeModelCallEvent
# and AfterModelCallEvent callbacks auto-sign every model call made through this agent.
hooks = AsqavStrandsHooks(agent_name="strands-demo")

# Strands picks up the model from the environment (AWS credentials for Bedrock, or any
# configured provider): https://strandsagents.com/latest/documentation/docs/user-guide/
agent = Agent(hooks=[hooks])

# === Run a query - every model call is signed automatically ===

print("Running Strands agent with asqav governance...")
result = agent("What is the capital of France? Answer in one short sentence.")

print("\nAgent result:")
print(result)

# === Inspect the signed audit trail ===

print(f"\nSigned {len(hooks._signatures)} model call(s) for this run:")
for sig in hooks._signatures:
    print(f"  - signature_id={sig.signature_id} verify={sig.verification_url}")

print("\nView the full audit trail at https://asqav.com/dashboard")
