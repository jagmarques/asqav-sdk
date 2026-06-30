"""Structured receipts with an optional context schema.

When you pass ``context_schema`` to ``Agent.sign()``, the SDK validates and
normalises the ``context`` dict BEFORE signing, so every receipt in your
audit trail has the same key order and the cloud can index them consistently.

If the context does not match the schema, ``AsqavValidationError`` is raised
BEFORE any network call, so no token is consumed and no partial record is
written.

Requirements:
    pip install asqav

Usage:
    ASQAV_API_KEY=sk_... python examples/schema_receipts_example.py
"""

from __future__ import annotations

import os

import asqav
from asqav import AsqavValidationError

# Schema: field -> {type: string|number|boolean|object|array, required?: bool}
PAYMENT_SCHEMA = {
    "user_id": {"type": "string", "required": True},
    "amount": {"type": "number", "required": True},
    "currency": {"type": "string", "required": True},
    "metadata": {"type": "object"},  # optional
}


def main() -> int:
    api_key = os.environ.get("ASQAV_API_KEY", "")
    if not api_key:
        print(
            "Set ASQAV_API_KEY before running.\n"
            "Get your key at https://asqav.com"
        )
        return 1

    asqav.init(api_key=api_key)
    agent = asqav.Agent.create("schema-demo")

    # --- Valid context: passes validation, keys are sorted before signing ---
    print("Signing with a valid, schema-validated context...")
    sig = agent.sign(
        "payment:initiate",
        {
            # Deliberately unsorted: SDK normalises to alphabetical key order.
            "user_id": "user_abc123",
            "currency": "EUR",
            "amount": 49.95,
            "metadata": {"source": "api"},
        },
        context_schema=PAYMENT_SCHEMA,
    )
    print(f"  signature_id : {sig.signature_id}")
    print(f"  verify       : {sig.verification_url}")

    # --- Invalid context: raises BEFORE the network call ---
    print("\nAttempting to sign with a missing required field (user_id)...")
    try:
        agent.sign(
            "payment:initiate",
            {"amount": 100.0, "currency": "USD"},  # user_id missing
            context_schema=PAYMENT_SCHEMA,
        )
    except AsqavValidationError as exc:
        print(f"  Caught (expected): {exc}")
        print(f"  Docs: {exc.docs_url}")

    # --- Wrong type: raises BEFORE the network call ---
    print("\nAttempting to sign with the wrong type for 'amount'...")
    try:
        agent.sign(
            "payment:initiate",
            {"user_id": "u1", "amount": "one-hundred", "currency": "USD"},
            context_schema=PAYMENT_SCHEMA,
        )
    except AsqavValidationError as exc:
        print(f"  Caught (expected): {exc}")

    # --- Custom callable validator (e.g. wrap jsonschema for full JSON Schema) ---
    print("\nSigning with a custom callable validator...")

    def my_validator(ctx: dict) -> None:
        if ctx.get("amount", 0) <= 0:
            raise ValueError("amount must be positive")

    sig2 = agent.sign(
        "payment:initiate",
        {"user_id": "user_xyz", "amount": 10.00, "currency": "GBP"},
        context_schema=my_validator,
    )
    print(f"  signature_id : {sig2.signature_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
