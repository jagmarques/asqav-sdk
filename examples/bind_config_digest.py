"""Bind your agent's declared configuration into the signed receipt.

Hash whatever declares what the agent is allowed to be (an agent manifest,
an AGENTS.md, a container image digest, an SBOM) with SHA-256, put the digest
in the sign context, and pin it with a context schema. The digest then lands
inside the cryptographically signed receipt, so anyone can later confirm which
declared configuration produced a given action at the public verify endpoint.

This uses the existing schema-validated sign context. No new wire field.

Run:
    export ASQAV_API_KEY=sk_...
    python examples/bind_config_digest.py

Same flow at the terminal:
    asqav sign payment:refund --context '{"config_digest":"sha256:...","config_kind":"agent-manifest"}'
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import asqav

# The declaration you want bound to every receipt this agent signs. In a real
# deployment this is your AGENTS.md, a manifest, or an image digest on disk.
MANIFEST_PATH = Path(__file__).with_name("agent-manifest.example.json")

# Pin the binding keys so a receipt that claims a config always carries both
# the digest and what kind of artifact it hashes. Extra context keys pass through.
CONFIG_BINDING_SCHEMA = {
    "config_digest": {"type": "string", "required": True},
    "config_kind": {"type": "string", "required": True},
}


def config_digest(path: Path) -> str:
    """Return ``sha256:<hex>`` over the raw bytes of a declaration file."""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    asqav.init()

    agent = asqav.Agent.create("config-binding-demo")

    digest = config_digest(MANIFEST_PATH)
    print(f"declaration : {MANIFEST_PATH.name}")
    print(f"config_digest: {digest}")

    # Bind the digest into the action the agent is signing. Anyone verifying
    # the receipt sees exactly which declared config authorized this action.
    sig = agent.sign(
        "payment:refund",
        {
            "amount": 49.95,
            "currency": "EUR",
            "config_digest": digest,
            "config_kind": "agent-manifest",
            "config_ref": MANIFEST_PATH.name,
        },
        context_schema=CONFIG_BINDING_SCHEMA,
    )

    print(f"signature_id : {sig.signature_id}")
    print(f"verify       : {sig.verification_url}")


if __name__ == "__main__":
    main()
