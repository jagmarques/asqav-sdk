"""Compliance bundle export for offline audit verification."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

FRAMEWORKS = {
    "eu_ai_act_art12": {
        "name": "EU AI Act - Article 12 Record-Keeping",
        "description": "Logging capabilities assessment per Article 12",
        "version": "2024/1689",
    },
    "eu_ai_act_art14": {
        "name": "EU AI Act - Article 14 Human Oversight",
        "description": "Human oversight assessment per Article 14",
        "version": "2024/1689",
    },
    "dora_ict": {
        "name": "DORA ICT Risk Management",
        "description": "Digital Operational Resilience Act assessment",
        "version": "2022/2554",
    },
    "soc2": {
        "name": "SOC 2 Type II",
        "description": "Service Organization Control audit evidence",
        "version": "2017",
    },
}


def _compute_merkle_root(hashes: list[str]) -> str:
    """Compute Merkle root from a list of hex hashes.

    Standard binary Merkle tree. Odd-length levels duplicate the last hash.
    Returns the empty-string SHA-256 hash when the list is empty.
    """
    if not hashes:
        return hashlib.sha256(b"").hexdigest()

    level = list(hashes)

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])

        next_level: list[str] = []
        for i in range(0, len(level), 2):
            combined = (level[i] + level[i + 1]).encode()
            next_level.append(hashlib.sha256(combined).hexdigest())
        level = next_level

    return level[0]


def _receipt_hash(receipt: dict) -> str:
    """Deterministic hash for a single receipt."""
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _normalize_signature(sig: Any) -> dict:
    """Convert a SignatureResponse, SignedActionResponse, or dict to a receipt dict."""
    if isinstance(sig, dict):
        return sig

    # Dataclass - pull known fields
    result: dict[str, Any] = {}
    for attr in (
        "signature_id",
        "signature",
        "action_id",
        "action_type",
        "timestamp",
        "signed_at",
        "verification_url",
        "agent_id",
        "payload",
        "algorithm",
        "signature_preview",
    ):
        if hasattr(sig, attr):
            result[attr] = getattr(sig, attr)
    return result


@dataclass
class ComplianceBundle:
    """Self-contained compliance bundle for offline audit verification."""

    framework: str
    framework_metadata: dict = field(default_factory=dict)
    generated_at: str = ""
    merkle_root: str = ""
    receipt_count: int = 0
    receipts: list[dict] = field(default_factory=list)
    verification: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize bundle to a plain dictionary."""
        return {
            "framework": self.framework,
            "framework_metadata": self.framework_metadata,
            "generated_at": self.generated_at,
            "merkle_root": self.merkle_root,
            "receipt_count": self.receipt_count,
            "receipts": self.receipts,
            "verification": self.verification,
        }

    def to_json(self) -> str:
        """Serialize bundle to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_file(self, path: str) -> None:
        """Write bundle JSON to a file."""
        with open(path, "w") as f:
            f.write(self.to_json())


def export_bundle(
    signatures: list,
    framework: str = "eu_ai_act_art12",
) -> ComplianceBundle:
    """Package signatures into a compliance bundle with Merkle root.

    Args:
        signatures: List of SignatureResponse, SignedActionResponse, or dicts.
        framework: Key from FRAMEWORKS dict identifying the compliance standard.

    Returns:
        A ComplianceBundle ready for export.

    Raises:
        ValueError: If the framework key is not recognized.
    """
    if framework not in FRAMEWORKS:
        raise ValueError(
            f"Unknown framework {framework!r}. "
            f"Valid options: {', '.join(sorted(FRAMEWORKS))}"
        )

    receipts = [_normalize_signature(s) for s in signatures]
    hashes = [_receipt_hash(r) for r in receipts]
    merkle_root = _compute_merkle_root(hashes)

    now = datetime.now(timezone.utc).isoformat()

    return ComplianceBundle(
        framework=framework,
        framework_metadata=FRAMEWORKS[framework],
        generated_at=now,
        merkle_root=merkle_root,
        receipt_count=len(receipts),
        receipts=receipts,
        verification={
            "merkle_root": merkle_root,
            "hash_algorithm": "sha256",
            "receipt_hashes": hashes,
        },
    )
