"""Compliance bundle export for offline audit verification.

:func:`export_bundle` packages receipts client-side with a Merkle root;
:func:`fetch_audit_pack` calls the cloud's signed ``/audit-pack/export``.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

FRAMEWORKS = {
    "eu_ai_act": {
        "name": "EU AI Act",
        "description": "Articles 12 + 26 record-keeping and human oversight",
        "version": "Regulation (EU) 2024/1689",
    },
    "dora": {
        "name": "DORA",
        "description": "Digital Operational Resilience Act, Article 17",
        "version": "Regulation (EU) 2022/2554",
    },
    "nydfs_500": {
        "name": "NYDFS Part 500",
        "description": "23 NYCRR Part 500 audit trail and incident notice",
        "version": "23 NYCRR 500",
    },
    "colorado_ai": {
        "name": "Colorado AI Act",
        "description": "SB 24-205 High-Risk AI System deployer obligations",
        "version": "SB 24-205",
    },
    "texas_traiga": {
        "name": "Texas TRAIGA",
        "description": "HB 149 intent-based liability and prohibited-use vocab",
        "version": "HB 149",
    },
    "nist_ai_rmf": {
        "name": "NIST AI RMF",
        "description": "Voluntary GOVERN / MAP / MEASURE / MANAGE framework",
        "version": "NIST AI 100-1",
    },
    "circia": {
        "name": "CIRCIA",
        "description": "Covered Cyber Incident reporting for critical infrastructure",
        "version": "Public Law 117-103",
    },
    "hipaa_security": {
        "name": "HIPAA Security Rule",
        "description": "45 CFR 164.312(b) audit controls",
        "version": "45 CFR 164",
    },
    "sec_17a4a": {
        "name": "SEC 17 CFR 240.17a-4(a)",
        "description": "Broker-dealer 6-year retention",
        "version": "17 CFR 240.17a-4(a)",
    },
    "sec_17a4b": {
        "name": "SEC 17 CFR 240.17a-4(b)",
        "description": "Broker-dealer 3-year retention",
        "version": "17 CFR 240.17a-4(b)",
    },
}

def _compute_merkle_root(hashes: list[str]) -> str:
    """Compute Merkle root from hex hashes (binary tree, odd levels duplicate the tail)."""
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
    framework: str = "eu_ai_act",
) -> ComplianceBundle:
    """Package signatures into a :class:`ComplianceBundle` with a Merkle root.

    Raises ``ValueError`` for unknown framework keys.
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


def fetch_audit_pack(
    *,
    start: str,
    end: str,
    agent_id: str | None = None,
    only_compliance: bool = True,
) -> dict[str, Any]:
    """Fetch the cloud-signed Audit Pack for a window via ``POST /audit-pack/export``.

    Returns the cloud response dict (bundle_digest, bundle_signature, receipts,
    regime_mapping, revocation_manifest, algorithm_registry_version).
    """
    from . import client as _client

    if _client._api_key is None:
        raise RuntimeError("asqav.init(api_key=...) must be called first.")
    if _client.httpx is None:
        raise RuntimeError("httpx is required; install with `pip install asqav[http]`.")

    url = _client._api_base.rstrip("/") + "/api/v1/audit-pack/export"
    body = {"start": start, "end": end, "only_compliance": only_compliance}
    if agent_id is not None:
        body["agent_id"] = agent_id

    from ._useragent import USER_AGENT

    resp = _client.httpx.post(
        url,
        json=body,
        headers={
            "Authorization": f"Bearer {_client._api_key}",
            "User-Agent": USER_AGENT,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()
