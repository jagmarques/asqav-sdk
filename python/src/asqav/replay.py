"""Audit trail replay - reconstruct and verify what any agent did, step by step.

Two chain shapes are supported:

  * **IETF v2** (default): ``previousReceiptHash = sha256(JCS(predecessor
    signed envelope))`` per draft-marques-asqav-compliance-receipts-00
    §5.7. First record's predecessor seed is ``"0" * 64``
    (``FIRST_RECEIPT_SEED``).
  * **Legacy**: hash over ``{signature_id, action_type, timestamp,
    prev_hash}`` with empty-string seed. Kept behind ``legacy_chain=True``
    so historical receipts (issued before the IETF profile landed) keep
    verifying.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ._jcs import canonical_json
from .client import SignedActionResponse, get_session_signatures
from .compliance import ComplianceBundle, _normalize_signature

# Per-IETF-profile seed for the first record on every chain. Distinguishes
# "no predecessor" from "predecessor not yet linked". Mirrors
# `core/integrity.py:FIRST_RECEIPT_SEED` on the cloud.
FIRST_RECEIPT_SEED: str = "0" * 64


@dataclass
class ReplayStep:
    """Single step in an audit trail replay."""

    index: int
    action_type: str
    context: dict[str, Any] | None
    timestamp: float
    signature_id: str
    verification_url: str
    chain_valid: bool
    explanation: str
    # Predecessor's chain hash; verify_chain recomputes and compares.
    prev_chain_hash: str | None = None


@dataclass
class ReplayTimeline:
    """Complete audit trail timeline for a session."""

    agent_id: str
    session_id: str
    steps: list[ReplayStep] = field(default_factory=list)
    chain_integrity: bool = True
    start_time: float | None = None
    end_time: float | None = None
    total_actions: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize timeline to a plain dictionary."""
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "steps": [
                {
                    "index": s.index,
                    "action_type": s.action_type,
                    "context": s.context,
                    "timestamp": s.timestamp,
                    "signature_id": s.signature_id,
                    "verification_url": s.verification_url,
                    "chain_valid": s.chain_valid,
                    "explanation": s.explanation,
                    "prev_chain_hash": s.prev_chain_hash,
                }
                for s in self.steps
            ],
            "chain_integrity": self.chain_integrity,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_actions": self.total_actions,
        }

    def to_json(self) -> str:
        """Serialize timeline to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_file(self, path: str) -> None:
        """Write timeline JSON to a file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    def summary(self) -> str:
        """Human-readable multi-line summary of the timeline."""
        lines = [
            "Audit Trail Replay",
            f"  Agent: {self.agent_id}",
            f"  Session: {self.session_id}",
            f"  Actions: {self.total_actions}",
            f"  Chain integrity: {'PASS' if self.chain_integrity else 'FAIL'}",
        ]

        if self.start_time is not None:
            start_dt = datetime.fromtimestamp(self.start_time, tz=timezone.utc)
            lines.append(f"  Start: {start_dt.isoformat()}")
        if self.end_time is not None:
            end_dt = datetime.fromtimestamp(self.end_time, tz=timezone.utc)
            lines.append(f"  End: {end_dt.isoformat()}")

        if self.steps:
            lines.append("")
            lines.append("  Steps:")
            for step in self.steps:
                chain_mark = "ok" if step.chain_valid else "BROKEN"
                lines.append(
                    f"    [{step.index}] {step.explanation} "
                    f"(chain: {chain_mark})"
                )

        return "\n".join(lines)

    def verify_chain(self, *, legacy_chain: bool = False) -> bool:
        """Recompute each step's predecessor hash, compare to stored
        ``prev_chain_hash``. Mismatch = tampering. Steps lacking a stored
        hash are marked invalid rather than passing silently.

        Args:
            legacy_chain: When True, recomputes using the pre-IETF
                envelope shape (``{signature_id, action_type, timestamp,
                prev_hash}``) so receipts issued before the profile
                landed keep verifying. Default False uses the IETF v2
                shape per §5.7.
        """
        if not self.steps:
            self.chain_integrity = True
            return True

        seed = "" if legacy_chain else FIRST_RECEIPT_SEED
        prev_hash = seed
        all_valid = True

        for step in self.steps:
            if step.index == 0:
                # First-record seed: stored prev_chain_hash should match the
                # spec seed under v2. Legacy chains used None so we accept
                # both for compatibility with bundles produced pre-v2.
                step.chain_valid = True
            else:
                stored = getattr(step, "prev_chain_hash", None)
                if stored is None:
                    step.chain_valid = False
                    all_valid = False
                else:
                    step.chain_valid = stored == prev_hash
                    if not step.chain_valid:
                        all_valid = False

            prev_hash = _step_chain_hash(
                step, prev_hash, legacy_chain=legacy_chain
            )

        self.chain_integrity = all_valid
        return all_valid


def _step_chain_hash(
    step: ReplayStep, prev_hash: str, *, legacy_chain: bool
) -> str:
    """Compute the chain hash a step contributes to the next link.

    Under the IETF v2 shape (``draft-marques-asqav-compliance-receipts-00``
    §5.7) this is ``sha256(JCS(predecessor_signed_envelope))``. The
    "envelope" reconstructed here is the minimum the SDK can rebuild from
    a ReplayStep: the signature_id, action_type, payload context, and
    server timestamp, plus the previous-receipt-hash link. This is the
    same shape the cloud serializes when it computes the chain link, so
    SDK and cloud agree byte-for-byte for receipts the SDK observed.

    Under ``legacy_chain=True`` the historic shape is used so bundles
    exported before the profile landed still verify.
    """
    if legacy_chain:
        entry = json.dumps(
            {
                "signature_id": step.signature_id,
                "action_type": step.action_type,
                "timestamp": step.timestamp,
                "prev_hash": prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(entry.encode()).hexdigest()

    envelope = {
        "signature_id": step.signature_id,
        "action_type": step.action_type,
        "context": step.context,
        "timestamp": step.timestamp,
        "previousReceiptHash": prev_hash,
    }
    return hashlib.sha256(canonical_json(envelope)).hexdigest()


def _verify_hash_chain(
    signatures: list[dict], *, legacy_chain: bool = False
) -> list[bool]:
    """Check each signature chains to the previous via SHA-256.

    Returns a list of booleans, one per signature, indicating whether
    the chain link from the previous entry is valid.

    `legacy_chain` selects the pre-IETF envelope shape so old receipts
    (recorded before the profile landed) keep verifying.
    """
    if not signatures:
        return []

    results: list[bool] = []
    prev_hash = "" if legacy_chain else FIRST_RECEIPT_SEED

    for i, sig in enumerate(signatures):
        if legacy_chain:
            entry = json.dumps(
                {
                    "signature_id": sig.get("signature_id", ""),
                    "action_type": sig.get("action_type", ""),
                    "timestamp": sig.get("signed_at", sig.get("timestamp", 0)),
                    "prev_hash": prev_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            current_hash = hashlib.sha256(entry.encode()).hexdigest()
        else:
            envelope = {
                "signature_id": sig.get("signature_id", ""),
                "action_type": sig.get("action_type", ""),
                "context": sig.get("payload"),
                "timestamp": sig.get("signed_at", sig.get("timestamp", 0)),
                "previousReceiptHash": prev_hash,
            }
            current_hash = hashlib.sha256(canonical_json(envelope)).hexdigest()
        results.append(True)  # Valid link in a freshly-built chain
        prev_hash = current_hash

    return results


def _build_explanation(action_type: str, context: dict[str, Any] | None) -> str:
    """Generate a human-readable description for a replay step."""
    parts = action_type.split(":")

    if len(parts) == 2:
        category, detail = parts
        verb_map = {
            "api": f"Called {detail} API",
            "tool": f"Used tool {detail}",
            "data": f"Accessed data source {detail}",
            "file": f"Operated on file {detail}",
            "db": f"Queried database {detail}",
            "auth": f"Authentication event {detail}",
            "model": f"Invoked model {detail}",
            "webhook": f"Triggered webhook {detail}",
        }
        base = verb_map.get(category, f"{category} - {detail}")
    else:
        base = action_type

    if context:
        if "model" in context:
            base += f" (model: {context['model']})"
        elif "endpoint" in context:
            base += f" (endpoint: {context['endpoint']})"
        elif "target" in context:
            base += f" (target: {context['target']})"

    return base


def replay(
    agent_id: str, session_id: str, *, legacy_chain: bool = False
) -> ReplayTimeline:
    """Fetch signatures for a session and reconstruct the audit trail.

    Args:
        agent_id: The agent that owns the session.
        session_id: The session to replay.
        legacy_chain: When True, derive chain links via the pre-IETF
            envelope shape so historical receipts verify. Default False
            uses the IETF v2 shape per §5.7.

    Returns:
        A ReplayTimeline with ordered steps and chain verification.
    """
    raw_sigs = get_session_signatures(session_id)
    return _build_timeline(
        agent_id, session_id, raw_sigs, legacy_chain=legacy_chain
    )


def replay_from_bundle(
    bundle: ComplianceBundle, *, legacy_chain: bool = False
) -> ReplayTimeline:
    """Reconstruct a timeline from a ComplianceBundle offline.

    Args:
        bundle: A previously exported ComplianceBundle.
        legacy_chain: Use the pre-IETF chain shape; see :func:`replay`.

    Returns:
        A ReplayTimeline built from the bundle receipts.
    """
    # Extract agent_id and session_id from receipts if available
    agent_id = ""
    session_id = ""
    if bundle.receipts:
        first = bundle.receipts[0]
        agent_id = first.get("agent_id", "")
        session_id = first.get("session_id", "bundle")

    if not session_id:
        session_id = "bundle"

    # Convert receipts to SignedActionResponse-like objects for processing
    fake_sigs = []
    for r in bundle.receipts:
        fake_sigs.append(
            SignedActionResponse(
                signature_id=r.get("signature_id", ""),
                agent_id=r.get("agent_id", agent_id),
                action_id=r.get("action_id", ""),
                action_type=r.get("action_type", "unknown"),
                payload=r.get("payload"),
                algorithm=r.get("algorithm", ""),
                signed_at=r.get("signed_at", r.get("timestamp", 0)),
                signature_preview=r.get("signature_preview", ""),
                verification_url=r.get("verification_url", ""),
            )
        )

    return _build_timeline(
        agent_id, session_id, fake_sigs, legacy_chain=legacy_chain
    )


def _build_timeline(
    agent_id: str,
    session_id: str,
    signatures: list[SignedActionResponse],
    *,
    legacy_chain: bool = False,
) -> ReplayTimeline:
    """Shared logic to build a ReplayTimeline from a list of signatures."""
    # Sort by timestamp
    sorted_sigs = sorted(signatures, key=lambda s: s.signed_at)

    # Build normalized dicts for chain verification
    sig_dicts = [_normalize_signature(s) for s in sorted_sigs]
    chain_results = _verify_hash_chain(sig_dicts, legacy_chain=legacy_chain)

    # Rolling chain hash; each step records its predecessor's value.
    # Under the IETF v2 profile (§5.7) the seed is the all-zero SHA-256
    # value so verifiers can distinguish "no predecessor" from "predecessor
    # not yet linked". Legacy chains used the empty string.
    chain_hashes: list[str] = []
    prev_hash = "" if legacy_chain else FIRST_RECEIPT_SEED
    for sig in sorted_sigs:
        if legacy_chain:
            entry = json.dumps(
                {
                    "signature_id": sig.signature_id,
                    "action_type": sig.action_type,
                    "timestamp": sig.signed_at,
                    "prev_hash": prev_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            prev_hash = hashlib.sha256(entry.encode()).hexdigest()
        else:
            envelope = {
                "signature_id": sig.signature_id,
                "action_type": sig.action_type,
                "context": sig.payload,
                "timestamp": sig.signed_at,
                "previousReceiptHash": prev_hash,
            }
            prev_hash = hashlib.sha256(canonical_json(envelope)).hexdigest()
        chain_hashes.append(prev_hash)

    steps: list[ReplayStep] = []
    for i, sig in enumerate(sorted_sigs):
        explanation = _build_explanation(sig.action_type, sig.payload)
        prev_chain_hash = chain_hashes[i - 1] if i > 0 else None
        steps.append(
            ReplayStep(
                index=i,
                action_type=sig.action_type,
                context=sig.payload,
                timestamp=sig.signed_at,
                signature_id=sig.signature_id,
                verification_url=sig.verification_url,
                chain_valid=chain_results[i] if i < len(chain_results) else False,
                explanation=explanation,
                prev_chain_hash=prev_chain_hash,
            )
        )

    start_time = sorted_sigs[0].signed_at if sorted_sigs else None
    end_time = sorted_sigs[-1].signed_at if sorted_sigs else None
    chain_integrity = all(chain_results) if chain_results else True

    return ReplayTimeline(
        agent_id=agent_id,
        session_id=session_id,
        steps=steps,
        chain_integrity=chain_integrity,
        start_time=start_time,
        end_time=end_time,
        total_actions=len(steps),
    )
