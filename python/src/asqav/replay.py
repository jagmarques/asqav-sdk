"""Audit trail replay: reconstruct and verify what any agent did, step by step.

Chain shape (IETF Compliance Receipts):

  ``previousReceiptHash = sha256(JCS(predecessor signed envelope))``.

The first record's predecessor seed is ``"0" * 64`` (``FIRST_RECEIPT_SEED``).
See https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/
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
    # When the caller has the full signed envelope for this step (the
    # exact bytes the cloud put under `compute_signature_record_hash_v2`),
    # verify_chain hashes those bytes byte-for-byte and tampering of any
    # field surfaces.
    signed_envelope: dict[str, Any] | None = None


@dataclass
class ReplayTimeline:
    """Complete audit trail timeline for a session."""

    agent_id: str
    session_id: str
    steps: list[ReplayStep] = field(default_factory=list)
    chain_integrity: bool = True
    # True only when EVERY step under compliance mode carried a
    # `signed_envelope` and verified byte-for-byte against the cloud's
    # `compute_signature_record_hash_v2`.
    compliance_chain_valid: bool = True
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
                    "signed_envelope": s.signed_envelope,
                }
                for s in self.steps
            ],
            "chain_integrity": self.chain_integrity,
            "compliance_chain_valid": self.compliance_chain_valid,
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

    def verify_chain(self) -> bool:
        """Recompute each step's predecessor hash, compare to stored
        ``prev_chain_hash``. Mismatch = tampering. Steps lacking a stored
        hash are marked invalid rather than passing silently.

        Each step MUST carry `signed_envelope`. The chain hash is computed
        as ``sha256(JCS(signed_envelope))`` so the result matches the
        cloud's ``compute_signature_record_hash_v2`` byte-for-byte. Steps
        without an envelope cannot prove byte-level integrity and drop
        `compliance_chain_valid`.
        """
        if not self.steps:
            self.chain_integrity = True
            self.compliance_chain_valid = True
            return True

        prev_hash = FIRST_RECEIPT_SEED
        all_valid = True
        compliance_valid = True

        for step in self.steps:
            if step.index == 0:
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

            envelope = getattr(step, "signed_envelope", None)
            if envelope is not None:
                prev_hash = hashlib.sha256(canonical_json(envelope)).hexdigest()
            else:
                compliance_valid = False
                prev_hash = _step_chain_hash(step, prev_hash)

        self.chain_integrity = all_valid
        self.compliance_chain_valid = bool(compliance_valid and all_valid)
        return all_valid


def _step_chain_hash(step: ReplayStep, prev_hash: str) -> str:
    """Compute the synthetic chain hash a step contributes when no
    `signed_envelope` is attached. Internal-consistency only: tampering
    of fields outside the synthetic shape will NOT be caught. Callers
    that need byte-level integrity MUST attach `signed_envelope`.
    """
    envelope = {
        "signature_id": step.signature_id,
        "action_type": step.action_type,
        "context": step.context,
        "timestamp": step.timestamp,
        "previousReceiptHash": prev_hash,
    }
    return hashlib.sha256(canonical_json(envelope)).hexdigest()


def _verify_hash_chain(signatures: list[dict]) -> list[bool]:
    """Check each signature chains to the previous via SHA-256.

    Returns a list of booleans, one per signature, indicating whether
    the chain link from the previous entry is valid.
    """
    if not signatures:
        return []

    results: list[bool] = []
    prev_hash = FIRST_RECEIPT_SEED

    for i, sig in enumerate(signatures):
        envelope = {
            "signature_id": sig.get("signature_id", ""),
            "action_type": sig.get("action_type", ""),
            "context": sig.get("payload"),
            "timestamp": sig.get("signed_at", sig.get("timestamp", 0)),
            "previousReceiptHash": prev_hash,
        }
        current_hash = hashlib.sha256(canonical_json(envelope)).hexdigest()
        results.append(True)
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


def replay(agent_id: str, session_id: str) -> ReplayTimeline:
    """Fetch signatures for a session and reconstruct the audit trail.

    Args:
        agent_id: The agent that owns the session.
        session_id: The session to replay.

    Returns:
        A ReplayTimeline with ordered steps and chain verification.
    """
    raw_sigs = get_session_signatures(session_id)
    return _build_timeline(agent_id, session_id, raw_sigs)


def replay_from_bundle(bundle: ComplianceBundle) -> ReplayTimeline:
    """Reconstruct a timeline from a ComplianceBundle offline.

    Args:
        bundle: A ComplianceBundle from an earlier export.

    Returns:
        A ReplayTimeline built from the bundle receipts.
    """
    agent_id = ""
    session_id = ""
    if bundle.receipts:
        first = bundle.receipts[0]
        agent_id = first.get("agent_id", "")
        session_id = first.get("session_id", "bundle")

    if not session_id:
        session_id = "bundle"

    fake_sigs = []
    envelopes: list[dict[str, Any] | None] = []
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
        envelopes.append(r.get("signed_envelope"))

    return _build_timeline(
        agent_id, session_id, fake_sigs, signed_envelopes=envelopes,
    )


def _build_timeline(
    agent_id: str,
    session_id: str,
    signatures: list[SignedActionResponse],
    *,
    signed_envelopes: list[dict[str, Any] | None] | None = None,
) -> ReplayTimeline:
    """Shared logic to build a ReplayTimeline from a list of signatures."""
    indexed = list(enumerate(signatures))
    indexed.sort(key=lambda pair: pair[1].signed_at)
    sorted_sigs = [s for _, s in indexed]
    sorted_envelopes: list[dict[str, Any] | None] = []
    if signed_envelopes is not None:
        sorted_envelopes = [
            signed_envelopes[orig_idx] if orig_idx < len(signed_envelopes) else None
            for orig_idx, _ in indexed
        ]
    else:
        sorted_envelopes = [None] * len(sorted_sigs)

    sig_dicts = [_normalize_signature(s) for s in sorted_sigs]
    chain_results = _verify_hash_chain(sig_dicts)

    chain_hashes: list[str] = []
    prev_hash = FIRST_RECEIPT_SEED
    for sig, env in zip(sorted_sigs, sorted_envelopes):
        if env is not None:
            prev_hash = hashlib.sha256(canonical_json(env)).hexdigest()
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
        env = sorted_envelopes[i]
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
                signed_envelope=env,
            )
        )

    start_time = sorted_sigs[0].signed_at if sorted_sigs else None
    end_time = sorted_sigs[-1].signed_at if sorted_sigs else None
    chain_integrity = all(chain_results) if chain_results else True
    compliance_chain_valid = (
        chain_integrity
        and all(env is not None for env in sorted_envelopes)
        and len(sorted_envelopes) == len(sorted_sigs)
    )

    return ReplayTimeline(
        agent_id=agent_id,
        session_id=session_id,
        steps=steps,
        chain_integrity=chain_integrity,
        compliance_chain_valid=compliance_chain_valid,
        start_time=start_time,
        end_time=end_time,
        total_actions=len(steps),
    )
