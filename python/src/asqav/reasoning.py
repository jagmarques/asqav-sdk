"""Reasoning-trace signing helper.

Captures an LLM action's prompt, reasoning trace, and output as a single
signed receipt. Only SHA-256 hashes are posted by default; raw text never
leaves the caller unless ``store_raw`` is set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _sha256(data: Any) -> str:
    """Return hex SHA-256 of ``data`` (dicts/lists JSON-serialized; bytes pass through)."""
    if isinstance(data, bytes):
        payload = data
    elif isinstance(data, str):
        payload = data.encode("utf-8")
    else:
        payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class ReasoningReceipt:
    """Wrapper around ``SignatureResponse`` plus the three captured hashes."""

    signature: Any  # SignatureResponse
    prompt_hash: str
    trace_hash: str
    output_hash: str
    raw_stored: bool = False
    retention_days: int | None = None

    @property
    def signature_id(self) -> str:
        return self.signature.signature_id

    @property
    def verification_url(self) -> str:
        return self.signature.verification_url

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature_id": self.signature.signature_id,
            "action_id": self.signature.action_id,
            "timestamp": self.signature.timestamp,
            "verification_url": self.signature.verification_url,
            "prompt_hash": self.prompt_hash,
            "trace_hash": self.trace_hash,
            "output_hash": self.output_hash,
            "raw_stored": self.raw_stored,
            "retention_days": self.retention_days,
        }


def sign_reasoning(
    agent: Any,
    prompt: Any,
    trace: Any,
    output: Any,
    action_type: str = "ai:reasoning",
    context: dict[str, Any] | None = None,
    trace_id: str | None = None,
    parent_id: str | None = None,
    store_raw: bool = False,
    retention_days: int | None = None,
) -> ReasoningReceipt:
    """Sign an LLM action's prompt + reasoning trace + output.

    Posts only SHA-256 hashes by default; set ``store_raw=True`` to additionally
    post raw payloads (subject to ``retention_days`` server-side).
    """
    prompt_hash = _sha256(prompt)
    trace_hash = _sha256(trace)
    output_hash = _sha256(output)

    merged_context: dict[str, Any] = dict(context) if context else {}
    merged_context["_prompt_hash"] = prompt_hash
    merged_context["_reasoning_trace_hash"] = trace_hash
    merged_context["_output_hash"] = output_hash

    if store_raw:
        merged_context["_raw_prompt"] = prompt
        merged_context["_raw_reasoning_trace"] = trace
        merged_context["_raw_output"] = output
        if retention_days is not None:
            merged_context["_raw_retention_days"] = int(retention_days)

    signature = agent.sign(
        action_type,
        context=merged_context,
        system_prompt_hash=prompt_hash,
        tool_inputs_hash=trace_hash,
        trace_id=trace_id,
        parent_id=parent_id,
    )

    return ReasoningReceipt(
        signature=signature,
        prompt_hash=prompt_hash,
        trace_hash=trace_hash,
        output_hash=output_hash,
        raw_stored=bool(store_raw),
        retention_days=int(retention_days) if retention_days is not None else None,
    )
