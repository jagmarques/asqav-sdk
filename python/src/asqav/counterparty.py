"""Counterparty acknowledgment binding helpers for the IETF Compliance Receipts profile.

Builds and verifies the ``protectmcp:acknowledgment`` binding that ties B's
receipt to A's originating envelope via a base64 SHA-256 over A's canonical
bytes (see draft-marques-asqav-compliance-receipts Sections 4.6 and 4.6.1).
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from typing import Any

from ._jcs import canonical_json

__all__ = [
    "ACKNOWLEDGMENT_RECEIPT_TYPE",
    "CounterpartyBinding",
    "compute_counterparty_binding",
    "verify_counterparty_binding",
]

#: Receipt type emitted by an acknowledging agent (B) over an originating receipt.
ACKNOWLEDGMENT_RECEIPT_TYPE: str = "protectmcp:acknowledgment"


@dataclass
class CounterpartyBinding:
    """Acknowledger's cross-agent byte-equality binding to an originating receipt.

    ``transport_label`` is operational only and MUST NOT serve as a basis for
    trust derivation.
    """

    envelope_hash: str = field(
        metadata={"description": "base64 SHA-256 of originating envelope JCS bytes."}
    )
    receipt_ref: str = field(
        metadata={"description": "Opaque resolvable id for the originating receipt."}
    )
    expect_ack_from: str | None = field(
        default=None,
        metadata={"description": "Expected acknowledger kid/issuer_id."},
    )
    transport_label: str | None = field(
        default=None,
        metadata={"description": "Operational transport hint (mcp|bus|http)."},
    )

    def to_wire(self) -> dict[str, Any]:
        """Return the wire-shape dict; drops ``None`` keys to match cloud JCS."""
        out: dict[str, Any] = {
            "envelope_hash": self.envelope_hash,
            "receipt_ref": self.receipt_ref,
        }
        if self.expect_ack_from is not None:
            out["expect_ack_from"] = self.expect_ack_from
        if self.transport_label is not None:
            out["transport_label"] = self.transport_label
        return out


def compute_envelope_hash(envelope: dict[str, Any]) -> str:
    """Base64 SHA-256 over the envelope's JCS bytes."""
    return base64.b64encode(hashlib.sha256(canonical_json(envelope)).digest()).decode()


def compute_counterparty_binding(
    originating_envelope: dict[str, Any],
    *,
    receipt_ref: str | None = None,
    expect_ack_from: str | None = None,
    transport_label: str | None = None,
) -> CounterpartyBinding:
    """Build a :class:`CounterpartyBinding` for the originating envelope.

    Callers MUST pass the bytes B actually received (not a re-canonicalization)
    so intermediary tampering is detectable. ``receipt_ref`` defaults to
    ``originating_envelope["payload"]["action_id"]`` when present.
    """
    if receipt_ref is None:
        payload = originating_envelope.get("payload")
        if isinstance(payload, dict):
            ref = payload.get("action_id") or payload.get("signature_id")
            if isinstance(ref, str):
                receipt_ref = ref
    if receipt_ref is None:
        raise ValueError(
            "receipt_ref is required: originating_envelope.payload has no action_id"
        )
    return CounterpartyBinding(
        envelope_hash=compute_envelope_hash(originating_envelope),
        receipt_ref=receipt_ref,
        expect_ack_from=expect_ack_from,
        transport_label=transport_label,
    )


@dataclass
class CounterpartyBindingVerification:
    """Outcome of the counterparty-binding sanity check; ``valid`` ANDs all populated booleans."""

    valid: bool
    envelope_hash_matches: bool
    kid_matches: bool | None
    label: str


def verify_counterparty_binding(
    acknowledgment_envelope: dict[str, Any],
    originating_envelope: dict[str, Any],
) -> CounterpartyBindingVerification:
    """Re-derive and compare the binding's ``envelope_hash``.

    ``label`` is one of ``matches``, ``mismatch``, ``unresolved``,
    ``kid_mismatch``. ``kid_matches`` is ``None`` when ``expect_ack_from`` was
    not declared.
    """
    payload = acknowledgment_envelope.get("payload")
    if not isinstance(payload, dict):
        return CounterpartyBindingVerification(False, False, None, "unresolved")
    binding = payload.get("counterparty_binding")
    if not isinstance(binding, dict):
        return CounterpartyBindingVerification(False, False, None, "unresolved")
    expected_hash = binding.get("envelope_hash")
    actual_hash = compute_envelope_hash(originating_envelope)
    envelope_hash_matches = expected_hash == actual_hash
    expect_ack_from = binding.get("expect_ack_from")
    kid_matches: bool | None = None
    if isinstance(expect_ack_from, str):
        sig = acknowledgment_envelope.get("signature")
        ack_kid = sig.get("kid") if isinstance(sig, dict) else None
        kid_matches = ack_kid == expect_ack_from
    if not envelope_hash_matches:
        return CounterpartyBindingVerification(False, False, kid_matches, "mismatch")
    if kid_matches is False:
        return CounterpartyBindingVerification(False, True, False, "kid_mismatch")
    return CounterpartyBindingVerification(True, True, kid_matches, "matches")
