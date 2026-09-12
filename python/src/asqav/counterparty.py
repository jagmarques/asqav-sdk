"""Byte bindings over the originating payload and exact signature object."""

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
BINDING_SCOPE = "envelope_minus_anchors"


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
    scope: str | None = field(
        default=None,
        metadata={"description": "Digest projection; populated by the computation helper."},
    )

    def to_wire(self) -> dict[str, Any]:
        """Serialize the supplied members without assigning an unchecked digest a scope."""
        out: dict[str, Any] = {
            "envelope_hash": self.envelope_hash,
            "receipt_ref": self.receipt_ref,
        }
        if self.expect_ack_from is not None:
            out["expect_ack_from"] = self.expect_ack_from
        if self.transport_label is not None:
            out["transport_label"] = self.transport_label
        if self.scope is not None:
            out["scope"] = self.scope
        return out


def compute_envelope_hash(envelope: dict[str, Any]) -> str:
    """Hash exactly payload and signature, retaining the signature's encoded spelling."""
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
        raise ValueError("counterparty_origin_unavailable: payload must be an object")
    signature = envelope.get("signature")
    if not isinstance(signature, dict) or any(
        not isinstance(signature.get(key), str) or not signature[key]
        for key in ("alg", "kid", "sig")
    ):
        raise ValueError("counterparty_origin_unavailable: signature object is incomplete")
    projection = {"payload": envelope["payload"], "signature": signature}
    stack = [(projection, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > 200:
            raise ValueError("counterparty_origin_unavailable: nesting exceeds 200 levels")
        children = (
            node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
        )
        stack.extend((child, depth + 1) for child in children)
    try:
        encoded = canonical_json(projection)
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise ValueError("counterparty_origin_unavailable: canonical bytes unavailable") from exc
    return base64.b64encode(hashlib.sha256(encoded).digest()).decode()


def compute_counterparty_binding(
    originating_envelope: dict[str, Any],
    *,
    receipt_ref: str | None = None,
    expect_ack_from: str | None = None,
    transport_label: str | None = None,
) -> CounterpartyBinding:
    """Build a :class:`CounterpartyBinding` for the originating envelope.

    Pass the peer's original signing envelope. Hosted admission needs an explicit
    originating signature_id as receipt_ref; the action_id fallback is an offline locator.
    """
    if receipt_ref is None:
        payload = (
            originating_envelope.get("payload") if isinstance(originating_envelope, dict) else None
        )
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
        scope=BINDING_SCOPE,
    )


@dataclass
class CounterpartyBindingVerification:
    """Three-state byte-binding outcome; an unknown applicable binding cannot pass."""

    valid: bool | None
    envelope_hash_matches: bool | None
    kid_matches: bool | None
    label: str | None


def _binding_digest(binding: dict[str, Any]) -> bytes | None:
    """Read a well-formed binding digest under either supported base64 alphabet."""
    if not isinstance(binding.get("receipt_ref"), str) or not binding["receipt_ref"]:
        return None
    if any(binding.get(key) is not None and not isinstance(binding[key], str)
           for key in ("expect_ack_from", "transport_label")):
        return None
    value = binding.get("envelope_hash")
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = value.replace("-", "+").replace("_", "/")
        decoded = base64.b64decode(normalized + "=" * (-len(normalized) % 4), validate=True)
    except (ValueError, TypeError):
        return None
    return decoded if len(decoded) == 32 else None


def verify_counterparty_binding(
    acknowledgment_envelope: dict[str, Any],
    originating_envelope: dict[str, Any] | None = None,
) -> CounterpartyBindingVerification:
    """Dispatch scope before resolving bytes; uncertainty is separate from a mismatch."""
    payload = (
        acknowledgment_envelope.get("payload")
        if isinstance(acknowledgment_envelope, dict) else None
    )
    if not isinstance(payload, dict):
        return CounterpartyBindingVerification(False, False, None, "malformed")
    if "counterparty_binding" not in payload:
        return CounterpartyBindingVerification(None, None, None, None)
    binding = payload.get("counterparty_binding")
    if not isinstance(binding, dict):
        return CounterpartyBindingVerification(False, False, None, "malformed")
    if "scope" not in binding:
        return CounterpartyBindingVerification(None, None, None, "legacy_scope")
    if binding["scope"] != BINDING_SCOPE:
        return CounterpartyBindingVerification(None, None, None, "unrecognised_scope")
    expected_hash = _binding_digest(binding)
    if expected_hash is None:
        return CounterpartyBindingVerification(False, False, None, "malformed")
    try:
        actual_hash = base64.b64decode(compute_envelope_hash(originating_envelope))
    except (TypeError, ValueError, RecursionError):
        return CounterpartyBindingVerification(None, None, None, "unresolved")
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
