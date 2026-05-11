"""Counterparty acknowledgment binding helpers for the IETF Compliance Receipts profile.

Public surface:

* :class:`CounterpartyBinding` - dataclass mirror of the cloud's
  ``core.envelope.CounterpartyBinding`` Pydantic model.
* :func:`compute_counterparty_binding` - builds the binding object from
  an originating envelope; computes the base64 SHA-256 over its
  canonical bytes.
* :func:`verify_counterparty_binding` - re-derives the digest from the
  originating envelope and reports the per-axis outcome.

The acknowledging receipt's ``receipt_type`` is
``protectmcp:acknowledgment``; the binding object travels on the signed
payload of B's receipt and gates verification per the staged IETF -04
revision (Sections 4.6 and 4.6.1).

See https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/
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

    ``envelope_hash`` is the base64-encoded SHA-256 digest over the
    originating receipt's full canonical signed envelope (signature
    bytes included). ``receipt_ref`` is the opaque resolvable locator
    the verifier uses to fetch A's envelope from the Audit Pack or a
    Deployer-published index. ``expect_ack_from`` is an OPTIONAL
    cross-check on the acknowledging receipt's signature ``kid``;
    ``transport_label`` is operational only and MUST NOT be used to
    derive trust.
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
        """Return the wire-shape dict suitable for placement on the signed payload.

        Drops keys whose value is ``None`` so omitted OPTIONAL fields do
        not appear in the canonical bytes; this keeps the JCS output
        byte-identical to the cloud's emit path.
        """
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
    """Base64 of SHA-256 over the originating envelope's full JCS bytes.

    Mirrors the cloud's ``core.envelope.compute_envelope_hash`` so the
    acknowledger and the verifier produce byte-identical digests across
    Python and TypeScript SDKs and the cloud emit path.
    """
    return base64.b64encode(hashlib.sha256(canonical_json(envelope)).digest()).decode()


def compute_counterparty_binding(
    originating_envelope: dict[str, Any],
    *,
    receipt_ref: str | None = None,
    expect_ack_from: str | None = None,
    transport_label: str | None = None,
) -> CounterpartyBinding:
    """Build a :class:`CounterpartyBinding` for the originating envelope.

    Args:
        originating_envelope: Full signed envelope (``{payload, signature, anchors?}``)
            of A's receipt. The digest is computed over the canonical bytes of
            the dict as-passed; callers MUST pass the bytes B actually received,
            not a re-canonicalization, so intermediary tampering is detectable.
        receipt_ref: Resolvable locator for A's receipt. Defaults to
            ``originating_envelope["payload"]["action_id"]`` when present so the
            common case can be a one-liner; callers SHOULD pass an explicit
            value when their Audit Pack production layer uses a different
            identifier scheme.
        expect_ack_from: Optional declared acknowledger identifier; the verifier
            cross-checks the acknowledging receipt's ``signature.kid`` against
            this value when present.
        transport_label: Optional operational transport hint.

    Returns:
        A :class:`CounterpartyBinding` ready to place on the acknowledgment
        receipt's signed payload via :meth:`CounterpartyBinding.to_wire`.
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
    """Outcome of the SDK-side counterparty-binding sanity check.

    ``valid`` is the AND of all populated booleans. Per-axis labels mirror
    the cloud's ``counterparty_binding_verified`` vocabulary in
    ``api/routes/verify.py``.
    """

    valid: bool
    envelope_hash_matches: bool
    kid_matches: bool | None
    label: str


def verify_counterparty_binding(
    acknowledgment_envelope: dict[str, Any],
    originating_envelope: dict[str, Any],
) -> CounterpartyBindingVerification:
    """Re-derive and compare the binding's ``envelope_hash``.

    Returns ``label`` in {``matches``, ``mismatch``, ``unresolved``,
    ``kid_mismatch``} so callers can distinguish a tampered handoff from
    a missing originator. ``kid_matches`` is None when
    ``expect_ack_from`` was not declared on the binding.
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
