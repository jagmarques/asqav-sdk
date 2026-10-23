"""Byte bindings over the originating payload and exact signature object."""

from __future__ import annotations

import base64
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from . import strict_json
from ._jcs import canonical_json

__all__ = [
    "ACKNOWLEDGMENT_RECEIPT_TYPE",
    "BINDING_SCOPE",
    "BINDING_MEMBERS",
    "PROFILE_VERSION",
    "CounterpartyBinding",
    "compute_counterparty_binding",
    "terms_digest_of",
    "verify_counterparty_binding",
]

#: Receipt type emitted by an acknowledging agent (B) over an originating one.
ACKNOWLEDGMENT_RECEIPT_TYPE: str = "protectmcp:acknowledgment"
BINDING_SCOPE = "envelope_minus_anchors"

#: Cross-org interaction profile version this SDK can speak (must-understand).
PROFILE_VERSION = "0.1"

#: Members a profile-declaring binding may carry; anything else is refused.
BINDING_MEMBERS = frozenset({
    "envelope_hash", "receipt_ref", "expect_ack_from", "transport_label",
    "scope", "profile_version", "terms", "terms_digest",
})


def _reject_negative_zero(value: Any) -> None:
    """A -0.0 term would hash differently across JCS implementations, so refuse it."""
    if isinstance(value, float) and value == 0.0 and math.copysign(1.0, value) < 0:
        raise ValueError(
            "counterparty_binding_malformed: terms carry a negative zero"
        )
    if isinstance(value, dict):
        for item in value.values():
            _reject_negative_zero(item)
    elif isinstance(value, list):
        for item in value:
            _reject_negative_zero(item)


def terms_digest_of(terms: Any) -> str:
    """SHA-256 over JCS bytes of the exact terms object (matches the cloud)."""
    if not isinstance(terms, dict):
        raise ValueError("counterparty_binding_malformed: terms must be an object")
    _reject_negative_zero(terms)
    try:
        encoded = canonical_json(terms)
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise ValueError(
            "counterparty_binding_malformed: terms are not canonically encodable"
        ) from exc
    return base64.b64encode(hashlib.sha256(encoded).digest()).decode()


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
    profile_version: str | None = field(
        default=None,
        metadata={"description": "Cross-org interaction profile version (absent is legacy)."},
    )
    terms: dict[str, Any] | None = field(
        default=None,
        metadata={"description": "Exact endorsed terms; terms_digest recomputes over it."},
    )
    terms_digest: str | None = field(
        default=None,
        metadata={"description": "base64 SHA-256 of the terms object's JCS bytes."},
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
        if self.profile_version is not None:
            out["profile_version"] = self.profile_version
        if self.terms is not None:
            out["terms"] = self.terms
        if self.terms_digest is not None:
            out["terms_digest"] = self.terms_digest
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
    try:
        # Raw-text ingest refuses the origin on such an integer; so does a dict.
        strict_json.assert_canonical_integers(envelope)
    except strict_json.UnsafeIntegerError as exc:
        raise ValueError("counterparty_origin_unavailable: integer outside +/-2**53") from exc
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
    profile_version: str | None = None,
    terms: dict[str, Any] | None = None,
) -> CounterpartyBinding:
    """Build a :class:`CounterpartyBinding` for the originating envelope.
    Hosted admission needs an explicit signature_id as receipt_ref."""
    if profile_version is not None and profile_version != PROFILE_VERSION:
        raise ValueError(
            f"counterparty_profile_unsupported: {profile_version!r} is not {PROFILE_VERSION!r}"
        )
    digest = terms_digest_of(terms) if terms is not None else None
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
        profile_version=profile_version,
        terms=terms,
        terms_digest=digest,
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
    profile_version = binding.get("profile_version")
    if profile_version is not None:
        if profile_version != PROFILE_VERSION:
            return CounterpartyBindingVerification(None, None, None, "profile_unsupported")
        if set(binding) - BINDING_MEMBERS:
            return CounterpartyBindingVerification(False, False, None, "malformed")
    terms = binding.get("terms")
    terms_digest = binding.get("terms_digest")
    if "terms" in binding or "terms_digest" in binding:
        if not isinstance(terms_digest, str) or not isinstance(terms, dict):
            return CounterpartyBindingVerification(False, False, None, "malformed")
        try:
            recomputed = terms_digest_of(terms)
        except ValueError:
            return CounterpartyBindingVerification(False, False, None, "malformed")
        if recomputed != terms_digest:
            return CounterpartyBindingVerification(False, False, None, "malformed")
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
