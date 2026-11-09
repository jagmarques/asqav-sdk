"""Offline checks of a compliance receipt."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from ..verifier.verify_receipt import _BARE_SHA256_RE
from ._sign_checks import _ALLOWED_CONTROL_KEYS, _SHA256_HEX_RE
from ._vocabulary import (
    _COMPLIANCE_REQUIRED_FIELDS,
    RECEIPT_TYPE_NAMESPACE,
    SANDBOX_STATE_NAMESPACE,
    SKEW_BOUND_SECONDS,
    _resolve_previous_receipt_hash,
)


@dataclass
class ComplianceReceiptVerification:
    """Outcome of the SDK-side Compliance Receipt sanity check.
    Each boolean is one MUST clause; the cloud is the authoritative verifier."""

    valid: bool
    fields_present: bool
    receipt_type_in_namespace: bool
    skew_within_bound: bool
    chain_link_rederives: bool
    errors: list[str]
    counterparty_binding_verified: bool | None = None


def _compliance_receipt_skew(signed: dict[str, Any], now: float | None, errors: list[str]) -> bool:
    from datetime import datetime
    # 3. 300-second skew bound.
    issued_at = signed.get("issued_at") or signed.get("signed_at")
    skew_within_bound = False
    if issued_at is None:
        errors.append("missing_issued_at")
    else:
        try:
            if isinstance(issued_at, (int, float)):
                ts = float(issued_at)
            else:
                # ISO 8601; tolerate trailing "Z".
                s = str(issued_at).replace("Z", "+00:00")
                ts = datetime.fromisoformat(s).timestamp()
            wall = now if now is not None else time.time()
            # Reject future skew > SKEW_BOUND_SECONDS; past follows retention.
            skew = ts - wall
            skew_within_bound = skew <= SKEW_BOUND_SECONDS
            if not skew_within_bound:
                errors.append("signed_at_skew")
        except (ValueError, TypeError):
            errors.append("invalid_issued_at")
    return skew_within_bound


def _compliance_receipt_chain(
    envelope: dict[str, Any], predecessor_envelope: dict[str, Any] | None, errors: list[str],
) -> bool:
    from .._jcs import canonical_json
    from ..replay import FIRST_RECEIPT_SEED
    # 4. Chain link rederives. Only checked when caller supplies the
    # predecessor envelope; first-record seeds skip this.
    chain_link_rederives = True
    expected_prev = _resolve_previous_receipt_hash(envelope)
    if expected_prev == FIRST_RECEIPT_SEED:
        # First record on the chain; nothing to rederive.
        pass
    elif predecessor_envelope is not None:
        # Cloud hashes only the inner compliance payload, not the bundle.
        _inner = predecessor_envelope.get("payload")
        if isinstance(_inner, dict) and (
            "signature" in predecessor_envelope or "anchors" in predecessor_envelope
        ):
            _chain_input: dict[str, Any] = _inner
        else:
            _chain_input = predecessor_envelope
        actual_prev = hashlib.sha256(canonical_json(_chain_input)).hexdigest()
        if actual_prev != expected_prev:
            chain_link_rederives = False
            errors.append("chain_link_mismatch")
    # No predecessor given: leave chain_link_rederives=True (unchecked).
    return chain_link_rederives


def _compliance_receipt_counterparty(
    envelope: dict[str, Any], originating_envelope: dict[str, Any] | None, errors: list[str],
) -> tuple[bool, bool | None]:
    counterparty_binding_verified: bool | None = None
    payload_obj = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else envelope
    has_binding = isinstance(payload_obj, dict) and "counterparty_binding" in payload_obj
    if has_binding:
        from ..counterparty import verify_counterparty_binding

        binding_obj = payload_obj["counterparty_binding"]
        has_full_envelope = "payload" in envelope
        ack_env = envelope if has_full_envelope else {"payload": payload_obj}
        outcome = verify_counterparty_binding(ack_env, originating_envelope)
        expects_ack_kid = (
            isinstance(binding_obj, dict) and isinstance(binding_obj.get("expect_ack_from"), str)
        )
        if outcome.label == "kid_mismatch" and expects_ack_kid and not has_full_envelope:
            # Payload-only input cannot supply ``signature.kid``; surface a
            # clear outcome instead of a silent kid-axis false negative.
            counterparty_binding_verified = False
            errors.append("counterparty_binding_requires_full_envelope")
        else:
            counterparty_binding_verified = outcome.valid
            if not outcome.valid:
                errors.append(f"counterparty_binding_{outcome.label}")
    return has_binding, counterparty_binding_verified


def _compliance_receipt_controls(_payload: dict[str, Any], errors: list[str]) -> None:
    # Reject malformed controls_evaluated (see false_control_attestation_guard).
    if isinstance(_payload, dict):
        _ce = _payload.get("controls_evaluated")
        if _ce is not None:
            if not isinstance(_ce, dict):
                errors.append("false_control_attestation_guard")
            else:
                if any(k not in _ALLOWED_CONTROL_KEYS for k in _ce):
                    errors.append("false_control_attestation_guard")
                _quorum = _ce.get("quorum")
                if _quorum is not None:
                    _ah = _quorum.get("attestation_hash") if isinstance(_quorum, dict) else None
                    if (
                        not isinstance(_quorum, dict)
                        or _quorum.get("fired") is not True
                        or not isinstance(_ah, str)
                        or not _BARE_SHA256_RE.match(_ah)
                    ):
                        errors.append("false_control_attestation_guard")
                _policy = _ce.get("policy")
                if _policy is not None:
                    _mc = _policy.get("matched_count") if isinstance(_policy, dict) else None
                    if (
                        not isinstance(_policy, dict)
                        or _policy.get("evaluated") is not True
                        or isinstance(_mc, bool)
                        or not isinstance(_mc, int)
                        or _mc < 1
                    ):
                        errors.append("false_control_attestation_guard")


def _compliance_receipt_mandate(_payload: dict[str, Any], errors: list[str]) -> None:
    # Reject malformed authorized_under_mandate (false_mandate guard).
    if isinstance(_payload, dict):
        _aum = _payload.get("authorized_under_mandate")
        if _aum is not None:
            _scope_digest = _aum.get("scope_digest") if isinstance(_aum, dict) else None
            if (
                not isinstance(_aum, dict)
                or not _aum.get("mandate_id")
                or not _aum.get("issuer_id")
                or _aum.get("verified") is not True
                or not isinstance(_scope_digest, str)
                or not _SHA256_HEX_RE.match(_scope_digest)
            ):
                errors.append("false_mandate_attestation_guard")


def verify_compliance_receipt(
    envelope: dict[str, Any],
    *,
    predecessor_envelope: dict[str, Any] | None = None,
    originating_envelope: dict[str, Any] | None = None,
    now: float | None = None,
) -> ComplianceReceiptVerification:
    """Local sanity-check on a Compliance Receipt envelope: REQUIRED fields, namespace,
    freshness skew, chain-link rederivation, and counterparty binding when supplied.
    Cloud is authoritative for signature checks, policy_digest resolution, and anchors.
    """
    errors: list[str] = []

    # Canonical wire envelope {payload, signature, anchors}: the signed fields
    # live inside the payload member; flat bundles keep them top-level.
    _inner = envelope.get("payload")
    if isinstance(_inner, dict) and ("signature" in envelope or "anchors" in envelope):
        signed = _inner
    else:
        signed = envelope

    # 1. REQUIRED fields present.
    missing = [f for f in _COMPLIANCE_REQUIRED_FIELDS if f not in signed]
    fields_present = not missing
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")

    # 2. ``type`` namespace; ``receipt_type`` is accepted too.
    rt = signed.get("type") or signed.get("receipt_type")
    receipt_type_in_namespace = rt in RECEIPT_TYPE_NAMESPACE
    if not receipt_type_in_namespace:
        errors.append("invalid_receipt_type")

    skew_within_bound = _compliance_receipt_skew(signed, now, errors)

    chain_link_rederives = _compliance_receipt_chain(envelope, predecessor_envelope, errors)

    # 5. Conditional MUSTs: sandbox_state, deny/rate_limit reason, anchors value
    _payload = signed
    if isinstance(_payload, dict):
        _sandbox = _payload.get("sandbox_state")
        if _sandbox is not None and _sandbox not in SANDBOX_STATE_NAMESPACE:
            errors.append("invalid_sandbox_state")
        _decision = _payload.get("decision")
        if _decision in {"deny", "rate_limit"} and not _payload.get("reason"):
            errors.append("missing_reason_on_deny_or_rate_limit")
    _anchors = envelope.get("anchors")
    if isinstance(_anchors, list):
        for _i, _entry in enumerate(_anchors):
            if not isinstance(_entry, dict):
                continue
            if not _entry.get("value"):
                errors.append(f"anchor_missing_value:{_i}")

    has_binding, counterparty_binding_verified = _compliance_receipt_counterparty(
        envelope, originating_envelope, errors,
    )

    _compliance_receipt_mandate(_payload, errors)

    _compliance_receipt_controls(_payload, errors)

    # Conditional-MUST failures surface via `errors` and `valid` only.
    _conditional_must_fail = any(
        e.startswith(
            (
                "invalid_sandbox_state",
                "missing_reason_on_deny_or_rate_limit",
                "anchor_missing_value:",
                "false_mandate_attestation_guard",
                "false_control_attestation_guard",
            )
        )
        for e in errors
    )
    valid = (
        fields_present
        and receipt_type_in_namespace
        and skew_within_bound
        and chain_link_rederives
        and not _conditional_must_fail
        and (not has_binding or counterparty_binding_verified is True)
    )
    return ComplianceReceiptVerification(
        valid=valid,
        fields_present=fields_present,
        receipt_type_in_namespace=receipt_type_in_namespace,
        skew_within_bound=skew_within_bound,
        chain_link_rederives=chain_link_rederives,
        errors=errors,
        counterparty_binding_verified=counterparty_binding_verified,
    )
