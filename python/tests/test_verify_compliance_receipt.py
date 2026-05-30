"""Local `verify_compliance_receipt`: required fields, namespace, skew, chain."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav._jcs import canonical_json
from asqav.client import (
    SKEW_BOUND_SECONDS,
    ComplianceReceiptVerification,
    verify_compliance_receipt,
)
from asqav.replay import FIRST_RECEIPT_SEED

# === Fixtures ===


def _good_envelope(**overrides) -> dict:
    """A receipt envelope that passes every check by default.

    Uses the spec wire form ``type`` (draft-marques-asqav-compliance-receipts
    Section 2.1) rather than the SDK-internal ``receipt_type`` attribute.
    """
    base = {
        "type": "protectmcp:decision",
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "issuer_id": "legal:Acme GmbH",
        "agent_id": "agent_001",
        "action_ref": "sha256:" + "a" * 64,
        "payload_digest": "sha256:" + "b" * 64,
        "policy_digest": "sha256:" + "c" * 64,
        "previousReceiptHash": FIRST_RECEIPT_SEED,
        "policy_decision": "permit",
    }
    base.update(overrides)
    return base


# === Happy path ===


def test_clean_first_receipt_validates() -> None:
    result = verify_compliance_receipt(_good_envelope())
    assert isinstance(result, ComplianceReceiptVerification)
    assert result.valid is True
    assert result.fields_present is True
    assert result.receipt_type_in_namespace is True
    assert result.skew_within_bound is True
    assert result.chain_link_rederives is True
    assert result.errors == []


def test_chain_link_rederives_against_predecessor() -> None:
    predecessor = _good_envelope()
    expected_prev = hashlib.sha256(canonical_json(predecessor)).hexdigest()
    successor = _good_envelope(previousReceiptHash=expected_prev)
    result = verify_compliance_receipt(
        successor, predecessor_envelope=predecessor
    )
    assert result.valid is True
    assert result.chain_link_rederives is True


# === Compliance-Receipt minimum-required-fields contract ===


def test_missing_required_field_flags_fields_present_false() -> None:
    env = _good_envelope()
    del env["issuer_id"]
    result = verify_compliance_receipt(env)
    assert result.fields_present is False
    assert result.valid is False
    assert any(e.startswith("missing_fields:") for e in result.errors)
    assert "issuer_id" in next(
        e for e in result.errors if e.startswith("missing_fields:")
    )


def test_multiple_missing_fields_all_reported() -> None:
    env = _good_envelope()
    del env["issuer_id"]
    del env["payload_digest"]
    result = verify_compliance_receipt(env)
    assert result.fields_present is False
    err = next(e for e in result.errors if e.startswith("missing_fields:"))
    assert "issuer_id" in err
    assert "payload_digest" in err


# === receipt_type namespace ===


def test_receipt_type_outside_namespace_rejected() -> None:
    env = _good_envelope(type="custom:weird")
    result = verify_compliance_receipt(env)
    assert result.receipt_type_in_namespace is False
    assert result.valid is False
    assert "invalid_receipt_type" in result.errors


def test_all_three_namespace_values_accepted() -> None:
    for rt in (
        "protectmcp:decision",
        "protectmcp:restraint",
        "protectmcp:lifecycle",
    ):
        env = _good_envelope(type=rt)
        result = verify_compliance_receipt(env)
        assert result.receipt_type_in_namespace is True, rt


def test_receipt_type_attribute_satisfies_namespace_but_not_required_fields() -> None:
    """When the envelope ships a ``receipt_type`` attribute instead of the
    wire ``type``, the namespace check falls back to it and passes; the
    REQUIRED-fields check still demands the wire ``type`` and fails."""
    env = _good_envelope()
    env["receipt_type"] = env.pop("type")
    result = verify_compliance_receipt(env)
    # Namespace fallback succeeds.
    assert result.receipt_type_in_namespace is True
    # But REQUIRED-fields check insists on the wire ``type``.
    assert result.fields_present is False
    assert any(
        e.startswith("missing_fields:") and "type" in e for e in result.errors
    )


# === 300-second skew bound ===


def test_skew_well_within_boundary_is_accepted() -> None:
    """A receipt 250s old (well inside the 300s bound) is accepted."""
    now = time.time()
    issued = datetime.fromtimestamp(
        now - (SKEW_BOUND_SECONDS - 50), tz=timezone.utc
    ).isoformat()
    env = _good_envelope(issued_at=issued)
    result = verify_compliance_receipt(env, now=now)
    assert result.skew_within_bound is True
    assert result.valid is True


def test_skew_just_over_past_boundary_is_accepted() -> None:
    """Spec Section 2.1 issued_at: verifiers MUST NOT reject a receipt
    solely because ``issued_at`` lies in the past. A receipt 305s old is
    therefore accepted by the freshness check; retention is enforced
    elsewhere."""
    now = time.time()
    issued = datetime.fromtimestamp(
        now - SKEW_BOUND_SECONDS - 5, tz=timezone.utc
    ).isoformat()
    env = _good_envelope(issued_at=issued)
    result = verify_compliance_receipt(env, now=now)
    assert result.skew_within_bound is True
    assert "signed_at_skew" not in result.errors


def test_skew_in_future_is_rejected() -> None:
    """Spec Section 2.1 issued_at: verifiers MUST reject when
    ``issued_at`` is more than SKEW_BOUND_SECONDS ahead of the
    verifier's own clock."""
    now = time.time()
    issued = datetime.fromtimestamp(
        now + SKEW_BOUND_SECONDS + 5, tz=timezone.utc
    ).isoformat()
    env = _good_envelope(issued_at=issued)
    result = verify_compliance_receipt(env, now=now)
    assert result.skew_within_bound is False
    assert result.valid is False
    assert "signed_at_skew" in result.errors


def test_skew_one_hour_in_past_is_accepted() -> None:
    """M4 audit fix: an hour-old receipt (skew = -3600s) MUST verify."""
    now = time.time()
    issued = datetime.fromtimestamp(now - 3600, tz=timezone.utc).isoformat()
    env = _good_envelope(issued_at=issued)
    result = verify_compliance_receipt(env, now=now)
    assert result.skew_within_bound is True
    assert result.valid is True


def test_skew_one_hour_in_future_is_rejected() -> None:
    """M4 audit fix: a receipt timestamped an hour ahead (skew = +3600s)
    is outside the 300s bound and MUST reject."""
    now = time.time()
    issued = datetime.fromtimestamp(now + 3600, tz=timezone.utc).isoformat()
    env = _good_envelope(issued_at=issued)
    result = verify_compliance_receipt(env, now=now)
    assert result.skew_within_bound is False
    assert result.valid is False
    assert "signed_at_skew" in result.errors


def test_skew_accepts_unix_seconds_too() -> None:
    now = time.time()
    env = _good_envelope(issued_at=now - 5.0)
    result = verify_compliance_receipt(env, now=now)
    assert result.skew_within_bound is True


def test_invalid_issued_at_is_flagged() -> None:
    env = _good_envelope(issued_at="not a timestamp")
    result = verify_compliance_receipt(env)
    assert result.skew_within_bound is False
    assert "invalid_issued_at" in result.errors


# === Chain link rederivation contract ===


def test_chain_link_mismatch_detected() -> None:
    predecessor = _good_envelope()
    forged_successor = _good_envelope(
        previousReceiptHash="f" * 64,  # not the JCS hash of `predecessor`
    )
    result = verify_compliance_receipt(
        forged_successor, predecessor_envelope=predecessor
    )
    assert result.chain_link_rederives is False
    assert "chain_link_mismatch" in result.errors


def test_first_record_skips_predecessor_check() -> None:
    """A receipt seeded with FIRST_RECEIPT_SEED never needs a predecessor."""
    env = _good_envelope(previousReceiptHash=FIRST_RECEIPT_SEED)
    result = verify_compliance_receipt(env)  # no predecessor passed
    assert result.chain_link_rederives is True


def test_no_predecessor_supplied_leaves_chain_check_passive() -> None:
    """When predecessor is unknown the helper does not fail rederivation;
    callers wanting strict checks must pass the predecessor."""
    env = _good_envelope(previousReceiptHash="a" * 64)
    result = verify_compliance_receipt(env)
    assert result.chain_link_rederives is True


# === Conditional MUSTs on the inner payload ===


def test_sandbox_state_out_of_enum_is_rejected() -> None:
    """`sandbox_state` is restricted to {enabled, disabled, unavailable};
    a receipt that carries any other token MUST verify as invalid."""
    env = _good_envelope(sandbox_state="sandboxed")
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert "invalid_sandbox_state" in result.errors


def test_sandbox_state_canonical_value_passes() -> None:
    env = _good_envelope(sandbox_state="enabled")
    result = verify_compliance_receipt(env)
    assert result.valid is True
    assert "invalid_sandbox_state" not in result.errors


def test_sandbox_state_absent_is_accepted() -> None:
    """The field is OPTIONAL at the syntactic layer; absence MUST NOT
    fail the helper."""
    env = _good_envelope()
    assert "sandbox_state" not in env
    result = verify_compliance_receipt(env)
    assert result.valid is True


def test_deny_decision_without_reason_is_rejected() -> None:
    """`reason` is REQUIRED whenever `decision` is `deny` or `rate_limit`."""
    env = _good_envelope(decision="deny")
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert "missing_reason_on_deny_or_rate_limit" in result.errors


def test_rate_limit_decision_without_reason_is_rejected() -> None:
    env = _good_envelope(decision="rate_limit")
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert "missing_reason_on_deny_or_rate_limit" in result.errors


def test_deny_decision_with_reason_passes() -> None:
    env = _good_envelope(decision="deny", reason="policy:outbound_blocked")
    result = verify_compliance_receipt(env)
    assert result.valid is True
    assert "missing_reason_on_deny_or_rate_limit" not in result.errors


def test_allow_decision_without_reason_is_accepted() -> None:
    """`reason` is conditional, not unconditional; allow-decisions without
    reason MUST verify."""
    env = _good_envelope(decision="allow")
    result = verify_compliance_receipt(env)
    assert result.valid is True


def test_anchor_with_empty_value_is_rejected() -> None:
    """`anchors[].value` is REQUIRED on every anchor entry; entries served
    without `value` MUST NOT be reported as anchor_valid_*=true."""
    env = _good_envelope()
    env["anchors"] = [{"type": "rfc3161", "value": ""}]
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert any(e.startswith("anchor_missing_value:") for e in result.errors)


def test_anchor_without_value_key_is_rejected() -> None:
    env = _good_envelope()
    env["anchors"] = [{"type": "opentimestamps", "status": "pending"}]
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert any(e.startswith("anchor_missing_value:") for e in result.errors)


def test_anchor_with_populated_value_passes() -> None:
    env = _good_envelope()
    env["anchors"] = [{"type": "rfc3161", "value": "dGVzdA=="}]
    result = verify_compliance_receipt(env)
    assert result.valid is True


def test_empty_anchors_array_does_not_trip_the_check() -> None:
    """An empty anchors array is the cloud's pre-anchor state; the helper
    leaves that condition to the anchors-required clause checked elsewhere."""
    env = _good_envelope()
    env["anchors"] = []
    result = verify_compliance_receipt(env)
    assert not any(e.startswith("anchor_missing_value:") for e in result.errors)


# === authorized_under_mandate (server-built, recognised structurally) ===


def _good_mandate(**overrides) -> dict:
    """A well-formed authorized_under_mandate attestation."""
    base = {
        "mandate_id": "man_abc123",
        "issuer_id": "legal:Acme GmbH",
        "verified": True,
        "scope_digest": "sha256:" + "d" * 64,
    }
    base.update(overrides)
    return base


def test_valid_authorized_under_mandate_passes() -> None:
    env = _good_envelope(authorized_under_mandate=_good_mandate())
    result = verify_compliance_receipt(env)
    assert result.valid is True
    assert "false_mandate_attestation_guard" not in result.errors


def test_authorized_under_mandate_absent_is_accepted() -> None:
    env = _good_envelope()
    assert "authorized_under_mandate" not in env
    result = verify_compliance_receipt(env)
    assert result.valid is True


def test_mandate_verified_not_true_is_rejected() -> None:
    """A self-declared attestation with verified!=true is a forged mandate."""
    env = _good_envelope(authorized_under_mandate=_good_mandate(verified=False))
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert "false_mandate_attestation_guard" in result.errors


def test_mandate_bad_scope_digest_is_rejected() -> None:
    """scope_digest must be the sha256:<64-hex> wire form."""
    env = _good_envelope(
        authorized_under_mandate=_good_mandate(scope_digest="deadbeef")
    )
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert "false_mandate_attestation_guard" in result.errors


def test_mandate_missing_mandate_id_is_rejected() -> None:
    bad = _good_mandate()
    del bad["mandate_id"]
    env = _good_envelope(authorized_under_mandate=bad)
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert "false_mandate_attestation_guard" in result.errors


# === controls_evaluated (server-built, recognised structurally) ===


def test_valid_controls_evaluated_passes() -> None:
    """A well-formed controls_evaluated block verifies; quorum carries a
    64-hex attestation_hash and policy carries matched_count >= 1."""
    env = _good_envelope(
        controls_evaluated={
            "quorum": {"fired": True, "attestation_hash": "e" * 64},
            "policy": {"evaluated": True, "matched_count": 2},
            "result": {"present": True},
        }
    )
    result = verify_compliance_receipt(env)
    assert result.valid is True
    assert "false_control_attestation_guard" not in result.errors


def test_controls_evaluated_absent_is_accepted() -> None:
    env = _good_envelope()
    assert "controls_evaluated" not in env
    result = verify_compliance_receipt(env)
    assert result.valid is True


def test_controls_evaluated_not_object_is_rejected() -> None:
    env = _good_envelope(controls_evaluated=["quorum"])
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert "false_control_attestation_guard" in result.errors


def test_controls_evaluated_unknown_key_is_rejected() -> None:
    """A forged control key the server never emits is a forged attestation."""
    env = _good_envelope(controls_evaluated={"backdoor": {"fired": True}})
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert "false_control_attestation_guard" in result.errors


def test_controls_evaluated_quorum_missing_hash_is_rejected() -> None:
    env = _good_envelope(controls_evaluated={"quorum": {"fired": True}})
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert "false_control_attestation_guard" in result.errors


def test_controls_evaluated_quorum_bad_hash_is_rejected() -> None:
    env = _good_envelope(
        controls_evaluated={
            "quorum": {"fired": True, "attestation_hash": "sha256:" + "e" * 64}
        }
    )
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert "false_control_attestation_guard" in result.errors


def test_controls_evaluated_policy_matched_count_zero_is_rejected() -> None:
    """4.7-lesson-7 trap: policy.evaluated=true with matched_count=0 claims a
    policy ran without a real match; a forged attestation MUST be rejected."""
    env = _good_envelope(
        controls_evaluated={"policy": {"evaluated": True, "matched_count": 0}}
    )
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert "false_control_attestation_guard" in result.errors


def test_controls_evaluated_policy_matched_count_bool_is_rejected() -> None:
    """matched_count must be a real int; a bool True must not satisfy >= 1."""
    env = _good_envelope(
        controls_evaluated={"policy": {"evaluated": True, "matched_count": True}}
    )
    result = verify_compliance_receipt(env)
    assert result.valid is False
    assert "false_control_attestation_guard" in result.errors


def test_valid_mandate_and_controls_together_pass() -> None:
    env = _good_envelope(
        authorized_under_mandate=_good_mandate(),
        controls_evaluated={
            "quorum": {"fired": True, "attestation_hash": "e" * 64},
            "policy": {"evaluated": True, "matched_count": 1},
        },
    )
    result = verify_compliance_receipt(env)
    assert result.valid is True
    assert result.errors == []
