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


def test_legacy_receipt_type_attribute_still_accepted_for_namespace() -> None:
    """Callers that have not yet migrated their emitter to the wire form
    may still ship a ``receipt_type`` attribute; the namespace check
    falls back to it. The REQUIRED-fields check, however, demands the
    wire ``type``."""
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
