"""Tests for the local-side `verify_compliance_receipt` helper.

The cloud is the authoritative verifier; the SDK helper covers the
four MUSTs the SDK can check without a network roundtrip:

* REQUIRED fields presence (§5),
* `receipt_type` in the `protectmcp:*` namespace (F1),
* 300-second skew bound (V6 / §5.1.2),
* `previousReceiptHash` rederives over the JCS bytes of the
  predecessor envelope (§5.7).
"""

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _good_envelope(**overrides) -> dict:
    """A receipt envelope that passes every check by default."""
    base = {
        "type": "protectmcp:decision",
        "receipt_type": "protectmcp:decision",
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


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Missing fields (§5)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# receipt_type namespace (F1)
# ---------------------------------------------------------------------------


def test_receipt_type_outside_namespace_rejected() -> None:
    env = _good_envelope(receipt_type="custom:weird", type="custom:weird")
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
        env = _good_envelope(receipt_type=rt, type=rt)
        result = verify_compliance_receipt(env)
        assert result.receipt_type_in_namespace is True, rt


# ---------------------------------------------------------------------------
# 300-second skew bound (V6)
# ---------------------------------------------------------------------------


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


def test_skew_just_over_boundary_is_rejected() -> None:
    """A receipt 305s old (just past the 300s bound) is rejected."""
    now = time.time()
    issued = datetime.fromtimestamp(
        now - SKEW_BOUND_SECONDS - 5, tz=timezone.utc
    ).isoformat()
    env = _good_envelope(issued_at=issued)
    result = verify_compliance_receipt(env, now=now)
    assert result.skew_within_bound is False
    assert result.valid is False
    assert "signed_at_skew" in result.errors


def test_skew_in_future_is_also_rejected() -> None:
    now = time.time()
    issued = datetime.fromtimestamp(
        now + SKEW_BOUND_SECONDS + 5, tz=timezone.utc
    ).isoformat()
    env = _good_envelope(issued_at=issued)
    result = verify_compliance_receipt(env, now=now)
    assert result.skew_within_bound is False


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


# ---------------------------------------------------------------------------
# Chain link rederivation (§5.7)
# ---------------------------------------------------------------------------


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
