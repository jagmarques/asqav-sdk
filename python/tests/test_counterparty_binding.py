"""SDK-side tests for the counterparty acknowledgment binding (IETF -04)."""

from __future__ import annotations

import base64
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav._jcs import canonical_json
from asqav.client import verify_compliance_receipt
from asqav.counterparty import (
    ACKNOWLEDGMENT_RECEIPT_TYPE,
    PROFILE_VERSION,
    CounterpartyBinding,
    compute_counterparty_binding,
    terms_digest_of,
    verify_counterparty_binding,
)


def _orig_envelope() -> dict:
    payload = {
        "v": 1,
        "action_id": "act_abc",
        "agent_id": "agt_A",
        "org_id": "org_1",
        "type": "protectmcp:decision",
        "issuer_id": "00000000000000000010",
        "action_ref": "sha256:" + "a" * 64,
        "payload_digest": {"hash": "sha256:" + "b" * 64, "size": 7},
        "policy_digest": "sha256:" + "c" * 64,
        "previousReceiptHash": "0" * 64,
        "issued_at": "2026-05-11T00:00:00Z",
        "decision": "allow",
        "mode": "payload",
        "action_type": "api:call",
        "context": {"model": "gpt-4"},
        "timestamp": "2026-05-11T00:00:00Z",
    }
    return {
        "payload": payload,
        "signature": {"alg": "ml-dsa-65", "kid": "kid-A", "sig": "AAAA"},
        "anchors": [],
    }


def test_envelope_hash_is_base64_sha256_of_jcs():
    env = _orig_envelope()
    binding = compute_counterparty_binding(env)
    expected = base64.b64encode(hashlib.sha256(canonical_json({"payload": env["payload"], "signature": env["signature"]})).digest()).decode()
    assert binding.envelope_hash == expected


def test_compute_counterparty_binding_defaults_receipt_ref_to_action_id():
    env = _orig_envelope()
    binding = compute_counterparty_binding(env)
    assert binding.receipt_ref == "act_abc"
    assert binding.expect_ack_from is None
    assert binding.transport_label is None


def test_compute_counterparty_binding_explicit_receipt_ref():
    env = _orig_envelope()
    binding = compute_counterparty_binding(
        env,
        receipt_ref="asqav-receipt://org/1/agent_A/seq/42",
        expect_ack_from="00000000000000000099",
        transport_label="mcp",
    )
    assert binding.receipt_ref == "asqav-receipt://org/1/agent_A/seq/42"
    assert binding.expect_ack_from == "00000000000000000099"
    assert binding.transport_label == "mcp"


def test_to_wire_drops_none_optional_fields():
    binding = CounterpartyBinding(envelope_hash="h", receipt_ref="r")
    wire = binding.to_wire()
    assert wire == {"envelope_hash": "h", "receipt_ref": "r"}


def test_to_wire_keeps_optional_fields_when_set():
    binding = CounterpartyBinding(
        envelope_hash="h",
        receipt_ref="r",
        expect_ack_from="kid-B",
        transport_label="bus",
    )
    wire = binding.to_wire()
    assert wire == {
        "envelope_hash": "h",
        "receipt_ref": "r",
        "expect_ack_from": "kid-B",
        "transport_label": "bus",
    }


def _ack_envelope(
    orig_env: dict,
    *,
    expect_ack_from: str | None = None,
    ack_kid: str = "kid-B",
) -> dict:
    binding = compute_counterparty_binding(orig_env, expect_ack_from=expect_ack_from)
    return {
        "payload": {
            "v": 1,
            "type": ACKNOWLEDGMENT_RECEIPT_TYPE,
            "issuer_id": "00000000000000000020",
            "action_ref": orig_env["payload"]["action_ref"],
            "payload_digest": {"hash": "sha256:" + "d" * 64, "size": 0},
            "policy_digest": "sha256:" + "e" * 64,
            "previousReceiptHash": "0" * 64,
            "issued_at": "2026-05-11T00:00:01Z",
            "decision": "allow",
            "counterparty_binding": binding.to_wire(),
        },
        "signature": {"alg": "ml-dsa-65", "kid": ack_kid, "sig": "BBBB"},
        "anchors": [],
    }


def test_verify_counterparty_binding_honest_pair_matches():
    orig = _orig_envelope()
    ack = _ack_envelope(orig)
    outcome = verify_counterparty_binding(ack, orig)
    assert outcome.valid is True
    assert outcome.label == "matches"
    assert outcome.envelope_hash_matches is True
    assert outcome.kid_matches is None


def test_verify_counterparty_binding_payload_flip_mismatch():
    orig = _orig_envelope()
    ack = _ack_envelope(orig)
    tampered = _orig_envelope()
    tampered["payload"]["context"] = {"model": "gpt-5"}
    outcome = verify_counterparty_binding(ack, tampered)
    assert outcome.valid is False
    assert outcome.label == "mismatch"


def test_verify_counterparty_binding_kid_mismatch():
    orig = _orig_envelope()
    ack = _ack_envelope(orig, expect_ack_from="kid-B", ack_kid="kid-attacker")
    outcome = verify_counterparty_binding(ack, orig)
    assert outcome.valid is False
    assert outcome.label == "kid_mismatch"
    assert outcome.kid_matches is False


def test_verify_counterparty_binding_absent_is_not_applicable():
    orig = _orig_envelope()
    ack = {"payload": dict(orig["payload"]), "signature": {"kid": "x"}}
    outcome = verify_counterparty_binding(ack, orig)
    assert outcome.valid is None
    assert outcome.label is None


def test_verify_compliance_receipt_fifth_check_fires_with_originating_envelope():
    orig = _orig_envelope()
    ack = _ack_envelope(orig)
    result = verify_compliance_receipt(
        ack["payload"],
        originating_envelope=orig,
    )
    assert result.counterparty_binding_verified is True


def test_verify_compliance_receipt_skips_binding_when_no_originating_envelope():
    orig = _orig_envelope()
    ack = _ack_envelope(orig)
    result = verify_compliance_receipt(ack["payload"])
    assert result.counterparty_binding_verified is None


def test_verify_compliance_receipt_binding_mismatch_marks_invalid():
    orig = _orig_envelope()
    ack = _ack_envelope(orig)
    tampered = _orig_envelope()
    tampered["payload"]["decision"] = "deny"
    result = verify_compliance_receipt(
        ack["payload"],
        originating_envelope=tampered,
    )
    assert result.counterparty_binding_verified is False
    assert any(e.startswith("counterparty_binding_") for e in result.errors)


def test_acknowledgment_receipt_type_in_namespace():
    from asqav.client import RECEIPT_TYPE_NAMESPACE

    assert ACKNOWLEDGMENT_RECEIPT_TYPE in RECEIPT_TYPE_NAMESPACE


def test_verify_compliance_receipt_payload_only_with_expect_ack_from_surfaces_error():
    """A payload-only caller with expect_ack_from cannot verify the kid axis."""
    orig = _orig_envelope()
    ack = _ack_envelope(orig, expect_ack_from="kid-B", ack_kid="kid-B")
    result = verify_compliance_receipt(
        ack["payload"], originating_envelope=orig
    )
    assert result.counterparty_binding_verified is False
    assert "counterparty_binding_requires_full_envelope" in result.errors
    assert not any(
        e == "counterparty_binding_kid_mismatch" for e in result.errors
    )


def test_verify_compliance_receipt_payload_only_without_expect_ack_from_still_works():
    orig = _orig_envelope()
    ack = _ack_envelope(orig)
    result = verify_compliance_receipt(
        ack["payload"], originating_envelope=orig
    )
    assert result.counterparty_binding_verified is True


def test_profiled_binding_emits_version_and_recomputable_terms_digest():
    env = _orig_envelope()
    terms = {"scope": "invoice:42", "amount": 100, "currency": "EUR"}
    binding = compute_counterparty_binding(
        env, profile_version=PROFILE_VERSION, terms=terms
    )
    assert binding.profile_version == PROFILE_VERSION
    assert binding.terms == terms
    expected = base64.b64encode(hashlib.sha256(canonical_json(terms)).digest()).decode()
    assert binding.terms_digest == expected
    wire = binding.to_wire()
    assert wire["profile_version"] == PROFILE_VERSION
    assert wire["terms_digest"] == expected


def test_unknown_profile_version_is_refused_at_construction():
    with pytest.raises(ValueError, match="counterparty_profile_unsupported"):
        compute_counterparty_binding(_orig_envelope(), profile_version="9.9")


def test_divergent_terms_produce_different_digests():
    a = terms_digest_of({"amount": 100})
    b = terms_digest_of({"amount": 101})
    assert a != b


def _ack_with_binding(binding_wire: dict) -> dict:
    return {
        "payload": {"v": 1, "type": ACKNOWLEDGMENT_RECEIPT_TYPE,
                    "counterparty_binding": binding_wire},
        "signature": {"alg": "ml-dsa-65", "kid": "kid-B", "sig": "BBBB"},
    }


def test_verify_accepts_profiled_binding():
    env = _orig_envelope()
    binding = compute_counterparty_binding(
        env, profile_version=PROFILE_VERSION, terms={"scope": "invoice:42"}
    )
    result = verify_counterparty_binding(_ack_with_binding(binding.to_wire()), env)
    assert (result.valid, result.label) == (True, "matches")


def test_verify_refuses_unknown_profile_version_without_downgrade():
    env = _orig_envelope()
    wire = compute_counterparty_binding(env).to_wire()
    wire["profile_version"] = "9.9"
    result = verify_counterparty_binding(_ack_with_binding(wire), env)
    assert (result.valid, result.label) == (None, "profile_unsupported")


def test_verify_refuses_digest_that_does_not_recompute_over_terms():
    env = _orig_envelope()
    wire = compute_counterparty_binding(
        env, profile_version=PROFILE_VERSION, terms={"amount": 100}
    ).to_wire()
    wire["terms"]["amount"] = 101
    result = verify_counterparty_binding(_ack_with_binding(wire), env)
    assert (result.valid, result.label) == (False, "malformed")


def test_verify_refuses_terms_digest_without_terms():
    env = _orig_envelope()
    wire = compute_counterparty_binding(
        env, profile_version=PROFILE_VERSION, terms={"amount": 100}
    ).to_wire()
    wire.pop("terms")
    result = verify_counterparty_binding(_ack_with_binding(wire), env)
    assert (result.valid, result.label) == (False, "malformed")


def test_verify_refuses_unknown_member_on_a_profiled_binding():
    env = _orig_envelope()
    wire = compute_counterparty_binding(env, profile_version=PROFILE_VERSION).to_wire()
    wire["required_extension"] = True
    result = verify_counterparty_binding(_ack_with_binding(wire), env)
    assert (result.valid, result.label) == (False, "malformed")


def test_negative_zero_terms_are_refused():
    with pytest.raises(ValueError):
        terms_digest_of({"amount": -0.0})
    env = _orig_envelope()
    wire = compute_counterparty_binding(
        env, profile_version=PROFILE_VERSION, terms={"amount": 1}
    ).to_wire()
    wire["terms"] = {"amount": -0.0}
    wire["terms_digest"] = "AAAA"
    result = verify_counterparty_binding(_ack_with_binding(wire), env)
    assert (result.valid, result.label) == (False, "malformed")


def test_null_terms_are_refused_not_treated_as_absent():
    env = _orig_envelope()
    wire = compute_counterparty_binding(
        env, profile_version=PROFILE_VERSION, terms={"amount": 1}
    ).to_wire()
    wire["terms"] = None
    wire["terms_digest"] = None
    result = verify_counterparty_binding(_ack_with_binding(wire), env)
    assert (result.valid, result.label) == (False, "malformed")
