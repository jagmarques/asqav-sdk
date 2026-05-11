"""SDK-side tests for the counterparty acknowledgment binding (IETF -04).

Covers:

* envelope_hash byte-stability (matches the cloud's compute_envelope_hash).
* compute_counterparty_binding default and explicit-locator paths.
* verify_counterparty_binding matches / mismatch / kid-mismatch / unresolved labels.
* verify_compliance_receipt fifth check fires only when originating_envelope is supplied.

The cloud is the authoritative verifier; these tests pin the SDK's local
sanity-check semantics so customer-side reproduction stays byte-stable.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav._jcs import canonical_json
from asqav.client import verify_compliance_receipt
from asqav.counterparty import (
    ACKNOWLEDGMENT_RECEIPT_TYPE,
    CounterpartyBinding,
    compute_counterparty_binding,
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
    expected = base64.b64encode(hashlib.sha256(canonical_json(env)).digest()).decode()
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


def test_verify_counterparty_binding_unresolved_no_binding_field():
    orig = _orig_envelope()
    ack = {"payload": dict(orig["payload"]), "signature": {"kid": "x"}}
    outcome = verify_counterparty_binding(ack, orig)
    assert outcome.valid is False
    assert outcome.label == "unresolved"


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
