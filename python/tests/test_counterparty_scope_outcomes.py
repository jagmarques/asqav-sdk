"""Binding scope is checked before digest resolution and stays distinct from integrity."""

import base64
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from asqav._jcs import canonical_json
from asqav.counterparty import compute_counterparty_binding, verify_counterparty_binding
from asqav import verify_receipt_offline
from asqav.verifier import verify_receipt as standalone
from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify

CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"


def fixture(name="asqav-31-counterparty-scope-match"):
    directory = CORPUS / name
    return [json.loads((directory / f).read_text()) for f in
            ("receipt.json", "originating_envelope.json", "jwks.json")]


def origin_envelope():
    return {
        "payload": {"action_id": "origin", "issuer_id": "origin-key"},
        "signature": {"alg": "Ed25519", "kid": "origin-key", "sig": "+/8="},
        "anchors": [{"type": "rfc3161", "value": "first"}],
    }


def acknowledgment(origin):
    return {
        "payload": {"counterparty_binding": compute_counterparty_binding(origin).to_wire()},
        "signature": {"kid": "ack-key"},
    }


def test_absent_scope_with_origin_is_unknown():
    origin = origin_envelope()
    ack = acknowledgment(origin)
    ack["payload"]["counterparty_binding"].pop("scope", None)
    result = verify_counterparty_binding(ack, origin)
    assert result.valid is None
    assert result.envelope_hash_matches is None
    assert result.label == "legacy_scope"


@pytest.mark.parametrize("scope,label", [(None, "unrecognised_scope"), ("other", "unrecognised_scope")])
def test_scope_dispatch_precedes_origin_and_malformed_digest(scope, label):
    ack = {"payload": {"counterparty_binding": {"scope": scope, "envelope_hash": []}}}
    result = verify_counterparty_binding(ack, object())
    assert result.valid is None
    assert result.label == label


def test_binding_digest_projects_exactly_payload_and_signature():
    origin = origin_envelope()
    original = copy.deepcopy(origin)
    expected = base64.b64encode(hashlib.sha256(canonical_json({
        "payload": origin["payload"], "signature": origin["signature"],
    })).digest()).decode()
    ack = acknowledgment(origin)
    binding = ack["payload"]["counterparty_binding"]
    assert binding["scope"] == "envelope_minus_anchors"
    assert binding["envelope_hash"] == expected
    upgraded = dict(origin, anchors=[{"proof": "upgraded"}], export_metadata={"ignored": True})
    assert verify_counterparty_binding(ack, upgraded).valid is True
    assert origin == original


def test_signature_spelling_is_part_of_commitment():
    origin = origin_envelope()
    ack = acknowledgment(origin)
    changed = copy.deepcopy(origin)
    changed["signature"]["sig"] = "-_8"
    assert verify_counterparty_binding(ack, changed).label == "mismatch"
    assert verify_counterparty_binding(acknowledgment(changed), changed).valid is True


def test_supported_scope_without_origin_is_unresolved():
    result = verify_counterparty_binding(acknowledgment(origin_envelope()), None)
    assert result.valid is None
    assert result.label == "unresolved"


@pytest.mark.parametrize("binding", [None, [], False, "binding"])
def test_malformed_present_binding_is_invalid(binding):
    result = verify_counterparty_binding({"payload": {"counterparty_binding": binding}}, None)
    assert result.valid is False
    assert result.label == "malformed"


@pytest.mark.parametrize("directory", sorted(p.name for p in CORPUS.glob("asqav-3[1-4]-counterparty*")))
def test_signed_public_and_standalone_outcomes(directory, capsys):
    receipt, origin, jwks = fixture(directory)
    expected = json.loads((CORPUS / directory / "expected.json").read_text())
    public = verify_receipt_offline(receipt, jwks, originating_envelope=origin)
    full = standalone.run_structured(receipt, jwks, counterparty=origin,
                                    trusted_tsa_keys=[(CORPUS / directory / "tsa_trust.pem").read_bytes()])
    for result in (public, full):
        assert result["verdict"] == expected["outcome"]
        assert result["failure_class"] == expected.get("failure_class")
        assert next(a for a in result["axes"] if a["name"] == "signature")["result"] == "PASS"
    assert next(a for a in full["axes"] if a["name"] == "anchors")["result"] == "PASS"
    code = standalone.run(receipt, jwks, None, counterparty=origin,
                          trusted_tsa_keys=[(CORPUS / directory / "tsa_trust.pem").read_bytes()])
    assert code == (0 if expected["outcome"] == "verified" else 1 if expected.get("failure_class") == "invalid" else 2)
    output = capsys.readouterr().out
    assert f"=> {expected['outcome']}" in output
    if expected.get("failure_class"):
        assert f"failure_class: {expected['failure_class']}" in output


def test_origin_context_is_local_to_concurrent_and_consecutive_calls():
    receipt, origin, jwks = fixture()
    before = copy.deepcopy(origin)
    def call(index):
        supplied = origin if index % 2 == 0 else None
        return verify_receipt_offline(receipt, jwks, originating_envelope=supplied)["verdict"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(call, range(12))) == ["verified", "unverified"] * 6
    assert verify_receipt_offline(receipt, jwks)["failure_class"] == "unverifiable"
    assert origin == before


def test_anchor_upgrade_does_not_change_signed_binding():
    receipt, origin, jwks = fixture()
    upgraded = dict(origin, anchors=origin["anchors"][1:])
    assert verify_receipt_offline(receipt, jwks, originating_envelope=upgraded)["verdict"] == "verified"
    assert compute_counterparty_binding(upgraded, receipt_ref="sig_counterparty_origin").envelope_hash == receipt["payload"]["counterparty_binding"]["envelope_hash"]


@pytest.mark.parametrize("origin", [None, [], {}, {"payload": []}, {"payload": {}, "signature": {}}])
def test_unavailable_origin_is_unverifiable_through_public_api(origin):
    receipt, _, jwks = fixture()
    result = verify_receipt_offline(receipt, jwks, originating_envelope=origin)
    assert result["failure_class"] == "unverifiable"
    assert "unresolved" in next(a for a in result["axes"] if a["name"] == "counterparty")["note"]


def test_deep_origin_is_unverifiable_and_scope_dispatch_comes_first():
    receipt, origin, jwks = fixture()
    deep = {}
    for _ in range(205):
        deep = {"nested": deep}
    origin["payload"]["deep"] = deep
    assert verify_receipt_offline(receipt, jwks, originating_envelope=origin)["failure_class"] == "unverifiable"
    receipt, _, jwks = fixture("asqav-33-counterparty-scope-absent")
    result = verify(receipt, ADAPTERS, jwks, context=VerificationContext(object()))
    assert "legacy_scope" in result.axis("counterparty").note


def test_forged_signature_dominates_unknown_binding():
    receipt, origin, jwks = fixture("asqav-33-counterparty-scope-absent")
    receipt["payload"]["decision"] = "deny"
    result = verify_receipt_offline(receipt, jwks, originating_envelope=origin)
    assert result["failure_class"] == "invalid"


def test_two_argument_adapter_hook_and_foreign_formats_ignore_context():
    from asqav.verifier.oracle import AerfAdapter
    class Compatible(AerfAdapter):
        def extra_axes(self, doc, key_provider):
            return [("compat", "PASS", "two-argument hook called")]
    adapter = Compatible()
    directory = CORPUS / "aerf-01-genesis"
    receipt = json.loads((directory / "receipt.json").read_text())
    context = VerificationContext(object())
    result = verify(receipt, [adapter], context=context)
    assert result.axis("compat").note == "two-argument hook called"
    for adapter in ADAPTERS:
        if adapter.name == "asqav-native":
            continue
        entry = next(e for e in json.loads((CORPUS / "manifest.json").read_text()) if e["format"] == adapter.name)
        directory = CORPUS / entry["dir"]
        from asqav.verifier.oracle.runner import _key_provider
        doc = json.loads((directory / "receipt.json").read_text())
        provider = _key_provider(directory, adapter.name)
        plain = verify(doc, [adapter], provider)
        contextual = verify(doc, [adapter], provider, context=context)
        assert contextual.verdict == plain.verdict
        assert [(a.axis, a.result) for a in contextual.axes] == [(a.axis, a.result) for a in plain.axes]


@pytest.mark.parametrize("axis_name,status", [("custom_guard", "FAIL"), ("counterparty", "FAIL"), ("counterparty", "SKIPPED")])
def test_native_subclass_keeps_its_two_argument_guard_with_context(axis_name, status):
    from asqav.verifier.oracle import AsqavNativeAdapter
    class Guarded(AsqavNativeAdapter):
        def extra_axes(self, doc, key_provider):
            axes = [axis for axis in super().extra_axes(doc, key_provider) if axis[0] != axis_name]
            return axes + [(axis_name, status, "subclass guard")]
    receipt, origin, jwks = fixture()
    for context in (VerificationContext(), VerificationContext(origin)):
        result = verify(receipt, [Guarded()], jwks, context=context)
        assert any(axis.axis == axis_name for axis in result.axes)
        assert result.axis(axis_name).result == status
        assert result.axis(axis_name).note == "subclass guard"
        assert result.verdict == "unverified"


def test_hosted_signature_envelope_retains_actual_acknowledging_kid():
    from asqav.verifier.verify_receipt import run_structured
    receipt, origin, jwks = fixture()
    receipt["signature_envelope"] = receipt["signature"]
    receipt["signature"] = receipt["signature_envelope"]["sig"]
    assert verify_receipt_offline(receipt, jwks, originating_envelope=origin)["verdict"] == "verified"
    report = run_structured(receipt, jwks, counterparty=origin)
    assert next(a for a in report["axes"] if a["name"] == "counterparty")["result"] == "PASS"
    receipt.pop("signature_envelope")
    report = run_structured(receipt, jwks, counterparty=origin)
    axis = next(a for a in report["axes"] if a["name"] == "counterparty")
    assert axis["result"] == "FAIL" and "kid_mismatch" in axis["note"]


def test_missing_signing_key_is_uncertain_with_a_matching_byte_binding():
    receipt, origin, _ = fixture()
    result = verify_receipt_offline(receipt, {"keys": []}, originating_envelope=origin)
    axes = {axis["name"]: axis for axis in result["axes"]}
    assert axes["counterparty"]["result"] == "PASS"
    assert axes["signature"]["result"] == "SKIPPED"
    assert result["verdict"] == "unverified" and result["failure_class"] == "unverifiable"


@pytest.mark.parametrize("kid,status", [("fixture-b", "FAIL"), ("fixture-a", "PASS"), ("fixture-key-a", "PASS")])
def test_acknowledging_kid_names_the_resolved_signing_key(kid, status, capsys):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    receipt, origin, _ = fixture()
    key = Ed25519PrivateKey.generate()
    receipt["payload"].update(issuer_id="fixture-a", agent_id="fixture-agent-a")
    receipt["payload"]["counterparty_binding"]["expect_ack_from"] = kid
    receipt["signature"]["kid"] = kid
    receipt["signature"]["sig"] = base64.b64encode(key.sign(canonical_json(receipt["payload"]))).decode()
    jwks = {"keys": [{"kid": "fixture-key-a", "issuer_id": "fixture-a", "agent_id": "fixture-agent-a",
                      "alg": "Ed25519", "status": "active", "public_key": base64.b64encode(
                          key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()}]}
    for report in (verify_receipt_offline(receipt, jwks, originating_envelope=origin),
                   standalone.run_structured(receipt, jwks, counterparty=origin)):
        axes = {axis["name"]: axis for axis in report["axes"]}
        assert axes["signature"]["result"] == "PASS"
        assert axes["counterparty"]["result"] == status
    standalone.run(receipt, jwks, None, counterparty=origin)
    output = capsys.readouterr().out
    mark = "  ok" if status == "PASS" else "FAIL"
    assert f"[{mark}] counterparty" in output

