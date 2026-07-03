"""Doors: one receipt, N standard envelopes, byte-identical inner across all.

Pins three properties per door:
  1. each wrapper yields a valid schema-shaped envelope containing the exact receipt,
  2. the round-trip inverse extracts the identical inner receipt,
  3. Python and TypeScript emit the same JCS bytes for the same input across the
     JSON-canonicalization-safe domain (BMP keys, integers up to 2**53), with the
     known astral-plane / big-integer divergence pinned as a documented limit.
"""
from __future__ import annotations

import json
from pathlib import Path

from asqav import doors

_ROOT = Path(__file__).resolve().parents[2]
_PARITY = _ROOT / "verifier" / "doors-parity"


def _load(name: str) -> dict:
    return json.loads((_PARITY / f"receipt-{name}.json").read_text())


COMPLIANCE = _load("compliance")
HASHMODE = _load("hashmode")


# --- byte-identical inner: the exact receipt is embedded in every door ---


def test_same_bytes_inner_across_all_doors() -> None:
    """All five doors carry the same receipt with an identical JCS byte string."""
    for receipt in (COMPLIANCE, HASHMODE):
        inners = [
            doors.receipt_from_vc(doors.to_w3c_vc(receipt)),
            doors.receipt_from_cloudevent(doors.to_cloudevent(receipt)),
            doors.receipt_from_otel_genai_attributes(doors.to_otel_genai_attributes(receipt)),
            doors.receipt_from_c2pa_assertion(doors.to_c2pa_assertion(receipt)),
            doors.receipt_from_erc8004_validation_request(
                doors.to_erc8004_validation_request(receipt)
            ),
        ]
        canon = {doors.canonical_json(i) for i in inners}
        assert len(canon) == 1
        assert doors.canonical_json(receipt) in canon


# --- W3C VC ---


def test_vc_shape_and_roundtrip() -> None:
    vc = doors.to_w3c_vc(COMPLIANCE)
    assert vc["@context"][0] == "https://www.w3.org/ns/credentials/v2"
    assert vc["type"] == ["VerifiableCredential", "AsqavReceiptCredential"]
    assert vc["credentialSubject"] == COMPLIANCE
    assert vc["proof"]["type"] == "AsqavReceiptSignature2026"
    assert vc["proof"]["signatureValue"] == COMPLIANCE["signature"]["sig"]
    assert vc["proof"]["receiptDigest"] == doors.receipt_digest(COMPLIANCE)
    assert vc["validFrom"] == COMPLIANCE["payload"]["issued_at"]
    assert doors.receipt_from_vc(vc) == COMPLIANCE


def test_vc_hashmode_signature_from_flat_fields() -> None:
    vc = doors.to_w3c_vc(HASHMODE)
    assert vc["proof"]["signatureValue"] == HASHMODE["signature_b64"]
    assert vc["proof"]["signatureAlgorithm"] == HASHMODE["algorithm"]
    assert doors.receipt_from_vc(vc) == HASHMODE


# --- CloudEvents ---


def test_cloudevent_shape_and_roundtrip() -> None:
    ce = doors.to_cloudevent(COMPLIANCE)
    assert ce["specversion"] == "1.0"
    assert ce["type"] == "com.asqav.receipt.v1"
    assert ce["datacontenttype"] == "application/json"
    assert ce["asqavreceiptdigest"] == doors.receipt_digest(COMPLIANCE)
    assert ce["time"] == COMPLIANCE["payload"]["issued_at"]
    assert ce["data"] == COMPLIANCE
    assert doors.receipt_from_cloudevent(ce) == COMPLIANCE


# --- OTel GenAI ---


def test_otel_genai_keys_and_roundtrip() -> None:
    attrs = doors.to_otel_genai_attributes(COMPLIANCE)
    assert attrs["gen_ai.operation.name"] == COMPLIANCE["payload"]["type"]
    assert attrs["gen_ai.agent.id"] == COMPLIANCE["payload"]["agent_id"]
    assert attrs["gen_ai.tool.name"] == COMPLIANCE["payload"]["tool_name"]
    assert attrs["asqav.receipt.digest"] == doors.receipt_digest(COMPLIANCE)
    assert all(isinstance(v, (str, int, float, bool)) for v in attrs.values())
    assert doors.receipt_from_otel_genai_attributes(attrs) == COMPLIANCE


def test_otel_hashmode_conversation_and_tool_call() -> None:
    attrs = doors.to_otel_genai_attributes(HASHMODE)
    assert attrs["gen_ai.tool.call.id"] == HASHMODE["action_id"]
    assert doors.receipt_from_otel_genai_attributes(attrs) == HASHMODE


# --- C2PA (raw-bytes rule) ---


def test_c2pa_raw_bytes_binding_and_roundtrip() -> None:
    import hashlib

    a = doors.to_c2pa_assertion(COMPLIANCE)
    assert a["label"] == "com.asqav.receipt.v1"
    assert a["hash_binding"]["rule"] == "raw-bytes"
    raw = doors.canonical_receipt_bytes(COMPLIANCE)
    assert a["hash_binding"]["hash"] == hashlib.sha256(raw).hexdigest()
    assert doors.receipt_from_c2pa_assertion(a) == COMPLIANCE


# --- ERC-8004 ---


def test_erc8004_validation_request_and_roundtrip() -> None:
    req = doors.to_erc8004_validation_request(COMPLIANCE, validator="0xabc")
    assert req["function"] == "validationRequest"
    assert req["args"]["validator"] == "0xabc"
    assert req["args"]["agentId"] == COMPLIANCE["payload"]["agent_id"]
    assert req["args"]["receiptCommitment"].startswith("0x")
    assert len(req["args"]["receiptCommitment"]) == 66  # 0x + 32-byte sha256
    assert req["hashAlgorithm"] == "asqav-commitment-sha256"
    assert doors.receipt_from_erc8004_validation_request(req) == COMPLIANCE


# --- purity + generic extract ---


def test_doors_do_not_mutate_input() -> None:
    before = json.dumps(COMPLIANCE, sort_keys=True)
    doors.wrap_all(COMPLIANCE)
    assert json.dumps(COMPLIANCE, sort_keys=True) == before


def test_generic_extract_recovers_receipt_from_every_door() -> None:
    envs = doors.wrap_all(COMPLIANCE)
    assert doors.extract_receipt(envs["w3c_vc"]) == COMPLIANCE
    assert doors.extract_receipt(envs["cloudevent"]) == COMPLIANCE
    assert doors.extract_receipt(envs["otel_genai"]) == COMPLIANCE
    assert doors.extract_receipt(envs["c2pa"]) == COMPLIANCE
    assert doors.extract_receipt(envs["erc8004"]) == COMPLIANCE


# --- cross-language parity: same JCS structure as the TS SDK ---


def test_parity_golden_matches_python_output() -> None:
    """Python wrap_all JCS equals the committed golden the TS test also checks."""
    for name, receipt in (("compliance", COMPLIANCE), ("hashmode", HASHMODE)):
        golden = (_PARITY / f"golden-{name}.jcs").read_bytes()
        assert doors.canonical_json(doors.wrap_all(receipt)) == golden


# --- honest parity boundary of the shared JCS canonicalizer doors serialize with ---


def test_parity_safe_domain_matches_shared_golden() -> None:
    """Non-vacuous parity across the JSON-canonicalization-safe domain.

    Input carries non-ASCII BMP object keys and an integer at the safe-range
    boundary. Both SDK suites assert their canonical bytes equal this same shared
    golden, so matching it proves Python and TS agree on the safe domain, unlike
    the ASCII-only receipt goldens which only exercised the trivial case.
    """
    fixture = json.loads((_PARITY / "parity-safe-input.json").read_text(encoding="utf-8"))
    golden = (_PARITY / "parity-safe.jcs").read_bytes()
    assert doors.canonical_json(fixture) == golden
    assert any(ord(c) > 0x7F for key in fixture for c in key)  # non-ASCII BMP keys
    assert fixture["big"] == 9007199254740991  # integer inside the safe range


def test_known_canonicalization_divergence_outside_safe_domain() -> None:
    """Guard the known pre-existing JCS-core limitation as a documented fact.

    Astral-plane object keys and integers above 2**53 make the Python and TS
    canonicalizers emit different bytes. This pins Python's bytes and asserts they
    differ from the committed TS bytes, so the boundary is visible and regression
    guarded rather than silently claimed equal. It needs a versioned migration to
    close, out of scope for the doors layer.
    """
    fixture = json.loads((_PARITY / "divergence-input.json").read_text(encoding="utf-8"))
    py_bytes = doors.canonical_json(fixture)
    py_golden = (_PARITY / "divergence-python.jcs").read_bytes()
    ts_golden = (_PARITY / "divergence-typescript.jcs").read_bytes()

    assert py_bytes == py_golden  # Python output pinned
    assert py_golden != ts_golden  # the two SDKs diverge here

    # cause 1: integer above 2**53 stays exact in Python, rounds in the TS number path
    assert b"9007199254740993" in py_golden
    assert b"9007199254740992" in ts_golden
    # cause 2: astral key (U+10000) sorts after U+FFFF in Python, before it in TS
    astral, bmp_max = "\U00010000".encode("utf-8"), "￿".encode("utf-8")
    assert py_golden.index(astral) > py_golden.index(bmp_max)
    assert ts_golden.index(astral) < ts_golden.index(bmp_max)
