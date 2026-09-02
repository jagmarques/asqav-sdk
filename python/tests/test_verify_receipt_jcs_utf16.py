"""The standalone verifier reproduces RFC 8785 member order and names the pre-cutover dialect.

Criterion 566. ``verify_receipt.py`` is the tool the exit manifest tells a customer to run,
so its ``canonical_json`` must produce the bytes the platform signs for every member name,
including names above U+FFFF where a code-point sort silently diverges.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from asqav.verifier import verify_receipt as v
from tests.test_verify_receipt import _envelope, _jwks_key, _ml_dsa_65, _valid_payload

ASTRAL = {"＠": 1, "😀": 1}
RFC8785_BYTES = '{"😀":1,"＠":1}'.encode("utf-8")
RFC8785_SHA256 = "425159f5c1f0575fbcbf9d05a8f60cde3d040eae5166aa2136657564048651b6"
CODE_POINT_BYTES = '{"＠":1,"😀":1}'.encode("utf-8")
CODE_POINT_SHA256 = "1c314559129cce00bc1b3caa2ee37fa3e81f926aee65b34f8e4e21856b2de83b"


def test_canonical_json_orders_member_names_by_utf16_code_unit() -> None:
    out = v.canonical_json(ASTRAL)
    assert out == RFC8785_BYTES
    assert hashlib.sha256(out).hexdigest() == RFC8785_SHA256


def test_canonical_json_never_reproduces_the_code_point_order() -> None:
    out = v.canonical_json(ASTRAL)
    assert out != CODE_POINT_BYTES
    assert hashlib.sha256(out).hexdigest() != CODE_POINT_SHA256


def test_pre_cutover_dialect_is_the_code_point_order() -> None:
    assert v.canonical_json_pre_cutover(ASTRAL) == CODE_POINT_BYTES


def test_canonical_json_matches_the_sdk_core_on_nested_and_coerced_keys() -> None:
    from asqav._jcs import canonical_json as sdk_canonical

    doc = {"z": [{"😀": {"b": 1, "＠": 2}}, {1: "int key", None: "null key"}], "￿": 0}
    assert v.canonical_json(doc) == sdk_canonical(doc)


def test_jwk_thumbprint_bytes_are_unchanged_for_ascii_members() -> None:
    jwk = {"pub": "QUFBQQ", "kty": "AKP", "alg": "ML-DSA-65"}
    expected = hashlib.sha256(
        json.dumps(jwk, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert v.jwk_thumbprint(jwk) == f"sha256:{expected}"


def test_has_supplementary_member_name_walks_every_depth() -> None:
    assert v.has_supplementary_member_name({"a": [{"b": {"😀": 1}}]})
    assert not v.has_supplementary_member_name({"a": [{"b": {"￿": "😀"}}]})


def _astral_payload(issued_at: str) -> dict:
    payload = _valid_payload("org-legit", "agt_two")
    payload["issued_at"] = issued_at
    payload["context"] = {"tool_input": ASTRAL}
    encoded = v.canonical_json(payload["context"])
    payload["payload_digest"] = {"hash": hashlib.sha256(encoded).hexdigest(), "size": len(encoded)}
    return payload


def _signature_axis(report: dict) -> tuple[str, str]:
    axis = next(a for a in report["axes"] if a["name"] == "signature")
    return axis["result"], axis["note"]


def _run(payload: dict, signed_bytes: bytes) -> dict:
    ml = _ml_dsa_65()
    pk, sk = ml.keygen()
    sig = ml.sign(sk, signed_bytes)
    jwks = {"keys": [_jwks_key("agent-two", "agt_two", "org-legit", pk)]}
    return v.run_structured(_envelope(payload, sig), jwks, None)


@pytest.fixture(autouse=True)
def _needs_ml_dsa():
    pytest.importorskip("dilithium_py")


def test_receipt_with_supplementary_member_names_verifies_over_rfc8785_bytes() -> None:
    payload = _astral_payload("2026-06-19T00:00:00+00:00")
    report = _run(payload, v.canonical_json(payload))
    result, note = _signature_axis(report)
    assert result == "PASS", note
    digest = next(a for a in report["axes"] if a["name"] == "payload_digest")
    assert digest["result"] == "PASS", digest["note"]


def test_pre_cutover_receipt_signed_over_code_point_bytes_is_named_not_verified() -> None:
    payload = _astral_payload("2026-06-19T00:00:00+00:00")
    report = _run(payload, v.canonical_json_pre_cutover(payload))
    result, note = _signature_axis(report)
    assert result == "FAIL"
    assert "pre-cutover dialect" in note
    assert report["verdict"] == v.VERDICT_UNVERIFIED


def test_post_cutover_receipt_signed_over_code_point_bytes_gets_no_retry() -> None:
    cutover = v._parse_stamp(v.JCS_UTF16_CUTOVER)
    assert cutover is not None
    later = cutover.replace(year=cutover.year + 1).isoformat()
    payload = _astral_payload(later)
    report = _run(payload, v.canonical_json_pre_cutover(payload))
    result, note = _signature_axis(report)
    assert result == "FAIL"
    assert "pre-cutover dialect" not in note
    assert report["verdict"] == v.VERDICT_UNVERIFIED


def test_bmp_only_receipt_is_untouched_by_the_diagnostic() -> None:
    payload = _valid_payload("org-legit", "agt_two")
    payload["issued_at"] = "2026-06-19T00:00:00+00:00"
    report = _run(payload, v.canonical_json_pre_cutover(payload))
    result, note = _signature_axis(report)
    assert result == "PASS", note
