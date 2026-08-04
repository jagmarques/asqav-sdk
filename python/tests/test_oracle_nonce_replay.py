"""Criterion 435: closed controls_evaluated key set and duplicate-nonce replay flag."""

from __future__ import annotations

from asqav.verifier.oracle.adapters.asqav_native import AsqavNativeAdapter


def _doc(nonce: str) -> dict:
    return {
        "payload": {
            "type": "protectmcp:decision",
            "issued_at": "2026-06-01T19:26:44Z",
            "issuer_id": "org-1",
            "agent_id": "agt-1",
            "action_ref": "sha256:" + "8" * 64,
            "payload_digest": {"hash": "8" * 64, "size": 1},
            "policy_digest": "sha256:" + "3" * 64,
            "previousReceiptHash": "0" * 64,
            "decision": "allow",
            "nonce": nonce,
        },
        "signature": {"alg": "ML-DSA-65", "kid": "k", "sig": "AAAA"},
        "anchors": [],
    }


def _axis_results(adapter: AsqavNativeAdapter, doc: dict) -> dict:
    return {name: res for name, res, _note in adapter.extra_axes(doc, {"keys": []})}


def test_adapter_flags_duplicate_nonce_across_receipts() -> None:
    adapter = AsqavNativeAdapter()
    first = _axis_results(adapter, _doc("ab" * 12))
    second = _axis_results(adapter, _doc("ab" * 12))
    assert first["nonce"] == "PASS"
    assert second["nonce"] == "FAIL"


def test_adapter_same_nonce_other_issuer_is_not_duplicate() -> None:
    adapter = AsqavNativeAdapter()
    a = _doc("cd" * 12)
    b = _doc("cd" * 12)
    b["payload"]["issuer_id"] = "org-2"
    assert _axis_results(adapter, a)["nonce"] == "PASS"
    assert _axis_results(adapter, b)["nonce"] == "PASS"


def test_adapter_schema_rejects_unknown_control_key() -> None:
    adapter = AsqavNativeAdapter()
    doc = _doc("ef" * 12)
    doc["payload"]["controls_evaluated"] = {"bogus": {}}
    res, note = adapter.schema(doc)
    assert res == "FAIL"
    assert "bogus" in note
