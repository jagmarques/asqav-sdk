"""Offline API declares its anchor scope (criterion 677).

verify_receipt_offline carries its TypeScript twin's scope declaration: every
result names what it did not check. At base this returns exactly
["axes","failure_class","fmt","verdict"].
"""
from __future__ import annotations

import json
from pathlib import Path

import asqav

_VECTORS = (
    Path(__file__).resolve().parents[2]
    / "verifier"
    / "conformance-vectors"
    / "asqav-27-anchors-absent"
)

#: verifyReceiptOffline top-level keys; typescript/tests/offline-verify.test.ts
#: pins the same literal set against a live TS call, so drift fails one side.
_TS_KEYS = {
    "fmt", "axes", "verdict", "failureClass", "signer",
    "firstFailingEdge", "notChecked", "coverage", "context",
    "trustBundle", "registryEdition", "registryVersion",
}


def _camel_to_snake(name: str) -> str:
    out = []
    for ch in name:
        if ch.isupper():
            out.append("_" + ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _offline_result() -> dict:
    receipt = json.loads((_VECTORS / "receipt.json").read_text())
    jwks = json.loads((_VECTORS / "jwks.json").read_text())
    return asqav.verify_receipt_offline(receipt, jwks)


def test_offline_result_declares_scope() -> None:
    result = _offline_result()
    assert set(result.keys()) >= {"not_checked", "coverage"}
    assert any(
        "anchor" in entry.get("requirement", "")
        for entry in result["not_checked"]
    ), "no anchor requirement in the non-coverage declaration"


def test_offline_key_set_matches_typescript_twin() -> None:
    result = _offline_result()
    assert set(result.keys()) == {_camel_to_snake(k) for k in _TS_KEYS}
