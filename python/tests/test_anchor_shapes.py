# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""The three anchors wire shapes: absent, empty array, and null.

The profile states it once: the member ABSENT and the member present as an
EMPTY ARRAY are both conformant and mean the same thing (no anchor presented);
a JSON null is a third shape found in the wild, is MALFORMED, and must be
reported unverifiable rather than read as either. None of the three changes
any digest: anchors sits outside the signed bytes, so asqav-17's Ed25519
signature stays valid under all three spellings, and the two corpus twins
(asqav-27 absent, asqav-28 null) pin the outcomes end to end.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from asqav.verifier import verify_receipt as v
from asqav.verifier.oracle import ADAPTERS
from asqav.verifier.oracle.core import verify as oracle_verify

_VECTORS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"

#: The one sentence all three engines report for the malformed shape, pinned
#: verbatim so a wording drift in any engine fails here and in the TS half.
ANCHORS_NULL_NOTE = (
    "anchors is null: malformed; absent or [] is the conformant spelling of no anchors"
)


def _vector(name: str):
    d = _VECTORS / name
    receipt = json.loads((d / "receipt.json").read_text())
    jwks = json.loads((d / "jwks.json").read_text())
    pred_doc = json.loads((d / "predecessor.json").read_text())
    pred = pred_doc["payload"] if isinstance(pred_doc.get("payload"), dict) else pred_doc
    return receipt, jwks, pred


def _mask_skew(note: str) -> str:
    # The skew note embeds live wall-clock seconds; everything else is deterministic.
    return re.sub(r"-?\d+s\b", "<n>s", note)


def _structured_shape(result: dict) -> list:
    return [
        (a["name"], a["result"], _mask_skew(a["note"]), a["failure_class"])
        for a in result["axes"]
    ]


def _oracle_shape(result) -> list:
    return [
        (a.axis, a.result, _mask_skew(a.note), a.failure_class) for a in result.axes
    ]


def test_run_structured_absent_and_empty_array_are_one_fact() -> None:
    """asqav-27 (member absent) and asqav-17 (member present as []) must be
    indistinguishable in every axis result, the verdict, and the canonical bytes."""
    receipt17, jwks, pred = _vector("asqav-17-seq-contiguous")
    receipt27, _, _ = _vector("asqav-27-anchors-absent")
    out17 = v.run_structured(receipt17, jwks, pred)
    out27 = v.run_structured(receipt27, jwks, pred)
    assert _structured_shape(out27) == _structured_shape(out17)
    assert out27["verdict"] == out17["verdict"]
    assert out27["failure_class"] == out17["failure_class"]
    assert out27["canonical_sha256"] == out17["canonical_sha256"]


def test_run_structured_null_anchors_is_malformed_not_no_anchors() -> None:
    """asqav-28 (``"anchors": null``): the structure axis FAILs, naming the member
    and the shape; the verdict reads unverified/unverifiable; the anchors axis is
    never reached, so null is not reported as "no anchors on this receipt"."""
    receipt, jwks, pred = _vector("asqav-28-anchors-null-malformed")
    out = v.run_structured(receipt, jwks, pred)
    structure = next(a for a in out["axes"] if a["name"] == "structure")
    assert structure["result"] == "FAIL", structure
    assert structure["note"] == ANCHORS_NULL_NOTE
    assert structure["failure_class"] == "unverifiable"
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "unverifiable"
    # The malformed member stops evaluation at the structure gate.
    assert out["coverage"]["stopped_at"] == "structure"
    assert "anchors" not in {a["name"] for a in out["axes"]}
    assert all(a["note"] != "no anchors on this receipt" for a in out["axes"])


def test_oracle_absent_and_empty_array_are_one_fact() -> None:
    """The oracle's axis results for asqav-27 equal asqav-17's, verdict included."""
    receipt17, jwks, pred = _vector("asqav-17-seq-contiguous")
    receipt27, _, _ = _vector("asqav-27-anchors-absent")
    res17 = oracle_verify(receipt17, ADAPTERS, key_provider=jwks, predecessor=pred)
    res27 = oracle_verify(receipt27, ADAPTERS, key_provider=jwks, predecessor=pred)
    assert _oracle_shape(res27) == _oracle_shape(res17)
    assert res27.verdict == res17.verdict == "verified"
    assert res27.failure_class == res17.failure_class
    assert res27.first_failing_edge == res17.first_failing_edge


def test_oracle_null_anchors_is_malformed_not_no_anchors() -> None:
    """The oracle FAILs asqav-28 at the structure axis: unverified, unverifiable,
    first failing edge structure - never a silent read of null as no anchors."""
    receipt, jwks, pred = _vector("asqav-28-anchors-null-malformed")
    res = oracle_verify(receipt, ADAPTERS, key_provider=jwks, predecessor=pred)
    structure = res.axis("structure")
    assert structure is not None
    assert structure.result == "FAIL", structure.note
    assert structure.note == ANCHORS_NULL_NOTE
    assert res.verdict == "unverified"
    assert res.failure_class == "unverifiable"
    assert res.first_failing_edge == "structure"
    assert all(a.note != "no anchors on this receipt" for a in res.axes)
