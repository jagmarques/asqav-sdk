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

import pytest

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


#: Exact version-gate notes, pinned so any wording drift across the three
#: engines fails the cross-engine equality below rather than passing loosely.
_VERSION_NOTES = {
    3: "unsupported wire version 3: unverifiable under this document, reported not failed",
    None: "no v member: not a Compliance Receipt of this document; absence is not version 1",
}


@pytest.mark.parametrize("version", [3, None])
def test_null_anchors_with_unrecognised_version_reports_version_fact(version) -> None:
    """679 C3: asqav-28 with payload.v unrecognised (3) or absent reports the
    wire-version fact on the structure axis in every engine, never the shape."""
    receipt, jwks, pred = _vector("asqav-28-anchors-null-malformed")
    mutated = json.loads(json.dumps(receipt))
    payload = mutated.get("payload", mutated)
    if version is None:
        payload.pop("v", None)
    else:
        payload["v"] = version
    out = v.run_structured(mutated, jwks, pred)
    structure = next(a for a in out["axes"] if a["name"] == "structure")
    res = oracle_verify(mutated, ADAPTERS, key_provider=jwks, predecessor=pred)
    ostruct = res.axis("structure")
    assert ostruct is not None
    assert (structure["result"], structure["note"]) == (ostruct.result, ostruct.note)
    assert (structure["result"], structure["note"]) == ("SKIPPED", _VERSION_NOTES[version])


@pytest.mark.parametrize("version", [1, 3, None])
def test_hash_mode_null_anchors_never_reports_shape_fact(version) -> None:
    """Boundary: a hash-mode envelope (payload None) carries no anchors
    spelling at all, so the anchors-shape rule never fires for any version."""
    d = _VECTORS / "asqav-05-hash-mode-prod"
    receipt = json.loads((d / "receipt.json").read_text())
    jwks = json.loads((d / "jwks.json").read_text())
    mutated = json.loads(json.dumps(receipt))
    mutated["anchors"] = None
    if version is None:
        mutated.pop("v", None)
    else:
        mutated["v"] = version
    out = v.run_structured(mutated, jwks, None)
    assert [(a["name"], a["result"]) for a in out["axes"]] == [("payload", "FAIL")]
    assert all(a["note"] != ANCHORS_NULL_NOTE for a in out["axes"])


def test_corpus_anchor_entries_use_only_declared_members() -> None:
    """Every anchor entry in every pinned vector uses only ANCHOR_ENTRY_MEMBERS.

    The draft declares the member set once (required type/value, optional
    status/anchor_block_hash/tsa_url/operator_id); the corpus must agree, so a
    future vector smuggling an undeclared member fails here, not in review."""
    declared = v.ANCHOR_ENTRY_MEMBERS
    assert declared == {"type", "value", "status", "anchor_block_hash", "tsa_url", "operator_id"}
    checked = 0
    for receipt_path in sorted(_VECTORS.glob("*/receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        for entry in receipt.get("anchors") or []:
            assert isinstance(entry, dict), receipt_path
            assert set(entry) <= declared, (receipt_path, sorted(set(entry) - declared))
            checked += 1
    assert checked > 0, "control: the corpus holds anchor entries to census"


@pytest.mark.parametrize("shape", ["empty", "absent", "null"])
def test_wire_version_does_not_change_anchors_shape_rule(shape) -> None:
    """679 C2: the absent/empty/null rule is identical for every recognised
    wire version; v=1 and v=2 report the same structure fact in each engine."""
    receipt, jwks, pred = _vector("asqav-17-seq-contiguous")
    outs = {}
    for version in (1, 2):
        mutated = json.loads(json.dumps(receipt))
        mutated.get("payload", mutated)["v"] = version
        if shape == "absent":
            mutated.pop("anchors", None)
        elif shape == "null":
            mutated["anchors"] = None
        out = v.run_structured(mutated, jwks, pred)
        structure = next(a for a in out["axes"] if a["name"] == "structure")
        res = oracle_verify(mutated, ADAPTERS, key_provider=jwks, predecessor=pred)
        ostruct = res.axis("structure")
        assert ostruct is not None
        outs[version] = (
            structure["result"], structure["note"], ostruct.result, ostruct.note)
    assert outs[1] == outs[2]
    expected = "FAIL" if shape == "null" else "PASS"
    assert outs[1][0] == expected and outs[1][2] == expected
    # Negative control: an unrecognised version reports a different fact.
    mutated = json.loads(json.dumps(receipt))
    mutated.get("payload", mutated)["v"] = 3
    out = v.run_structured(mutated, jwks, pred)
    structure = next(a for a in out["axes"] if a["name"] == "structure")
    assert (structure["result"], structure["note"]) != outs[1][:2]


def test_unknown_anchor_member_yields_byte_identical_axis() -> None:
    """An entry carrying an undeclared member verifies exactly as without it.

    Unknown members are ignored by the checks and preserved in the envelope, so
    member growth across revisions can neither fail nor silently alter an axis.
    The probe member is asserted undeclared so the test cannot pass vacuously."""
    assert "proof_of_future" not in v.ANCHOR_ENTRY_MEMBERS
    receipt, jwks, pred = _vector("asqav-24-anchor-block-hash-prod")
    assert receipt.get("anchors"), "control: the pinned vector carries anchors"
    mutated = json.loads(json.dumps(receipt))
    for entry in mutated["anchors"]:
        entry["proof_of_future"] = "qhV5..."
    plain = v.run_structured(receipt, jwks, pred)
    extra = v.run_structured(mutated, jwks, pred)
    assert _structured_shape(extra) == _structured_shape(plain)
    assert extra["verdict"] == plain["verdict"]
    assert extra["failure_class"] == plain["failure_class"]
