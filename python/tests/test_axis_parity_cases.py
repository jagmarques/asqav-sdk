"""Anchor-binding, clock-skew, and signed-expiry axis parity, Python half.

One JSON table drives both the Python and the TypeScript axis implementations, so
a rule that drifts in one language fails a suite. The TypeScript half lives in
typescript/tests/verifier-axis-parity.test.ts and reads the same file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import asqav
from asqav.verifier.oracle.core import MAX_NESTING_DEPTH as ORACLE_MAX_NESTING_DEPTH
from asqav.verifier.verify_receipt import (
    SKEW_BOUND_SECONDS,
    check_anchors,
    check_expiry,
    check_skew,
    envelope_minus_anchors_jcs,
    normalise_envelope,
    run_structured,
)

CASES_FILE = Path(__file__).parent.parent.parent / "verifier" / "axis-parity-cases.json"
TABLE = json.loads(CASES_FILE.read_text())
PIPELOCK_VECTOR = (
    CASES_FILE.parent / "conformance-vectors" / "pipelock-ev2-01-proxy-decision"
)


    # A table that silently empties would make every case below vacuous.
def test_table_is_populated() -> None:
    assert len(TABLE["anchors"]) >= 25, f"only {len(TABLE['anchors'])} anchor cases"
    assert len(TABLE["skew"]) >= 15, f"only {len(TABLE['skew'])} skew cases"
    assert len(TABLE["expiry"]) >= 15, f"only {len(TABLE['expiry'])} expiry cases"
    # A table of only-PASS or only-FAIL cases cannot tell a working axis from a
    # stuck one, so both directions have to be present.
    outcomes = {c["expect"]["result"] for c in TABLE["expiry"]}
    assert outcomes == {"PASS", "FAIL"}, outcomes


    # The limits are part of the contract the other language mirrors.
def test_bounds_match_the_table() -> None:
    assert SKEW_BOUND_SECONDS == 300
    assert ORACLE_MAX_NESTING_DEPTH == 200


@pytest.mark.parametrize("case", TABLE["anchors"], ids=[c["name"] for c in TABLE["anchors"]])
def test_anchor_case(case: dict) -> None:
    result, note = check_anchors(case["envelope"])
    assert result == case["expect"]["result"], f"{case['name']}: note {note}"
    # The digest in the note is the JCS of the envelope minus anchors, so an exact
    # match also proves the two canonicalisers agree byte for byte.
    assert note == case["expect"]["note"], case["name"]


@pytest.mark.parametrize("case", TABLE["skew"], ids=[c["name"] for c in TABLE["skew"]])
def test_skew_case(case: dict) -> None:
    result, note = check_skew(case["issued_at"])
    assert result == case["expect"]["result"], f"{case['name']}: note {note}"
    assert case["expect"]["note_contains"] in note, case["name"]


@pytest.mark.parametrize("case", TABLE["expiry"], ids=[c["name"] for c in TABLE["expiry"]])
def test_expiry_case(case: dict) -> None:
    result, note = check_expiry(case["payload"])
    assert result == case["expect"]["result"], f"{case['name']}: note {note}"
    assert case["expect"]["note_contains"] in note, f"{case['name']}: note {note}"


def test_expiry_reads_only_the_signed_bytes() -> None:
    """An expires_at outside the signed payload must not move the window.

    anchors and the envelope keys are unsigned, so a lapsed receipt that carries a
    future expires_at beside its payload has to stay FAIL.
    """
    signed = {"expires_at": "2020-01-01T00:00:00Z"}
    assert check_expiry(signed)[0] == "FAIL"
    envelope = {
        "payload": dict(signed),
        "signature": {"alg": "ML-DSA-65", "kid": "k1", "sig": "AAAA"},
        "anchors": [],
        "expires_at": "2099-01-01T00:00:00Z",
    }
    axes = {a["name"]: a for a in run_structured(envelope, {"keys": []}, None)["axes"]}
    assert axes["expiry"]["result"] == "FAIL", axes["expiry"]


    # The reference pipelock receipt with this case's chain_prev_hash substituted.
def _pipelock_receipt(case: dict) -> dict:
    doc = json.loads((PIPELOCK_VECTOR / "receipt.json").read_text())
    if case.get("omit"):
        doc.pop("chain_prev_hash", None)
    elif "value" in case:
        doc["chain_prev_hash"] = case["value"]
    return doc


@pytest.mark.parametrize(
    "case", TABLE["chain_prev_hash"], ids=[c["name"] for c in TABLE["chain_prev_hash"]]
)
def test_chain_prev_hash_case(case: dict) -> None:
    keys = json.loads((PIPELOCK_VECTOR / "keys.json").read_text())
    result = asqav.verify_receipt_offline(_pipelock_receipt(case), keys)
    axes = {a["name"]: a["result"] for a in result["axes"]}
    assert result["verdict"] == case["expect"]["verdict"], f"{case['name']}: axes {axes}"
    assert result["failure_class"] == case["expect"]["failure_class"], (
        f"{case['name']}: axes {axes}"
    )
    assert axes["chain"] == case["expect"]["chain"], f"{case['name']}: axes {axes}"
    assert axes["signature"] == case["expect"]["signature"], f"{case['name']}: axes {axes}"


def test_chain_prev_hash_table_discriminates() -> None:
    """Both a genesis PASS and a non-genesis SKIPPED have to be present.

    A table of one outcome cannot tell a working genesis rule from one that reads
    every value as an absent link.
    """
    chains = {c["expect"]["chain"] for c in TABLE["chain_prev_hash"]}
    assert chains == {"PASS", "SKIPPED"}, chains
    non_string = [
        c
        for c in TABLE["chain_prev_hash"]
        if "value" in c and not isinstance(c["value"], str) and c["value"] is not None
    ]
    assert len(non_string) >= 6, f"only {len(non_string)} non-string values"
    assert all(c["expect"]["chain"] == "SKIPPED" for c in non_string), non_string


@pytest.mark.parametrize("case", TABLE["normalise"], ids=[c["name"] for c in TABLE["normalise"]])
def test_normalise_case(case: dict) -> None:
    env = normalise_envelope(case["raw"])
    digest = hashlib.sha256(envelope_minus_anchors_jcs(env)).hexdigest()
    # The digest is the bytes the anchors axis binds, so an exact match proves both
    # halves normalise and canonicalise the same envelope.
    assert digest == case["expect"]["digest"], case["name"]
    assert check_anchors(env)[0] == case["expect"]["anchors_axis"], case["name"]
