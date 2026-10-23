"""Reassessment review through the production attribution caller (488 C27)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="signed fixtures need the cryptography lib")

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.assessment_store import AssessmentStore
from asqav.verifier.oracle.live_compare import classify_and_attribute
from asqav.verifier.oracle.retain import record_verify_result

_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"
_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"
_PINNED = "2026-10-16T00:00:00Z"
_T1 = "2019-06-01T00:00:00Z"
_T2 = "2026-09-21T00:00:00Z"


def _load(name: str) -> tuple[bytes, str, dict]:
    raw = (_FIX / f"{name}.json").read_bytes()
    return raw, hashlib.sha256(raw).hexdigest(), json.loads(raw)


def _run(doc: dict):
    return verify(doc, ADAPTERS, context=VerificationContext(evaluation_time=_PINNED))


def _expiring() -> tuple[dict, str]:
    doc = json.loads((_CORPUS / "asqav-01-genesis-permit" / "receipt.json").read_bytes())
    doc["payload"]["expires_at"] = "2020-01-01T00:00:00Z"
    raw = json.dumps(doc, sort_keys=True).encode()
    return json.loads(raw.decode()), hashlib.sha256(raw).hexdigest()


def test_retained_then_live_agrees_without_mutating_store() -> None:
    _, digest, doc = _load("conflict-allow")
    store = AssessmentStore()
    record_verify_result(store, digest, _run(doc))
    out, case = classify_and_attribute(store, digest, _run(doc), _run(doc), _run(doc))
    assert out.label == "agreement" and out.differing == ()
    assert case.label == "not_applicable"
    assert len(store) == 1, "review is pure: nothing appended"


def test_missing_original_raises_instead_of_classifying() -> None:
    _, digest, doc = _load("conflict-allow")
    with pytest.raises(ValueError, match="no retained original"):
        classify_and_attribute(AssessmentStore(), digest, _run(doc))


def test_reassessment_split_attributes_through_the_caller() -> None:
    _, digest, doc = _load("conflict-allow")
    store = AssessmentStore()
    record_verify_result(store, digest, _run(doc))
    # SYNTHETIC verdict override (the only non-live element): one oracle
    # version cannot split on identical views; verdict folds from axes
    split = dataclasses.replace(_run(doc), verdict="unverified")
    out, case = classify_and_attribute(store, digest, split, _run(doc), _run(doc))
    assert out.label == "unexplained_needs_investigation"
    assert case.label == "spec_bug_candidate" and case.pinned == (digest,)


def test_moved_rerun_reports_flake_through_the_caller() -> None:
    doc, digest = _expiring()

    def era(t: str):
        return verify(doc, ADAPTERS, context=VerificationContext(evaluation_time=t))

    store = AssessmentStore()
    record_verify_result(store, digest, era(_T1))
    first = era(_T1)
    flipped = "unverified" if first.verdict == "verified" else "verified"
    # SYNTHETIC verdict override (the only non-live element); the rerun at
    # _T2 moves the expiry axis live, so the split cannot pin to a rerun
    split = dataclasses.replace(era(_T1), verdict=flipped)
    out, case = classify_and_attribute(store, digest, split, era(_T1), era(_T2))
    assert out.label == "unexplained_needs_investigation"
    assert case.label == "nondeterministic_flake" and "second" in case.note


def test_honest_drift_surfaces_through_the_retained_original() -> None:
    doc, digest = _expiring()

    def era(t: str):
        return verify(doc, ADAPTERS, context=VerificationContext(evaluation_time=t))

    store = AssessmentStore()
    record_verify_result(store, digest, era(_T1))
    out, case = classify_and_attribute(store, digest, era(_T2), era(_T2), era(_T2))
    assert out.label == "agreement" and "verification_time" in out.differing
    assert case.label == "not_applicable"
