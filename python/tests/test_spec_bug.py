"""Spec-bug evidence bar (488 C23, C2 core).

An unexplained split becomes an attributable spec_bug_candidate only
when both inputs re-execute to identical axes on identical digests.
Unit views follow the C13 pattern; the live probe proves re-execution
determinism on real oracle output (the property the bar relies on).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="live probe needs the cryptography lib")

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.disagreement import SpecCase, classify, spec_case
from asqav.verifier.oracle.retain import axes_dict, context_dict

_CTX = {
    "evaluator": "c23-probe",
    "trust_roots": ["root-a"],
    "revocation_snapshot": "snap-1",
    "evaluation_time": "2029-06-01T00:00:00Z",
}
_AXES = {"expiry": "PASS", "skew": "PASS"}
_DOC = "ee" * 32
_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"


def _view(**over: object) -> dict:
    base: dict = {
        "doc_sha256": _DOC,
        "axes": dict(_AXES),
        "context": dict(_CTX),
        "agreement": None,
    }
    for key, value in over.items():
        base[key] = value
    return base


def _unexplained() -> tuple[dict, dict]:
    return _view(), _view()


def test_agreement_is_not_applicable() -> None:
    out = spec_case(_view(), _view(), "verified", "verified", _view(), _view())
    assert out.label == "not_applicable" and out.pinned == ()


def test_different_subject_is_not_applicable() -> None:
    first, second = _view(), _view(doc_sha256="ff" * 32)
    assert classify(first, second, "verified", "unverified").label == "different_subject"
    assert spec_case(first, second, "verified", "unverified", first, second).label == "not_applicable"


def test_unexplained_without_reruns_is_insufficient() -> None:
    first, second = _unexplained()
    out = spec_case(first, second, "verified", "unverified")
    assert isinstance(out, SpecCase) and out.label == "evidence_insufficient"


def test_rerun_on_other_bytes_is_insufficient() -> None:
    first, second = _unexplained()
    out = spec_case(first, second, "verified", "unverified", _view(doc_sha256="ff" * 32), second)
    assert out.label == "evidence_insufficient"


def test_digestless_split_cannot_be_a_candidate() -> None:
    first, second = _view(doc_sha256=None), _view(doc_sha256=None)
    assert classify(first, second, "verified", "unverified").label == "unexplained_needs_investigation"
    out = spec_case(first, second, "verified", "unverified", _view(doc_sha256=None), _view(doc_sha256=None))
    assert out.label == "evidence_insufficient" and out.pinned == ()


def test_moved_axes_are_a_flake_not_a_spec_bug() -> None:
    first, second = _unexplained()
    moved = _view(axes={"expiry": "FAIL", "skew": "PASS"})
    out = spec_case(first, second, "verified", "unverified", moved, second)
    assert out.label == "nondeterministic_flake" and out.pinned == ()


def test_reproduced_split_is_a_candidate_with_pinned_bytes() -> None:
    first, second = _unexplained()
    out = spec_case(first, second, "verified", "unverified", _view(), _view())
    assert out.label == "spec_bug_candidate" and out.pinned == (_DOC,)


def test_live_reexecution_reproduces_axes() -> None:
    raw = (_FIX / "conflict-allow.json").read_bytes()
    doc = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    ctx = VerificationContext(evaluation_time="2026-10-16T12:00:00Z", evaluator="c23-probe")
    views = []
    for _ in range(2):
        res = verify(doc, ADAPTERS, context=ctx)
        views.append({"doc_sha256": digest, "axes": axes_dict(res),
                      "context": context_dict(res), "agreement": None})
    assert views[0]["axes"] == views[1]["axes"] != {}
    assert views[0] == views[1], "re-execution from retained bytes reproduces the view"
