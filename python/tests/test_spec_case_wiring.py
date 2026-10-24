"""Wire spec_case into the live path (488 C25, C2 core).
attribute_live_split maps live results (+ live reruns) to views, then classifies and attributes;
attribute_split is the same pipeline over retained views. Reachable branches run fully live. The
candidate branch cannot: verdicts fold from axes (oracle semantic table), so one oracle version
never splits on identical views - the split there is synthetic and marked as such, while digests,
axes, contexts and reruns stay live."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="signed fixtures need the cryptography lib")

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.live_compare import attribute_live_split, attribute_split
from asqav.verifier.oracle.retain import axes_dict, context_dict

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


def _view(result, digest: str) -> dict:
    return {"doc_sha256": digest, "axes": axes_dict(result),
            "context": context_dict(result), "agreement": None}


def _expiring() -> tuple[dict, str]:
    doc = json.loads((_CORPUS / "asqav-01-genesis-permit" / "receipt.json").read_bytes())
    doc["payload"]["expires_at"] = "2020-01-01T00:00:00Z"
    raw = json.dumps(doc, sort_keys=True).encode()
    return json.loads(raw.decode()), hashlib.sha256(raw).hexdigest()


def test_live_agreement_needs_no_attribution() -> None:
    raw, digest, doc = _load("conflict-allow")
    assert _run(doc).verdict == "verified"
    out, case = attribute_live_split(
        digest, _run(doc), digest, _run(doc), _run(doc), _run(doc), first_bytes=raw)
    assert out.label == "agreement" and out.differing == ()
    assert case.label == "not_applicable"


def test_live_drift_pair_needs_no_attribution() -> None:
    doc, digest = _expiring()

    def era(t: str):
        return verify(doc, ADAPTERS, context=VerificationContext(evaluation_time=t))

    out, case = attribute_live_split(digest, era(_T1), digest, era(_T2), era(_T1), era(_T2))
    assert out.label == "agreement" and "verification_time" in out.differing
    assert case.label == "not_applicable"


def test_live_conflict_stays_split_without_fusion() -> None:
    allow_raw, allow_digest, allow = _load("conflict-allow")
    _, deny_digest, deny = _load("conflict-deny")
    out, case = attribute_live_split(
        allow_digest, _run(allow), deny_digest, _run(deny), _run(allow), _run(deny))
    assert out.label == "different_subject" and out.differing == ()
    assert case.label == "not_applicable"


def test_live_rerun_bytes_must_match() -> None:
    _, digest, doc = _load("conflict-allow")
    with pytest.raises(ValueError, match="rerun_a digest"):
        attribute_live_split(digest, _run(doc), digest, _run(doc),
                             _run(doc), _run(doc), rerun_a_bytes=b"not the receipt")


def test_rerun_bytes_bind_per_side_on_split_digests() -> None:
    allow_raw, allow_digest, allow = _load("conflict-allow")
    deny_raw, deny_digest, deny = _load("conflict-deny")
    assert allow_digest != deny_digest, "sides differ or the binding is vacuous"
    out, case = attribute_live_split(
        allow_digest, _run(allow), deny_digest, _run(deny),
        _run(allow), _run(deny), rerun_a_bytes=allow_raw, rerun_b_bytes=deny_raw)
    assert out.label == "different_subject" and case.label == "not_applicable"


def test_swapped_rerun_bytes_raise_per_side() -> None:
    allow_raw, allow_digest, allow = _load("conflict-allow")
    deny_raw, deny_digest, deny = _load("conflict-deny")
    with pytest.raises(ValueError, match="rerun_a digest"):
        attribute_live_split(
            allow_digest, _run(allow), deny_digest, _run(deny),
            _run(allow), _run(deny), rerun_a_bytes=deny_raw, rerun_b_bytes=allow_raw)


def test_candidate_pins_the_live_digest_through_the_pipeline() -> None:
    raw, digest, doc = _load("conflict-allow")
    first, second = _run(doc), _run(doc)
    assert _view(first, digest) == _view(second, digest), "oracle is deterministic"
    assert len(first.axes) > 0, "axes are real oracle output, not empty"
    # SYNTHETIC verdict override (the only non-live element): one oracle
    # version cannot split on identical views; verdict folds from axes
    split = dataclasses.replace(second, verdict="unverified")
    out, case = attribute_live_split(
        digest, first, digest, split, _run(doc), _run(doc), first_bytes=raw)
    assert out.label == "unexplained_needs_investigation"
    assert case.label == "spec_bug_candidate" and case.pinned == (digest,)


def test_pipeline_without_the_split_stays_not_applicable() -> None:
    _, digest, doc = _load("conflict-allow")
    first, second = _run(doc), _run(doc)
    out, case = attribute_split(
        _view(first, digest), _view(second, digest), first.verdict, second.verdict,
        _view(_run(doc), digest), _view(_run(doc), digest))
    assert out.label == "agreement" and case.label == "not_applicable"


def test_pipeline_with_moved_live_rerun_reports_flake() -> None:
    doc, digest = _expiring()

    def era(t: str):
        return verify(doc, ADAPTERS, context=VerificationContext(evaluation_time=t))

    first, second = era(_T1), era(_T1)
    assert _view(first, digest) == _view(second, digest), "same era, same views"
    flipped = "unverified" if first.verdict == "verified" else "verified"
    # SYNTHETIC verdict override (the only non-live element); the rerun at
    # _T2 moves the expiry axis live, so the split cannot pin to a rerun
    split = dataclasses.replace(second, verdict=flipped)
    out, case = attribute_live_split(digest, first, digest, split, era(_T1), era(_T2))
    assert out.label == "unexplained_needs_investigation"
    assert case.label == "nondeterministic_flake" and "second" in case.note
