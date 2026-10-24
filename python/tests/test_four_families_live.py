"""Four disagreement families through the live oracle (488 C22, C7 core).

Unifies C14's synthetic families with C21's real signatures and C17's
time drift: every family runs verify() + compare_live() over real oracle
output, never hand-built views. Success agrees; contradictory signed
terms stay a different subject; dropped evidence fails only the
signature axis; trust/time drift agrees with verification_time recorded.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from asqav.verifier.oracle import ADAPTERS, VerificationContext, verify
from asqav.verifier.oracle.live_compare import compare_live

_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"
_T1 = "2026-10-16T12:00:00Z"
_T2 = "2031-01-01T00:00:00Z"


def _load(name: str) -> tuple[dict, str]:
    raw = (_FIX / f"{name}.json").read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _ctx(evaluation_time: str = _T1) -> VerificationContext:
    return VerificationContext(
        evaluation_time=evaluation_time,
        trust_roots=["root-a"],
        revocation_snapshot="snap-1",
        evaluator="c22-probe",
    )


def test_family_1_success_agrees_live() -> None:
    doc, digest = _load("conflict-allow")
    out = compare_live(digest, verify(doc, ADAPTERS, context=_ctx()),
                       digest, verify(doc, ADAPTERS, context=_ctx()))
    assert out.label == "agreement" and out.differing == ()


def test_family_2_contradictory_signed_terms_stay_split() -> None:
    allow, allow_digest = _load("conflict-allow")
    deny, deny_digest = _load("conflict-deny")
    assert allow_digest != deny_digest, "conflicting terms are different bytes"
    ra, rd = (verify(allow, ADAPTERS, context=_ctx()),
              verify(deny, ADAPTERS, context=_ctx()))
    assert (ra.verdict, rd.verdict) == ("verified", "verified")
    out = compare_live(allow_digest, ra, deny_digest, rd)
    assert out.label == "different_subject" and out.differing == ()


def test_family_3_dropped_evidence_fails_only_signature() -> None:
    doc, digest = _load("conflict-allow")
    dropped = copy.deepcopy(doc)
    del dropped["credentialSubject"]["outcome"]
    raw = json.dumps(dropped, sort_keys=True).encode()
    missing = verify(json.loads(raw), ADAPTERS, context=_ctx())
    assert missing.verdict == "unverified"
    assert [a.axis for a in missing.axes if a.result != "PASS"] == ["signature"]
    out = compare_live(digest, verify(doc, ADAPTERS, context=_ctx()),
                       hashlib.sha256(raw).hexdigest(), missing)
    assert out.label == "different_subject" and out.differing == ()


def test_family_4_trust_time_drift_agrees_with_drift_recorded() -> None:
    doc, digest = _load("conflict-allow")
    early = verify(doc, ADAPTERS, context=_ctx(_T1))
    late = verify(doc, ADAPTERS, context=_ctx(_T2))
    assert (early.verdict, late.verdict) == ("verified", "verified")
    out = compare_live(digest, early, digest, late)
    assert out.label == "agreement" and out.differing == ("verification_time",)
