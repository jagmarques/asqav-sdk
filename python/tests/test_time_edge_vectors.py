"""Time-edge conformance vectors, Python half (criterion 422).

DST transition days in both hemispheres, ambiguous and nonexistent local times,
explicit offsets around midnight, and far-past stamps within retention. One JSON
table drives both verifiers; the TypeScript half lives in
typescript/tests/verifier-time-edge-cases.test.ts and reads the same file.

Every case freezes the wall clock, so no verdict depends on the run date.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from asqav.verifier import verify_receipt as vr
from asqav.verifier.verify_receipt import SKEW_BOUND_SECONDS, _parse_stamp

CASES_FILE = Path(__file__).parent.parent.parent / "verifier" / "time-edge-cases.json"
TABLE = json.loads(CASES_FILE.read_text())
TIME_EDGE_VECTOR = (
    CASES_FILE.parent / "conformance-vectors" / "asqav-12-time-edge-expiry"
)

try:
    from dilithium_py.ml_dsa import ML_DSA_65 as _ML_DSA_CHECK  # noqa: F401

    _DILITHIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep, always present in CI
    _DILITHIUM_AVAILABLE = False


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)


    # A datetime subclass whose now() is pinned; check_skew/check_expiry read it
def _freeze(monkeypatch, iso: str) -> None:
    instant = _utc(iso)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz is not None else instant

    monkeypatch.setattr(vr, "datetime", Frozen)


    # A table that silently empties would make every test below vacuous
def test_table_is_populated() -> None:
    assert SKEW_BOUND_SECONDS == 300
    assert len(TABLE["instants"]) >= 4, f"only {len(TABLE['instants'])} instant vectors"
    assert len(TABLE["skew_bounds"]) >= 6, f"only {len(TABLE['skew_bounds'])} skew cases"
    assert len(TABLE["expiry"]) >= 6, f"only {len(TABLE['expiry'])} expiry cases"
    outcomes = {c["expect"]["result"] for c in TABLE["skew_bounds"]}
    assert outcomes == {"PASS", "FAIL"}, outcomes


@pytest.mark.parametrize("case", TABLE["instants"], ids=[c["name"] for c in TABLE["instants"]])
def test_parse_pins_the_utc_instant(case: dict) -> None:
    # Wire stamps carry explicit offsets; the parse result is the UTC instant
    ts = _parse_stamp(case["stamp"])
    assert ts is not None, case["name"]
    assert ts.astimezone(timezone.utc).isoformat() == case["utc"], case["name"]


@pytest.mark.parametrize("case", TABLE["instants"], ids=[c["name"] for c in TABLE["instants"]])
def test_instant_probes_pin_normalisation(case: dict, monkeypatch) -> None:
    # Frozen at utc-301s the stamp is exactly 301s ahead and FAILs; at utc-300s
    # and utc it PASSes. Only the true UTC instant draws all three verdicts, so
    # the probes pin both the instant and the future-only 300s bound
    for probe in case["probes"]:
        _freeze(monkeypatch, probe["clock"])
        result, note = vr.check_skew(case["stamp"])
        assert result == probe["result"], f"{case['name']} @ {probe['clock']}: {note}"
        assert note == probe["note"], f"{case['name']} @ {probe['clock']}: {note}"


def test_ambiguous_local_time_resolves_to_two_instants() -> None:
    # The fall-back wall clock 01:30 occurs twice; the offsets pick the instant
    first = next(c for c in TABLE["instants"] if c["name"] == "north-fall-back-first-occurrence")
    second = next(
        c for c in TABLE["instants"] if c["name"] == "north-fall-back-second-occurrence"
    )
    assert first["stamp"][:16] == second["stamp"][:16], "same wall clock, two offsets"
    delta = _utc(second["utc"]) - _utc(first["utc"])
    assert delta.total_seconds() == 3600, delta


@pytest.mark.parametrize(
    "case", TABLE["skew_bounds"], ids=[c["name"] for c in TABLE["skew_bounds"]]
)
def test_skew_bound_case(case: dict, monkeypatch) -> None:
    _freeze(monkeypatch, TABLE["frozen_clock"])
    result, note = vr.check_skew(case["issued_at"])
    expect = case["expect"]
    assert result == expect["result"], f"{case['name']}: {note}"
    if expect.get("exact"):
        assert note == expect["note"], case["name"]
    else:
        assert expect["note_contains"] in note, case["name"]


def test_far_past_passes_exactly_like_a_fresh_stamp(monkeypatch) -> None:
    # Retention-bounded past is unbounded: a ~3-year-old stamp draws the same
    # PASS a fresh one does, under the same frozen clock
    _freeze(monkeypatch, TABLE["frozen_clock"])
    fresh = next(c for c in TABLE["skew_bounds"] if c["name"] == "fresh-at-frozen-clock")
    old = next(c for c in TABLE["skew_bounds"] if c["name"] == "far-past-three-years")
    assert vr.check_skew(fresh["issued_at"])[0] == "PASS"
    assert vr.check_skew(old["issued_at"])[0] == "PASS"
    assert "within bound" in vr.check_skew(old["issued_at"])[1]


@pytest.mark.parametrize("case", TABLE["expiry"], ids=[c["name"] for c in TABLE["expiry"]])
def test_expiry_case(case: dict, monkeypatch) -> None:
    _freeze(monkeypatch, TABLE["frozen_clock"])
    result, note = vr.check_expiry(case["payload"])
    assert result == case["expect"]["result"], f"{case['name']}: {note}"
    assert note == case["expect"]["note"], case["name"]


def test_unreadable_expires_at_fails_closed_on_its_own_axis(monkeypatch) -> None:
    _freeze(monkeypatch, TABLE["frozen_clock"])
    result, note = vr.check_expiry({"expires_at": "not-a-stamp"})
    assert result == "FAIL"
    assert "refused rather than read as no expiry" in note


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_time_edge_corpus_vector_expiry_never_folds_the_verdict() -> None:
    # asqav-12: extreme +14:00 issued_at (past, skew PASS) and a lapsed signed
    # expires_at; the expiry axis FAILs alone and the verdict stays PASS (426)
    receipt = json.loads((TIME_EDGE_VECTOR / "receipt.json").read_text())
    jwks = json.loads((TIME_EDGE_VECTOR / "jwks.json").read_text())
    result = vr.run_structured(receipt, jwks)
    axes = {a["name"]: a for a in result["axes"]}
    assert result["verdict"] == "verified", result["axes"]
    assert axes["skew"]["result"] == "PASS", axes["skew"]
    assert axes["expiry"]["result"] == "FAIL", axes["expiry"]
    assert "lapsed" in axes["expiry"]["note"], axes["expiry"]
    non_expiry = [a for a in result["axes"] if a["name"] != "expiry"]
    assert all(a["result"] == "PASS" for a in non_expiry), non_expiry


@pytest.mark.skipif(not _DILITHIUM_AVAILABLE, reason="dilithium-py not installed")
def test_time_edge_corpus_vector_matches_its_expected_outcome() -> None:
    expected = json.loads((TIME_EDGE_VECTOR / "expected.json").read_text())
    assert expected["outcome"] == "verified"
    from asqav.verifier.oracle.runner import run_one

    outcome = run_one(
        TIME_EDGE_VECTOR, expected["format"], expected["outcome"], expected["reason_code"]
    )
    assert outcome.ok, outcome.detail
