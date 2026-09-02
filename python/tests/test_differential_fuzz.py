"""Differential fuzz gate: the canonicalizers must not disagree.

Runs a fixed, seeded corpus so a failure is reproducible from the seed printed in
the report. The TypeScript engine joins only when ``typescript/dist`` is built;
the Python engines always compare, so the gate never silently degrades to one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "verifier"))

import differential_fuzz as fuzz  # noqa: E402

SEEDS = [0, 1, 2, 3]
ITERATIONS = 250


    # One engine cannot diverge from itself, so a green run would prove nothing.
def test_at_least_two_engines_are_compared(capsys: pytest.CaptureFixture[str]) -> None:
    engines = fuzz.engines_available()
    # Printed so a CI log records WHICH engines ran: a silently degraded roster
    # still passes every divergence test, because fewer engines cannot disagree.
    with capsys.disabled():
        print(f"engines compared: {', '.join(engines)}")
    assert len(engines) >= 2, (
        f"only {engines} available; build typescript/dist so the gate is differential"
    )


def test_a_built_typescript_bundle_joins_both_of_its_engines() -> None:
    """With typescript/dist built, all four engines compare, verifier included.

    The verifier engine is the one the SDK emitter cannot stand in for: it is the
    canonicalizer a third party runs, so a roster missing it hides exactly the
    divergence class the corpus exists to catch.
    """
    dist = (
        Path(__file__).resolve().parents[2]
        / "typescript"
        / "dist"
        / "verifier"
        / "index.js"
    )
    if not dist.exists():
        pytest.skip("typescript/dist is not built in this environment")
    engines = fuzz.engines_available()
    assert "typescript" in engines and "typescript-verifier" in engines, engines
    assert len(engines) >= 4, f"built bundle but only {engines} compared"


@pytest.mark.parametrize("seed", SEEDS)
def test_no_divergence_between_canonicalizers(seed: int) -> None:
    divergences = fuzz.run(iterations=ITERATIONS, seed=seed, unsafe_numbers=False)
    assert not divergences, (
        f"canonicalizer divergence at seed {seed}: "
        f"{json.dumps(divergences[0], ensure_ascii=False)[:800]}"
    )


    # The corpus must actually reach the astral keys it exists to cover.
def test_corpus_generates_supplementary_plane_keys() -> None:
    import random

    rng = random.Random(0)
    docs = [fuzz.generate(rng) for _ in range(ITERATIONS)]

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    astral = {k for d in docs for k in keys(d) if any(ord(c) > 0xFFFF for c in k)}
    assert astral, "corpus reached no supplementary-plane member name"


    # A code-point sort must be caught, or a green run proves nothing.
def test_gate_detects_a_code_point_sorting_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def code_point_canonical(obj: object) -> bytes:
        return json.dumps(
            obj, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")

    monkeypatch.setattr(fuzz, "sdk_canonical", code_point_canonical)
    assert fuzz.run(iterations=ITERATIONS, seed=0, unsafe_numbers=False), (
        "the fuzz gate passed an engine sorting by code point; it has no bite"
    )


    # The engine a customer actually runs offline must be compared, not only the emitter.
def test_standalone_verifier_is_one_of_the_engines() -> None:
    assert "standalone" in fuzz.engines_available()


    # The verifier's own asqav dialect joins whenever the verifier bundle is built.
def test_typescript_verifier_engine_joins_when_built() -> None:
    dist = _ROOT / "typescript" / "dist" / "verifier" / "index.js"
    if not dist.exists():
        pytest.skip("typescript/dist/verifier not built")
    assert "typescript-verifier" in fuzz.engines_available()


    # A verification path sorting by code point must be caught, or the gate is blind to it.
def test_gate_detects_a_code_point_sorting_standalone_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def code_point_canonical(obj: object) -> bytes:
        return json.dumps(
            obj, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")

    monkeypatch.setattr(fuzz, "standalone_canonical", code_point_canonical)
    assert fuzz.run(iterations=ITERATIONS, seed=0, unsafe_numbers=False), (
        "the fuzz gate passed a standalone verifier sorting by code point; it has no bite"
    )

