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
def test_at_least_two_engines_are_compared() -> None:
    engines = fuzz.engines_available()
    assert len(engines) >= 2, (
        f"only {engines} available; build typescript/dist so the gate is differential"
    )


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
