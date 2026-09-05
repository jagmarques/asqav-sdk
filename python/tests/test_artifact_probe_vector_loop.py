# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""The artifact probe's corpus loop runs each vector the way the requirement-map
builder does.

The probe writes its loop as a source string and executes it inside a throwaway
venv against the INSTALLED package, so a repository test cannot call it
directly. What it can do is execute the same bytes: ``_CORPUS_RUNNER_SOURCE``
is the module-level constant the probe writes, and this file exec's it against
the source package, so a drift between the loop the probe ships and the loop
under test fails here instead of in the weekly artifact.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

from asqav.verifier import verify_receipt as vr

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VECTORS = _REPO_ROOT / "verifier" / "conformance-vectors"


def _load_loop_source() -> str:
    spec = importlib.util.spec_from_file_location(
        "probe_published_artifacts",
        _REPO_ROOT / "verifier" / "artifact_probe" / "probe_published_artifacts.py",
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    return probe._CORPUS_RUNNER_SOURCE


_LOOP_SOURCE = _load_loop_source()


def _run_corpus_loop(monkeypatch, spy=None) -> dict:
    """Execute the probe's loop source in-process against the local corpus.

    The source reads the corpus path from sys.argv and prints one JSON object;
    both are captured. A spy, when given, replaces run_structured BEFORE the
    exec: the loop's own ``from ... import run_structured`` resolves the
    attribute at exec time, so it binds the spy, and the spy records exactly
    what the loop hands the verifier.
    """
    if spy is not None:
        monkeypatch.setattr(vr, "run_structured", spy)
    monkeypatch.setattr(sys, "argv", ["run_corpus.py", str(_VECTORS)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(_LOOP_SOURCE, "run_corpus.py", "exec"), {})
    return json.loads(buf.getvalue())


def test_asqav_24_observes_verified_with_its_shipped_anchor_material(monkeypatch) -> None:
    """asqav-24 ships tsa_trust.pem + bitcoin_headers.json; handed in the builder's
    shapes, the anchors axis completes and the declared verified reproduces."""
    out = _run_corpus_loop(monkeypatch)
    r24 = out["asqav-24-anchor-block-hash-prod"]
    assert r24["declared"] == "verified"
    assert r24["observed"] == "verified", r24


def test_every_drift_entry_carries_an_axis_name_and_a_note(monkeypatch) -> None:
    """The drift report names, per vector, the first non-PASS axis and its note.

    Every drifting vector in this corpus has a non-PASS axis to name (the
    all-PASS fallback in the probe covers a shape no current vector produces).
    """
    out = _run_corpus_loop(monkeypatch)
    drift = {
        n: r
        for n, r in out.items()
        if not str(r["observed"]).startswith("ERROR") and r["declared"] != r["observed"]
    }
    assert drift, "the drift set is empty; the assertion would be vacuous"
    for name, r in sorted(drift.items()):
        first = r["first_nonpass"]
        assert first is not None, f"{name}: drift without a named axis"
        assert first["name"], f"{name}: empty axis name in {first}"
        assert first["note"], f"{name}: empty axis note in {first}"


def test_predecessor_unwrapped_and_anchor_material_passed(monkeypatch) -> None:
    """What run_structured receives: the predecessor PAYLOAD (not the envelope),
    and the vector's own anchor material in the builder's shapes."""
    calls = []
    real = vr.run_structured

    def spy(envelope, jwks, predecessor_payload=None, **kwargs):
        calls.append(
            {
                "envelope": envelope,
                "jwks": jwks,
                "predecessor_payload": predecessor_payload,
                **kwargs,
            }
        )
        return real(envelope, jwks, predecessor_payload, **kwargs)

    _run_corpus_loop(monkeypatch, spy=spy)

    receipt03 = json.loads((_VECTORS / "asqav-03-chain-link" / "receipt.json").read_text())
    pred03 = json.loads((_VECTORS / "asqav-03-chain-link" / "predecessor.json").read_text())
    call03 = next(c for c in calls if c["envelope"] == receipt03)
    assert call03["predecessor_payload"] == pred03["payload"], (
        "the loop must hand run_structured the predecessor's payload member, "
        "the same unwrap the requirement-map builder does"
    )

    receipt24 = json.loads(
        (_VECTORS / "asqav-24-anchor-block-hash-prod" / "receipt.json").read_text()
    )
    call24 = next(c for c in calls if c["envelope"] == receipt24)
    tsa_keys = call24["trusted_tsa_keys"]
    assert isinstance(tsa_keys, list) and tsa_keys and all(
        isinstance(k, bytes) for k in tsa_keys
    ), "tsa_trust.pem must pass as a list of raw bytes, the builder's shape"
    assert isinstance(call24["bitcoin_headers"], dict), (
        "bitcoin_headers.json must pass as parsed JSON, keyed by height as a string"
    )
