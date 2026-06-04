"""Conformance-vector runner - drive the oracle over the vector corpus.

The corpus uses the AERF dir-per-vector layout as a superset: a top-level
``manifest.json`` lists ``{dir, format, outcome, reason_code, notes}`` and each
``<NN-name>/`` directory carries ``receipt.json``, an ``expected.json``, optional
``predecessor.json`` for chain vectors, and optional key material (``jwks.json``
for Asqav-native, ``keys.json`` mapping key_id->hex for AERF).

``outcome`` is mapped to the oracle verdict: PASS -> verdict PASS, FAIL -> verdict
FAIL. ``run_corpus`` returns one ``VectorOutcome`` per vector so callers (pytest
or the CLI) can assert pass/fail counts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import ADAPTERS
from .core import verify

#: manifest outcome token -> the VerifyResult.verdict it must produce.
_OUTCOME_TO_VERDICT = {"PASS": "PASS", "FAIL": "FAIL"}


@dataclass(frozen=True)
class VectorOutcome:
    """The result of running one corpus vector against the oracle.

    Fields:
      dir: the vector directory name.
      expected_outcome: the manifest's PASS / FAIL outcome.
      actual_verdict: the oracle's PASS / FAIL / INCOMPLETE verdict.
      ok: True when the actual verdict matches the expected outcome.
      detail: per-axis notes for a failing match.
    """

    dir: str
    expected_outcome: str
    actual_verdict: str
    ok: bool
    detail: str


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _key_provider(vec_dir: Path, fmt: str):
    if fmt == "asqav-native":
        return _load(vec_dir / "jwks.json")
    if fmt == "aerf":
        return _load(vec_dir / "keys.json")
    if fmt == "acta":
        return _load(vec_dir / "acta-keys.json")
    if fmt == "agentreceipts":
        # did:key receipts self-resolve (no file); did:agent/web carry an injected map.
        return _load(vec_dir / "did_map.json")
    return None


def run_one(vec_dir: Path, fmt: str, expected_outcome: str) -> VectorOutcome:
    """Run a single vector directory and compare against its expected outcome."""
    receipt = _load(vec_dir / "receipt.json")
    predecessor = _load(vec_dir / "predecessor.json")
    key_provider = _key_provider(vec_dir, fmt)
    result = verify(receipt, ADAPTERS, key_provider=key_provider, predecessor=predecessor)

    want_verdict = _OUTCOME_TO_VERDICT.get(expected_outcome)
    ok = want_verdict is not None and result.verdict == want_verdict
    detail = "; ".join(f"{a.axis}={a.result}({a.note})" for a in result.axes)
    return VectorOutcome(vec_dir.name, expected_outcome, result.verdict, ok, detail)


def run_corpus(corpus_root: Path) -> list[VectorOutcome]:
    """Run every vector named in ``corpus_root/manifest.json``."""
    manifest = json.loads((corpus_root / "manifest.json").read_text())
    out = []
    for entry in manifest:
        out.append(
            run_one(corpus_root / entry["dir"], entry["format"], entry["outcome"])
        )
    return out


def main() -> int:
    """Run the bundled corpus and print a per-vector report; nonzero on mismatch."""
    root = Path(__file__).resolve().parent.parent / "conformance-vectors"
    results = run_corpus(root)
    for r in results:
        mark = "ok" if r.ok else "FAIL"
        print(f"  [{mark:>4}] {r.dir:<28} expect={r.expected_outcome:<5} got={r.actual_verdict}")
        if not r.ok:
            print(f"         {r.detail}")
    passed = sum(1 for r in results if r.ok)
    print(f"\n  => {passed}/{len(results)} vectors matched expected outcome")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
