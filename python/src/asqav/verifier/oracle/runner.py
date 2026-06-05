"""Conformance-vector runner - drive the oracle over the vector corpus.

The corpus uses the AERF dir-per-vector layout as a superset: a top-level
``manifest.json`` lists ``{dir, format, outcome, reason_code, notes}`` and each
``<NN-name>/`` directory carries ``receipt.json``, an ``expected.json``, optional
``predecessor.json`` for chain vectors, and optional key material (``jwks.json``
for Asqav-native, ``keys.json`` mapping key_id->hex for AERF).

``outcome`` maps to the verdict one-for-one (PASS/FAIL/INCOMPLETE). INCOMPLETE
lets the corpus carry a vector whose signature axis cannot be checked in CI (a
real ML-DSA-65 prod receipt without dilithium-py), pinning that the verifier
downgrades rather than false-PASSing. ``run_corpus`` returns one ``VectorOutcome``
per vector so callers can assert counts.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import ADAPTERS
from .core import verify

#: manifest outcome token -> the VerifyResult.verdict it must produce.
_OUTCOME_TO_VERDICT = {"PASS": "PASS", "FAIL": "FAIL", "INCOMPLETE": "INCOMPLETE"}


@dataclass(frozen=True)
class VectorOutcome:
    """The result of running one corpus vector against the oracle.

    Fields:
      dir: the vector directory name.
      expected_outcome: the manifest's PASS / FAIL / INCOMPLETE outcome.
      actual_verdict: the oracle's PASS / FAIL / INCOMPLETE verdict.
      ok: True when the actual verdict matches the expected outcome.
      reason_code: the manifest reason_code for this vector.
      detail: per-axis notes for a failing match.
    """

    dir: str
    expected_outcome: str
    actual_verdict: str
    ok: bool
    reason_code: str
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


def run_one(vec_dir: Path, fmt: str, expected_outcome: str, reason_code: str = "") -> VectorOutcome:
    """Run a single vector directory and compare against its expected outcome."""
    receipt = _load(vec_dir / "receipt.json")
    predecessor = _load(vec_dir / "predecessor.json")
    key_provider = _key_provider(vec_dir, fmt)
    result = verify(receipt, ADAPTERS, key_provider=key_provider, predecessor=predecessor)

    want_verdict = _OUTCOME_TO_VERDICT.get(expected_outcome)
    ok = want_verdict is not None and result.verdict == want_verdict
    detail = "; ".join(f"{a.axis}={a.result}({a.note})" for a in result.axes)
    return VectorOutcome(vec_dir.name, expected_outcome, result.verdict, ok, reason_code, detail)


def run_corpus(corpus_root: Path) -> list[VectorOutcome]:
    """Run every vector named in ``corpus_root/manifest.json``."""
    manifest = json.loads((corpus_root / "manifest.json").read_text())
    out = []
    for entry in manifest:
        out.append(
            run_one(corpus_root / entry["dir"], entry["format"], entry["outcome"], entry.get("reason_code", ""))
        )
    return out


def _tolerated(outcome: VectorOutcome) -> bool:
    """Accept the stronger PASS only for the optional-dep ML-DSA skip vector, never broadly."""
    return outcome.ok or (
        outcome.expected_outcome == "INCOMPLETE"
        and outcome.actual_verdict == "PASS"
        and outcome.reason_code == "signature_skipped_no_dilithium"
    )


def _default_corpus_root() -> Path:
    """Resolve the conformance-vector corpus at the repo root for dev/CI runs.

    The corpus stays at ``<repo>/verifier/conformance-vectors`` so the TypeScript
    parity gate and the published governance URL keep pointing at one place. An
    installed wheel does not ship the corpus, so this default only resolves in a
    source checkout; the ``ASQAV_CONFORMANCE_VECTORS`` env var overrides it.
    """
    env = os.environ.get("ASQAV_CONFORMANCE_VECTORS")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    # oracle -> verifier -> asqav -> src -> python -> <repo root>
    return here.parents[5] / "verifier" / "conformance-vectors"


def main() -> int:
    """Run the bundled corpus and print a per-vector report; nonzero on mismatch."""
    root = _default_corpus_root()
    results = run_corpus(root)
    for r in results:
        mark = "ok" if _tolerated(r) else "FAIL"
        print(f"  [{mark:>4}] {r.dir:<28} expect={r.expected_outcome:<5} got={r.actual_verdict}")
        if not _tolerated(r):
            print(f"         {r.detail}")
    passed = sum(1 for r in results if _tolerated(r))
    print(f"\n  => {passed}/{len(results)} vectors matched expected outcome")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
