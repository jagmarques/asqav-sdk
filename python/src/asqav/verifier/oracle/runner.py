"""Conformance-vector runner - drive the oracle over the vector corpus.

The corpus uses the AERF dir-per-vector layout as a superset: a top-level
``manifest.json`` lists ``{dir, format, outcome, reason_code, notes}`` and each
``<NN-name>/`` directory carries ``receipt.json``, an ``expected.json``, optional
``predecessor.json`` for chain vectors, and optional key material (``jwks.json``
for Asqav-native, ``keys.json`` mapping key_id->hex for AERF).

``outcome`` maps to the verdict one-for-one under the criterion 418 taxonomy:
PASS, INVALID (a binding check ran and failed), UNVERIFIABLE (recomputation
could not complete). A vector whose receipt file itself violates strict ingest
(a duplicate JSON member name, criterion 419) therefore never reaches hashing
or the signature check: the runner reports UNVERIFIABLE straight from the parse.
``run_corpus`` returns one ``VectorOutcome`` per vector so callers can assert
counts.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from asqav.strict_json import DuplicateJsonMemberError, strict_loads

from . import ADAPTERS
from .core import verify
from .taxonomy import PASS, UNVERIFIABLE

#: manifest outcome token -> the VerifyResult.verdict it must produce.
_OUTCOME_TO_VERDICT = {"PASS": PASS, "INVALID": "INVALID", "UNVERIFIABLE": UNVERIFIABLE}


@dataclass(frozen=True)
class VectorOutcome:
    """The result of running one corpus vector against the oracle.

    Fields:
      dir: the vector directory name.
      expected_outcome: the manifest's PASS / INVALID / UNVERIFIABLE outcome.
      actual_verdict: the oracle's PASS / INVALID / UNVERIFIABLE verdict.
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
    """Parse one vector file under strict ingest; None when the file is absent."""
    if not path.exists():
        return None
    return strict_loads(path.read_text())


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
    if fmt == "pipelock-evidence-v2":
        # keys.json: {signer_key_id: hex_32_bytes} - the raw Ed25519 key, no embedding.
        return _load(vec_dir / "keys.json")
    return None


def _parse_failure_outcome(
    vec_dir: Path, expected_outcome: str, reason_code: str, exc: Exception
) -> VectorOutcome:
    """Criterion 419: a strict-ingest failure ends the vector before any check.

    The duplicate member (or malformed JSON) never reaches hashing,
    canonicalisation, or the signature check, so the only verdict the oracle
    can honestly reach is UNVERIFIABLE.
    """
    reason = "duplicate_member" if isinstance(exc, DuplicateJsonMemberError) else "parse_failed"
    return VectorOutcome(
        vec_dir.name, expected_outcome, UNVERIFIABLE,
        expected_outcome == UNVERIFIABLE, reason_code, f"{reason}: {exc}",
    )


    # Run a single vector directory and compare against its expected outcome.
def run_one(vec_dir: Path, fmt: str, expected_outcome: str, reason_code: str = "") -> VectorOutcome:
    try:
        receipt = _load(vec_dir / "receipt.json")
        predecessor = _load(vec_dir / "predecessor.json")
        key_provider = _key_provider(vec_dir, fmt)
    except (DuplicateJsonMemberError, json.JSONDecodeError) as exc:
        return _parse_failure_outcome(vec_dir, expected_outcome, reason_code, exc)
    result = verify(receipt, ADAPTERS, key_provider=key_provider, predecessor=predecessor)

    want_verdict = _OUTCOME_TO_VERDICT.get(expected_outcome)
    ok = want_verdict is not None and result.verdict == want_verdict
    detail = "; ".join(
        f"{a.axis}={a.result}({a.reason_code}:{a.note})" for a in result.axes
    )
    return VectorOutcome(vec_dir.name, expected_outcome, result.verdict, ok, reason_code, detail)


    # Run every vector named in ``corpus_root/manifest.json``.
def run_corpus(corpus_root: Path) -> list[VectorOutcome]:
    # The manifest is a verification input too: strict ingest, no silent collapse
    manifest = strict_loads((corpus_root / "manifest.json").read_text())
    out = []
    for entry in manifest:
        out.append(
            run_one(corpus_root / entry["dir"], entry["format"], entry["outcome"], entry.get("reason_code", ""))
        )
    return out


    # Accept the stronger PASS only for the optional-dep ML-DSA skip vector, never broadly.
def _tolerated(outcome: VectorOutcome) -> bool:
    return outcome.ok or (
        outcome.expected_outcome == UNVERIFIABLE
        and outcome.actual_verdict == PASS
        and outcome.reason_code == "crypto_dependency_missing"
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


    # Run the bundled corpus and print a per-vector report; nonzero on mismatch.
def main() -> int:
    root = _default_corpus_root()
    results = run_corpus(root)
    for r in results:
        mark = "ok" if _tolerated(r) else "FAIL"
        print(f"  [{mark:>4}] {r.dir:<34} expect={r.expected_outcome:<12} got={r.actual_verdict}")
        if not _tolerated(r):
            print(f"         {r.detail}")
    passed = sum(1 for r in results if _tolerated(r))
    print(f"\n  => {passed}/{len(results)} vectors matched expected outcome")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
