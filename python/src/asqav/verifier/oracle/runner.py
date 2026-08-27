"""Conformance-vector runner - drive the oracle over the vector corpus.

The corpus uses the AERF dir-per-vector layout as a superset: a top-level
``manifest.json`` lists ``{dir, format, outcome, failure_class, reason_code,
notes}`` and each ``<NN-name>/`` directory carries ``receipt.json``, an
``expected.json``, optional ``predecessor.json`` for chain vectors, and optional
key material (``jwks.json`` for Asqav-native, ``keys.json`` mapping key_id->hex
for AERF).

``outcome`` speaks the public verdict vocabulary (criteria 418/438):
verified / verified_keyed / unverified, and an ``unverified`` entry pins its
``failure_class`` (invalid / unverifiable) so the two classes are never
collapsed. ``run_corpus`` returns one ``VectorOutcome`` per vector so callers
can assert counts. Every receipt/record file is parsed with duplicate-member
rejection (criterion 419); a receipt that fails to parse is a terminal
unverified/unverifiable outcome, never verified.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from asqav import strict_json

from . import ADAPTERS
from .core import FAILURE_UNVERIFIABLE, VERDICT_UNVERIFIED, VERDICT_VERIFIED, verify

#: manifest outcome tokens -> the VerifyResult.verdict they must produce.
_OUTCOME_TO_VERDICT = {
    "verified": "verified",
    "verified_keyed": "verified_keyed",
    "unverified": "unverified",
}


@dataclass(frozen=True)
class VectorOutcome:
    """The result of running one corpus vector against the oracle.

    Fields:
      dir: the vector directory name.
      expected_outcome: the manifest's verified / verified_keyed / unverified.
      actual_verdict: the oracle's verdict in the same vocabulary.
      ok: True when the actual verdict (and pinned failure class) matches.
      reason_code: the manifest reason_code for this vector.
      detail: per-axis notes for a failing match.
      expected_failure_class: the manifest failure_class pin ("" when unpinned).
      actual_failure_class: the oracle failure_class ("" on a verified verdict).
    """

    dir: str
    expected_outcome: str
    actual_verdict: str
    ok: bool
    reason_code: str
    detail: str
    expected_failure_class: str = ""
    actual_failure_class: str = ""


def _load(path: Path) -> dict | None:
    # Strict ingest (419): a duplicated member name is a terminal parse failure.
    return strict_json.loads(path.read_text()) if path.exists() else None


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
    if fmt == "w3c-vc":
        # did:web resolves from an injected DID-document map; the oracle never fetches.
        return _load(vec_dir / "did_map.json")
    if fmt == "pipelock-evidence-v2":
        # keys.json: {signer_key_id: hex_32_bytes} - the raw Ed25519 key, no embedding.
        return _load(vec_dir / "keys.json")
    return None


def _parse_failure_outcome(
    name: str, expected_outcome: str, expected_failure_class: str, reason_code: str, exc: Exception
) -> VectorOutcome:
    # A terminal ingest failure: nothing was hashed, checked, or verified (419).
    detail = f"ingest=FAIL(terminal parse failure before any hashing: {exc})"
    ok = expected_outcome == VERDICT_UNVERIFIED and expected_failure_class in (
        "",
        FAILURE_UNVERIFIABLE,
    )
    return VectorOutcome(
        name,
        expected_outcome,
        VERDICT_UNVERIFIED,
        ok,
        reason_code,
        detail,
        expected_failure_class,
        FAILURE_UNVERIFIABLE,
    )


    # Run a single vector directory and compare against its expected outcome.
def run_one(
    vec_dir: Path,
    fmt: str,
    expected_outcome: str,
    reason_code: str = "",
    expected_failure_class: str = "",
) -> VectorOutcome:
    try:
        receipt = _load(vec_dir / "receipt.json")
        predecessor = _load(vec_dir / "predecessor.json")
        key_provider = _key_provider(vec_dir, fmt)
    except (strict_json.DuplicateMemberError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        # Terminal ingest failure (419): nothing is hashed, checked, or verified.
        return _parse_failure_outcome(
            vec_dir.name, expected_outcome, expected_failure_class, reason_code, exc
        )
    result = verify(receipt, ADAPTERS, key_provider=key_provider, predecessor=predecessor)

    want_verdict = _OUTCOME_TO_VERDICT.get(expected_outcome)
    ok = want_verdict is not None and result.verdict == want_verdict
    actual_failure_class = result.failure_class or ""
    if ok and expected_failure_class:
        ok = actual_failure_class == expected_failure_class
    detail = "; ".join(f"{a.axis}={a.result}({a.note})" for a in result.axes)
    return VectorOutcome(
        vec_dir.name,
        expected_outcome,
        result.verdict,
        ok,
        reason_code,
        detail,
        expected_failure_class,
        actual_failure_class,
    )


    # Run every vector named in ``corpus_root/manifest.json``.
def run_corpus(corpus_root: Path) -> list[VectorOutcome]:
    manifest = strict_json.loads((corpus_root / "manifest.json").read_text())
    out = []
    for entry in manifest:
        out.append(
            run_one(
                corpus_root / entry["dir"],
                entry["format"],
                entry["outcome"],
                entry.get("reason_code", ""),
                entry.get("failure_class", ""),
            )
        )
    return out


    # Accept the stronger verified only for the optional-dep ML-DSA skip vector.
def _tolerated(outcome: VectorOutcome) -> bool:
    return outcome.ok or (
        outcome.expected_outcome == VERDICT_UNVERIFIED
        and outcome.actual_verdict == VERDICT_VERIFIED
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


    # Run the bundled corpus and print a per-vector report; nonzero on mismatch.
def main() -> int:
    root = _default_corpus_root()
    results = run_corpus(root)
    for r in results:
        mark = "ok" if _tolerated(r) else "FAIL"
        got = r.actual_verdict
        if r.actual_failure_class:
            got += f" ({r.actual_failure_class})"
        print(f"  [{mark:>4}] {r.dir:<38} expect={r.expected_outcome:<16} got={got}")
        if not _tolerated(r):
            print(f"         {r.detail}")
    passed = sum(1 for r in results if _tolerated(r))
    print(f"\n  => {passed}/{len(results)} vectors matched expected outcome")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
