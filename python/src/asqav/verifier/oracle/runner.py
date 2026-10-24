"""Conformance-vector runner - drive the oracle over the vector corpus.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from asqav import strict_json

from ..verify_receipt import evaluate_anchors
from . import ADAPTERS, crypto
from .core import (
    FAILURE_UNVERIFIABLE,
    VERDICT_UNVERIFIED,
    VERDICT_VERIFIED,
    AxisResult,
    verify,
)
from .semantic_classify import classify

#: manifest outcome tokens -> the VerifyResult.verdict they must produce.
_OUTCOME_TO_VERDICT = {
    "verified": "verified",
    "verified_keyed": "verified_keyed",
    "unverified": "unverified",
}


#: Anchors axis status for a receipt that carries no anchors.
NOT_APPLICABLE = "not_applicable"
ANCHORS_AXIS = "anchors"
#: Trusted anchor material evaluate_anchors takes, apart from the receipt.
TRUSTED_MATERIAL_KEYS = ("trusted_tsa_keys", "bitcoin_headers", "trust_bundle")


def anchors_axis(receipt: object, trusted_material: dict | None = None) -> AxisResult:
    """Anchors axis beside the verdict, never folded into it.
    Anchors without trusted material SKIP; only a real proof PASSes."""
    anchors = receipt.get("anchors") if isinstance(receipt, dict) else None
    if anchors is None or anchors == []:
        return AxisResult(ANCHORS_AXIS, NOT_APPLICABLE, "receipt carries no anchors")
    supplied = {k: v for k, v in (trusted_material or {}).items() if v is not None}
    if isinstance(anchors, list) and not supplied:
        return AxisResult(
            ANCHORS_AXIS, crypto.SKIPPED,
            "receipt carries anchors; no trusted anchor material supplied apart from it",
            FAILURE_UNVERIFIABLE,
        )
    ev = evaluate_anchors(receipt, **{k: supplied[k] for k in TRUSTED_MATERIAL_KEYS if k in supplied})
    return AxisResult(ANCHORS_AXIS, ev.result, ev.note, classify(ANCHORS_AXIS, ev.result, ev.note))


@dataclass(frozen=True)
class VectorOutcome:
    """The result of running one corpus vector against the oracle.
    """

    dir: str
    expected_outcome: str
    actual_verdict: str
    ok: bool
    reason_code: str
    detail: str
    expected_failure_class: str = ""
    actual_failure_class: str = ""
    anchors: AxisResult | None = None


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
        originating_envelope = _load(vec_dir / "originating_envelope.json")
        key_provider = _key_provider(vec_dir, fmt)
    except (strict_json.DuplicateMemberError, strict_json.UnsafeIntegerError,
            json.JSONDecodeError, UnicodeDecodeError) as exc:
        # Terminal ingest failure, e.g. an integer past 2**53: never verified.
        return _parse_failure_outcome(
            vec_dir.name, expected_outcome, expected_failure_class, reason_code, exc
        )
    from .adapter import VerificationContext

    result = verify(receipt, ADAPTERS, key_provider=key_provider, predecessor=predecessor,
                    context=VerificationContext(originating_envelope))

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
        anchors_axis(receipt),
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
    """Resolve the conformance-vector corpus at the repo root.
    It is ``<repo>/verifier/conformance-vectors`` so both consumers agree."""
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
