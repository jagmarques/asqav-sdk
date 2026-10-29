"""Conformance-vector runner - drive the oracle over the vector corpus.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from asqav import strict_json

from ..oversight_attestation import NOT_APPLICABLE, oversight_attestation
from ..verify_receipt import evaluate_anchors
from . import ADAPTERS, crypto
from .core import (
    FAILURE_UNVERIFIABLE,
    VERDICT_UNVERIFIED,
    VERDICT_VERIFIED,
    AxisResult,
    verify,
)
from .semantic_classify import classify as classify_failure

#: manifest outcome tokens -> the VerifyResult.verdict they must produce.
_OUTCOME_TO_VERDICT = {
    "verified": "verified",
    "verified_keyed": "verified_keyed",
    "unverified": "unverified",
}


#: Anchors axis status for a receipt that carries no anchors.
ANCHORS_AXIS = "anchors"
#: Trusted anchor material evaluate_anchors takes, apart from the receipt.
TRUSTED_MATERIAL_KEYS = ("trusted_tsa_keys", "bitcoin_headers", "trust_bundle")

#: The expected.json member naming the requirements a vector's verdict covers.
AXIS_SCOPE_KEY = "axis_scope"
AXIS_SCOPE_EXERCISED = "exercised"
AXIS_SCOPE_NOT_APPLICABLE = NOT_APPLICABLE
#: The requirement the anchors axis is evidence for.
ANCHOR_REQUIREMENT = "REQ-ANCHOR"


def _anchors_skip_note(axis_scope: object) -> str:
    """Why anchors SKIP; names a declaration that calls them not applicable."""
    waived = axis_scope.get(AXIS_SCOPE_NOT_APPLICABLE) if isinstance(axis_scope, dict) else None
    if isinstance(waived, list) and ANCHOR_REQUIREMENT in waived:
        return (
            f"receipt carries anchors, yet expected.json {AXIS_SCOPE_KEY} declares "
            f"{ANCHOR_REQUIREMENT} {AXIS_SCOPE_NOT_APPLICABLE}; no trusted anchor "
            "material supplied, so not checked"
        )
    return "receipt carries anchors; no trusted anchor material supplied apart from it"


def anchors_axis(
    receipt: object, trusted_material: dict | None = None, axis_scope: object = None
) -> AxisResult:
    """Anchors axis beside the verdict, never folded into it; only a real proof
    PASSes. No trusted material: SKIP. A declaration never waives material."""
    anchors = receipt.get("anchors") if isinstance(receipt, dict) else None
    if anchors is None or anchors == []:
        return AxisResult(ANCHORS_AXIS, NOT_APPLICABLE, "receipt carries no anchors")
    supplied = {k: v for k, v in (trusted_material or {}).items() if v is not None}
    if isinstance(anchors, list) and not supplied:
        return AxisResult(
            ANCHORS_AXIS, crypto.SKIPPED, _anchors_skip_note(axis_scope), FAILURE_UNVERIFIABLE,
        )
    ev = evaluate_anchors(receipt, **{k: supplied[k] for k in TRUSTED_MATERIAL_KEYS if k in supplied})
    return AxisResult(ANCHORS_AXIS, ev.result, ev.note, classify_failure(ANCHORS_AXIS, ev.result, ev.note))


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
    oversight_attestation: str = ""
    expected_oversight_attestation: str = ""


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
        # did:key receipts self-resolve; did:agent/web carry an injected map.
        return _load(vec_dir / "did_map.json")
    if fmt == "w3c-vc":
        # did:web resolves from an injected DID-document map; never fetched.
        return _load(vec_dir / "did_map.json")
    if fmt == "pipelock-evidence-v2":
        # keys.json: {signer_key_id: hex_32_bytes} - the raw Ed25519 key.
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


def load_inputs(vec_dir: Path, fmt: str) -> tuple:
    """A vector's oracle inputs: receipt, predecessor, envelope, keys."""
    return (
        _load(vec_dir / "receipt.json"),
        _load(vec_dir / "predecessor.json"),
        _load(vec_dir / "originating_envelope.json"),
        _key_provider(vec_dir, fmt),
    )


def verify_inputs(receipt, predecessor, originating_envelope, key_provider):
    """Run the oracle over the inputs load_inputs returns."""
    from .adapter import VerificationContext

    return verify(receipt, ADAPTERS, key_provider=key_provider, predecessor=predecessor,
                  context=VerificationContext(originating_envelope))


    # Run a single vector directory and compare against its expected outcome.
def run_one(
    vec_dir: Path,
    fmt: str,
    expected_outcome: str,
    reason_code: str = "",
    expected_failure_class: str = "",
) -> VectorOutcome:
    try:
        inputs = load_inputs(vec_dir, fmt)
    except (strict_json.DuplicateMemberError, strict_json.UnsafeIntegerError,
            json.JSONDecodeError, UnicodeDecodeError) as exc:
        # Terminal ingest failure, e.g. an integer past 2**53: never verified.
        return _parse_failure_outcome(
            vec_dir.name, expected_outcome, expected_failure_class, reason_code, exc
        )
    receipt = inputs[0]
    # The corpus declaration, not an input: unparseable JSON raises here.
    declared = _load(vec_dir / "expected.json")
    axis_scope = declared.get(AXIS_SCOPE_KEY) if isinstance(declared, dict) else None
    result = verify_inputs(*inputs)

    want_verdict = _OUTCOME_TO_VERDICT.get(expected_outcome)
    ok = want_verdict is not None and result.verdict == want_verdict
    actual_failure_class = result.failure_class or ""
    if ok and expected_failure_class:
        ok = actual_failure_class == expected_failure_class
    report = oversight_attestation(
        receipt.get("payload") if isinstance(receipt, dict) else None,
        inputs[3],
    )
    expected_report = declared.get("oversight_attestation") if isinstance(declared, dict) else None
    if expected_report is not None:
        # A declared state that is not a string never matches
        ok = ok and isinstance(expected_report, str) and report == expected_report
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
        anchors_axis(receipt, axis_scope=axis_scope),
        report,
        expected_report if isinstance(expected_report, str) else "",
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
        if r.expected_oversight_attestation:
            got += f" oversight_attestation={r.oversight_attestation}"
        print(f"  [{mark:>4}] {r.dir:<38} expect={r.expected_outcome:<16} got={got}")
        if not _tolerated(r):
            print(f"         {r.detail}")
    passed = sum(1 for r in results if _tolerated(r))
    print(f"\n  => {passed}/{len(results)} vectors matched expected outcome")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
