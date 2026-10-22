"""Oracle CLI; exit 0 verified, 1 invalid, 2 unverifiable. Strict ingest rejects
a duplicate member name before hashing; ``review`` never mutates the store."""
# Ships inside the standalone ``asqav-verify`` executable; stays offline and
# opens only paths given on the command line.
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from asqav import strict_json

from . import ADAPTERS, VerificationContext
from .agreement import envelope_of
from .assessment_store import AssessmentStore
from .core import FAILURE_INVALID, VERDICT_UNVERIFIED, verify
from .live_compare import classify_and_attribute
from .retain import reassess_and_classify, record_verify_result


def _attribution_out(digest: str, disagreement: object, case: object) -> int:
    # One JSON shape for every classification caller; 1 flags a split.
    print(json.dumps({
        "digest": digest,
        "disagreement": {"label": disagreement.label,
                         "differing": list(disagreement.differing),
                         "conflict": disagreement.conflict},
        "spec_case": {"label": case.label, "note": case.note, "pinned": list(case.pinned)},
    }, indent=2))
    return 1 if disagreement.label == "unexplained_needs_investigation" else 0


def _exit_code(verdict: str, failure_class: str | None) -> int:
    # verified/verified_keyed -> 0; invalid -> 1; unverifiable keeps the exit-2
    # blocked-verification state the old INCOMPLETE verdict carried.
    if verdict != VERDICT_UNVERIFIED:
        return 0
    return 1 if failure_class == FAILURE_INVALID else 2


def _load(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        text = Path(path).read_text()
    except OSError as exc:
        print(f"asqav-verify: cannot read {path}: {exc.strerror or exc}", file=sys.stderr)
        raise SystemExit(2) from None
    try:
        return strict_json.loads(text)
    except (strict_json.DuplicateMemberError, strict_json.UnsafeIntegerError) as exc:
        # Terminal ingest failure (419): never hash last-wins or rounded bytes.
        print(f"asqav-verify: {path} rejected: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except json.JSONDecodeError as exc:
        print(f"asqav-verify: {path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


def _load_raw(path: str) -> tuple[bytes, dict]:
    """Read receipt bytes plus the strict-parsed doc; failures exit 2."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        print(f"asqav-verify: cannot read {path}: {exc.strerror or exc}", file=sys.stderr)
        raise SystemExit(2) from None
    try:
        return raw, strict_json.loads(raw.decode("utf-8"))
    except (strict_json.DuplicateMemberError, strict_json.UnsafeIntegerError) as exc:
        # Terminal ingest failure (419): never hash last-wins or rounded bytes.
        print(f"asqav-verify: {path} rejected: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"asqav-verify: {path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


# --context accepts only the retained attribution fields (488 C37): a stated
# originating_envelope is never retained, so it would not reach attribution.
_CONTEXT_TEXT_FIELDS = ("evaluator", "revocation_snapshot", "policy_revision",
                        "evidence_knowledge_cutoff", "evaluation_time",
                        "event_time", "effective_time")


def _load_context(path: str | None) -> VerificationContext | None:
    """Caller-stated trust/time inputs for the live side (488 C36, C37).

    None keeps live defaults. Otherwise the file must hold a JSON
    object of retained attribution fields with JSON-native values;
    anything else exits 2 before any verification runs."""
    if path is None:
        return None
    try:
        obj = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, ValueError) as exc:
        print(f"asqav-verify: context rejected: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    if not isinstance(obj, dict):
        print("asqav-verify: context rejected: not a JSON object", file=sys.stderr)
        raise SystemExit(2)
    for key in obj:
        if key != "trust_roots" and key not in _CONTEXT_TEXT_FIELDS:
            print(f"asqav-verify: context rejected: {key} is not retained",
                  file=sys.stderr)
            raise SystemExit(2)
    roots = obj.get("trust_roots")
    if roots is not None and not (
            isinstance(roots, list) and all(isinstance(r, str) for r in roots)):
        print("asqav-verify: context rejected: trust_roots must list strings",
              file=sys.stderr)
        raise SystemExit(2)
    for key in _CONTEXT_TEXT_FIELDS:
        if obj.get(key) is not None and not isinstance(obj[key], str):
            print(f"asqav-verify: context rejected: {key} must be a string",
                  file=sys.stderr)
            raise SystemExit(2)
    when = obj.get("evaluation_time")
    if isinstance(when, str):
        try:
            datetime.fromisoformat(when.replace("Z", "+00:00"))
        except ValueError:
            print("asqav-verify: context rejected: unparseable evaluation_time",
                  file=sys.stderr)
            raise SystemExit(2) from None
    try:
        return VerificationContext(**obj)
    except TypeError as exc:
        print(f"asqav-verify: context rejected: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


def record(store_path: str, receipt_path: str, reason: str | None = None) -> int:
    """Retain one live assessment (488 C32, C2 flow caller).
    New: print digest; retained: reassess. Exit 0 reviewed, 1 split, 2 error."""
    raw, receipt = _load_raw(receipt_path)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = Path(store_path).read_text()
    except FileNotFoundError:
        store = AssessmentStore()
    except OSError as exc:
        print(f"asqav-verify: cannot read {store_path}: {exc.strerror or exc}",
              file=sys.stderr)
        return 2
    else:
        try:
            store = AssessmentStore.from_jsonl(text)
        except ValueError as exc:
            print(f"asqav-verify: store rejected: {exc}", file=sys.stderr)
            return 2
    result = verify(receipt, ADAPTERS)
    # Bytes-derived envelope ref (488 C39a): stored for later
    # conflict classification; unboundable receipts keep None.
    try:
        agreement = asdict(envelope_of(receipt, ADAPTERS))
    except ValueError:
        agreement = None
    if store.replay(digest) is None:
        try:
            record_verify_result(store, digest, result, agreement=agreement)
        except ValueError as exc:
            print(f"asqav-verify: store rejected: {exc}", file=sys.stderr)
            return 2
        try:
            Path(store_path).write_text(store.to_jsonl())
        except OSError as exc:
            print(f"asqav-verify: cannot write {store_path}: {exc.strerror or exc}",
                  file=sys.stderr)
            return 2
        print(json.dumps({"digest": digest, "recorded": len(store)}))
        return 0
    try:
        entry, disagreement, case = reassess_and_classify(
            store, digest, result, agreement=agreement, reason=reason)
    except ValueError as exc:
        print(f"asqav-verify: {exc}", file=sys.stderr)
        return 2
    try:
        Path(store_path).write_text(store.to_jsonl())
    except OSError as exc:
        print(f"asqav-verify: cannot write {store_path}: {exc.strerror or exc}",
              file=sys.stderr)
        return 2
    return _attribution_out(digest, disagreement, case)


def review(store_path: str, receipt_path: str,
           rerun_a_path: str | None = None, rerun_b_path: str | None = None,
           context_path: str | None = None) -> int:
    """Review one live receipt against its retained original (488 C28).

    The first production flow over the attribution path: the store
    supplies the original, the receipt plus reruns verify live, and
    classify_and_attribute classifies plus runs the spec-bug bar.
    Exit 0 reviewed, 1 unexplained split, 2 input/retention failure."""
    try:
        store = AssessmentStore.from_jsonl(Path(store_path).read_text())
    except OSError as exc:
        print(f"asqav-verify: cannot read {store_path}: {exc.strerror or exc}",
              file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"asqav-verify: store rejected: {exc}", file=sys.stderr)
        return 2
    live_context = None
    if context_path is not None:
        try:
            live_context = _load_context(context_path)
        except SystemExit:
            return 2
    raw, receipt = _load_raw(receipt_path)
    digest = hashlib.sha256(raw).hexdigest()
    reruns = []
    for tag, rerun_path in (("rerun_a", rerun_a_path), ("rerun_b", rerun_b_path)):
        if rerun_path is None:
            reruns.append(None)
            continue
        rerun_raw, rerun_doc = _load_raw(rerun_path)
        if hashlib.sha256(rerun_raw).hexdigest() != digest:
            print(f"asqav-verify: {tag} is not a re-execution of the reviewed bytes",
                  file=sys.stderr)
            return 2
        # Reruns re-execute the same live evaluation, so they inherit
        # the caller-stated context (C36); defaults when unstated.
        reruns.append(verify(rerun_doc, ADAPTERS, context=live_context))
    result = verify(receipt, ADAPTERS, context=live_context)
    # Live envelope re-derived from bytes; the stored ref is the other
    # comparandum, never this side. Unboundable receipts have none.
    try:
        live_agreement = asdict(envelope_of(receipt, ADAPTERS))
    except ValueError:
        live_agreement = None
    try:
        disagreement, case = classify_and_attribute(store, digest, result, *reruns,
                                                    agreement=live_agreement)
    except ValueError as exc:
        print(f"asqav-verify: {exc}", file=sys.stderr)
        return 2
    return _attribution_out(digest, disagreement, case)


    # Verify one receipt file and print the verdict + per-axis result as JSON.
def run(receipt_path: str, keys_path: str | None, predecessor_path: str | None) -> int:
    receipt = _load(receipt_path)
    key_provider = _load(keys_path)
    predecessor = _load(predecessor_path)

    result = verify(receipt, ADAPTERS, key_provider=key_provider, predecessor=predecessor)

    report = {
        "format": result.fmt,
        "verdict": result.verdict,
        "failure_class": result.failure_class,
        "axes": [asdict(a) for a in result.axes],
    }
    print(json.dumps(report, indent=2))
    return _exit_code(result.verdict, result.failure_class)


def _verify_main(args: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="asqav-verify",
        description="Universal neutral verifier for agent receipts (signature, chain, structure).",
    )
    p.add_argument("receipt", help="path to the receipt JSON to verify")
    p.add_argument(
        "--keys",
        help="path to the key provider (JWKS for asqav-native, {key_id:hex} for aerf, "
        "{key_id:pem} for acta, did_map for agentreceipts)",
    )
    p.add_argument("--predecessor", help="path to the predecessor receipt JSON for the chain check")
    parsed = p.parse_args(args)
    return run(parsed.receipt, parsed.keys, parsed.predecessor)


def _review_main(args: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="asqav-verify review",
        description="Review a live receipt against its retained original.",
    )
    p.add_argument("store", help="path to the retained assessment JSONL log")
    p.add_argument("receipt", help="path to the live receipt JSON to review")
    p.add_argument("--rerun-a", help="path to a live re-execution of the receipt bytes")
    p.add_argument("--rerun-b", help="path to a second live re-execution")
    p.add_argument("--context", help="path to a JSON object of live-side "
                   "trust/time inputs (trust_roots, evaluation_time, ...)")
    parsed = p.parse_args(args)
    return review(parsed.store, parsed.receipt, parsed.rerun_a, parsed.rerun_b,
                  parsed.context)


def _record_main(args: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="asqav-verify record",
        description="Retain a live assessment into the store log.",
    )
    p.add_argument("store", help="path to the retained assessment JSONL log")
    p.add_argument("receipt", help="path to the receipt JSON to retain")
    p.add_argument("--reason", default=None,
                   help="label recorded on a reassessment (e.g. key-compromised)")
    parsed = p.parse_args(args)
    return record(parsed.store, parsed.receipt, reason=parsed.reason)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["review"]:
        return _review_main(args[1:])
    if args[:1] == ["record"]:
        return _record_main(args[1:])
    return _verify_main(args)


if __name__ == "__main__":
    sys.exit(main())
