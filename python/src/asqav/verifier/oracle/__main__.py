"""CLI entry point for the universal neutral verifier.

Verify one agent receipt from the command line, across every format the oracle
knows, with one self-contained call::

    python -m asqav.verifier.oracle receipt.json [--keys keys.json] [--predecessor pred.json]

It loads the receipt (strict ingest: a duplicated JSON member name at any depth
is a terminal parse failure, rejected before any hashing - criterion 419), runs
:func:`verify` over the bundled ``ADAPTERS``, prints the verdict and per-axis
result as JSON, and sets the exit status from the verdict vocabulary
(criteria 418/438):

  - 0  verified / verified_keyed
  - 1  unverified, failure_class=invalid (a binding was proven broken)
  - 2  unverified, failure_class=unverifiable (verification could not complete)

Exit 2 is NOT a verified outcome: it means a check (the signature, typically)
could not be run, and the verifier never reports a broken receipt as verified.

``--keys`` is the format-shaped key provider: a JWKS dict for Asqav-native, a
``{key_id: hex}`` map for AERF, an ``{key_id: pem}`` map for ACTA, or a did_map
for agentreceipts. did:key receipts self-resolve and need no ``--keys``.

A second command records a live assessment into the retained log::

    python -m asqav.verifier.oracle record store.jsonl receipt.json

It verifies the receipt, appends the assessment (a missing store file
starts a new log; a broken one fails), and prints the digest. Bytes
already retained reassess instead: the reassessment is persisted and
classified against the retained original. A third
command reviews a live reassessment against its retained original::

    python -m asqav.verifier.oracle review store.jsonl receipt.json [--rerun-a a.json --rerun-b b.json]

It loads the retained log (a broken chain is a terminal failure), verifies the
receipt plus reruns (a rerun that is not a re-execution of the reviewed bytes
is a terminal failure), and runs the full attribution pipeline
(:func:`classify_and_attribute <asqav.verifier.oracle.live_compare.classify_and_attribute>`).
It prints the disagreement and spec-case outcome as JSON and exits 0 when
reviewed, 1 on an unexplained split (it needs investigation), 2 on an
input or retention failure. ``review`` never mutates the store file.
With ``--context ctx.json`` the live side (receipt plus reruns) verifies
under those caller-stated trust/time inputs, so a trust-split between
the retained era and now classifies instead of hiding behind defaults.

This is the entry point bundled into the single-file ``asqav-verify`` binary, so
it stays free of network calls and reads only the files named on the command
line.
"""
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
        "disagreement": {"label": disagreement.label, "differing": list(disagreement.differing)},
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
        # Terminal ingest failure (419): never hash or verify last-wins or rounded bytes.
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
        # Terminal ingest failure (419): never hash or verify last-wins or rounded bytes.
        print(f"asqav-verify: {path} rejected: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"asqav-verify: {path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


# --context accepts only the retained attribution fields (488 C37):
# originating_envelope drives verification but is never retained, so a
# stated envelope would be a live input invisible to attribution.
_CONTEXT_TEXT_FIELDS = ("evaluator", "revocation_snapshot", "policy_revision",
                        "evidence_knowledge_cutoff", "evaluation_time")


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


def record(store_path: str, receipt_path: str) -> int:
    """Retain one live assessment (488 C32, C2 flow caller).

    New bytes append an original and print the digest (exit 0). Bytes
    already retained reassess instead: the reassessment is persisted,
    then classified against the retained original through
    reassess_and_classify (exit 0 reviewed, 1 unexplained split).
    Store or input failures exit 2."""
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
            store, digest, result, agreement=agreement)
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
    try:
        disagreement, case = classify_and_attribute(store, digest, result, *reruns)
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
    parsed = p.parse_args(args)
    return record(parsed.store, parsed.receipt)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["review"]:
        return _review_main(args[1:])
    if args[:1] == ["record"]:
        return _record_main(args[1:])
    return _verify_main(args)


if __name__ == "__main__":
    sys.exit(main())
