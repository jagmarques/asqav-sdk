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

This is the entry point bundled into the single-file ``asqav-verify`` binary, so
it stays free of network calls and reads only the files named on the command
line.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from asqav import strict_json

from . import ADAPTERS
from .core import FAILURE_INVALID, VERDICT_UNVERIFIED, verify


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
    except strict_json.DuplicateMemberError as exc:
        # Terminal ingest failure (419): never hash or verify last-wins bytes.
        print(f"asqav-verify: {path} rejected: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except json.JSONDecodeError as exc:
        print(f"asqav-verify: {path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


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


def main(argv: list[str] | None = None) -> int:
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
    args = p.parse_args(argv)
    return run(args.receipt, args.keys, args.predecessor)


if __name__ == "__main__":
    sys.exit(main())
