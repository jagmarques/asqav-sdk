"""CLI entry point for the universal neutral verifier.

Verify one agent receipt from the command line, across every format the oracle
knows, with one self-contained call::

    python -m asqav.verifier.oracle receipt.json [--keys keys.json] [--predecessor pred.json]

It loads the receipt, runs :func:`verify` over the bundled ``ADAPTERS``, prints
the verdict and per-axis result as JSON, and sets the exit status from the
verdict: 0 on PASS, 1 on INVALID (a binding check ran and failed), 2 on
UNVERIFIABLE (recomputation could not complete). UNVERIFIABLE is never a PASS,
and a duplicate JSON member name at any depth is a terminal parse failure that
exits 2 before any hashing or signature check (criterion 419).

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

from asqav.strict_json import DuplicateJsonMemberError, strict_loads

from . import ADAPTERS
from .core import verify

#: verdict -> process exit status; PASS is the only success, UNVERIFIABLE is not a PASS.
_EXIT = {"PASS": 0, "INVALID": 1, "UNVERIFIABLE": 2}


def _load(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        text = Path(path).read_text()
    except OSError as exc:
        print(f"asqav-verify: cannot read {path}: {exc.strerror or exc}", file=sys.stderr)
        raise SystemExit(2) from None
    try:
        # Strict ingest (criterion 419): a duplicate member name is terminal and
        # exits before any hashing, canonicalisation, or signature check
        return strict_loads(text)
    except DuplicateJsonMemberError as exc:
        print(f"asqav-verify: {path}: strict ingest refused it: {exc}", file=sys.stderr)
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
        "classification": result.classification,
        "axes": [asdict(a) for a in result.axes],
    }
    print(json.dumps(report, indent=2))
    return _EXIT.get(result.verdict, 2)


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
