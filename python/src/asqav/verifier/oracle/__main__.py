"""CLI entry point for the universal neutral verifier.

Verify one agent receipt from the command line, across every format the oracle
knows, with one self-contained call::

    python -m asqav.verifier.oracle receipt.json [--keys keys.json] [--predecessor pred.json]

It loads the receipt, runs :func:`verify` over the bundled ``ADAPTERS``, prints
the verdict and per-axis result as JSON, and sets the exit status from the
verdict: 0 on PASS, 1 on FAIL, 2 on INCOMPLETE. INCOMPLETE means an axis (the
signature, typically) could not be checked - it is never reported as a PASS.

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

from . import ADAPTERS
from .core import verify

#: verdict -> process exit status; PASS is the only success, INCOMPLETE is not a PASS.
_EXIT = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}


def _load(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        text = Path(path).read_text()
    except OSError as exc:
        print(f"asqav-verify: cannot read {path}: {exc.strerror or exc}", file=sys.stderr)
        raise SystemExit(2) from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"asqav-verify: {path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


def run(receipt_path: str, keys_path: str | None, predecessor_path: str | None) -> int:
    """Verify one receipt file and print the verdict + per-axis result as JSON."""
    receipt = _load(receipt_path)
    key_provider = _load(keys_path)
    predecessor = _load(predecessor_path)

    result = verify(receipt, ADAPTERS, key_provider=key_provider, predecessor=predecessor)

    report = {
        "format": result.fmt,
        "verdict": result.verdict,
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
