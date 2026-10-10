# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Deterministically generate verifier/float-repr-cases.json (criterion 747).

Each row is a JSON float lexeme with what Python writes for it: repr(float), the canonical bytes
of {"v": lexeme} and the counterparty envelope hash of an originator carrying it. The lexemes
come from a seeded generator that uses only integer draws, so the table is the same on every
platform. --check compares the file to a fresh run and never writes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))

from asqav.verifier.verify_receipt import _envelope_hash, canonical_json  # noqa: E402

OUT = ROOT / "verifier" / "float-repr-cases.json"
SEED = 747
PER_ZONE = 300
SIGNATURE = {"alg": "ML-DSA-65", "kid": "k1", "sig": "AAAA"}

#: Hand-picked edges: the repr layout switches at 1e-4 and 1e16, the shortest
#: digit string at the subnormal and maximum ends, and zero keeps its sign.
EDGES = (
    "0.0001", "1e-4", "9.999999999999999e-05", "0.00009999999999999999", "1e-05", "0.00001",
    "1e-9", "1e-10", "5e-324", "2.2250738585072014e-308", "1.7976931348623157e308",
    "1e16", "10000000000000000.0", "9999999999999998.0", "1e15", "999999999999999.9",
    "1e20", "1e21", "1e22", "9.999999999999999e22", "123456789012345680.0",
    "76355072847686510.0", "7.635507284768651e+16", "0.1", "0.5", "1.0", "100.0", "500.0",
    "4.35", "1e-07", "1e-7", "0.0000001", "-1.5e-7", "-1e16", "-0.0001",
    "-0.0", "0.0", "-0e0", "0e5", "-0.00", "-0.0e-5", "0.0E+3",
)


def _digits(rng: random.Random, n: int) -> str:
    return str(rng.randrange(1, 10)) + "".join(str(rng.randrange(10)) for _ in range(n - 1))


def _sign(rng: random.Random) -> str:
    return "-" if rng.randrange(2) else ""


def _small(rng: random.Random) -> str:
    """1e-9 <= |x| < 1e-4, spelled as an exponent or as positional decimals."""
    ds = _digits(rng, rng.randrange(1, 18))
    e = rng.randrange(5, 10)
    form = rng.randrange(4)
    if form == 0:
        return f"{_sign(rng)}{ds[0]}.{ds[1:] or '0'}e-{e}"
    if form == 1:
        return f"{_sign(rng)}{ds[0]}.{ds[1:] or '0'}E-{e:02d}"
    if form == 2:
        return f"{_sign(rng)}0.{'0' * (e - 1)}{ds}"
    return f"{_sign(rng)}{ds}e-{e + len(ds) - 1}"


def _large(rng: random.Random) -> str:
    """1e16 <= |x| < 1e21, spelled as an exponent or as a long integer-valued decimal."""
    form = rng.randrange(3)
    if form == 0:
        ds = _digits(rng, rng.randrange(1, 18))
        return f"{_sign(rng)}{ds[0]}.{ds[1:] or '0'}e{rng.randrange(16, 21)}"
    if form == 1:
        ds = _digits(rng, rng.randrange(1, 18))
        return f"{_sign(rng)}{ds[0]}.{ds[1:] or '0'}E+{rng.randrange(16, 21)}"
    return f"{_sign(rng)}{_digits(rng, rng.randrange(17, 22))}.0"


def _mid(rng: random.Random) -> str:
    """Decimals across 1e-6 .. 1e18, covering both switch points from either side."""
    ds = _digits(rng, rng.randrange(1, 18))
    e = rng.randrange(-6, 19)
    return f"{_sign(rng)}{ds[0]}.{ds[1:] or '0'}e{e}"


def _bits(rng: random.Random) -> str:
    """A uniformly random finite double, spelled as its own repr."""
    while True:
        (x,) = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))
        if math.isfinite(x):
            return repr(x)


def _row(zone: str, lexeme: str) -> dict:
    parsed = json.loads('{"v":' + lexeme + "}")
    text = '{"payload":{"v":' + lexeme + '},"signature":' + json.dumps(SIGNATURE) + "}"
    return {
        "zone": zone,
        "lexeme": lexeme,
        "repr": repr(parsed["v"]),
        "canonical": canonical_json(parsed).decode(),
        "envelope_hash": _envelope_hash(json.loads(text)),
    }


def build() -> dict:
    rng = random.Random(SEED)
    rows = [_row("edge", lx) for lx in EDGES]
    for zone, make in (("small", _small), ("large", _large), ("mid", _mid), ("bits", _bits)):
        rows += [_row(zone, make(rng)) for _ in range(PER_ZONE)]
    return {
        "note": (
            "Expected values are Python's: repr(float), verify_receipt.canonical_json of "
            '{"v": lexeme}, and _envelope_hash of {"payload": {"v": lexeme}, "signature": '
            "signature}. python/tests/test_float_repr_parity.py recomputes them and "
            "typescript/tests/verifier-float-repr.test.ts feeds the same lexemes through "
            "the float-preserving parser. Regenerate with verifier/generate_float_repr_cases.py. "
            "Zones: small 1e-9 <= |x| < 1e-4, large 1e16 <= |x| < 1e21, mid 1e-6..1e18, "
            "bits uniform doubles, edge hand-picked."
        ),
        "seed": SEED,
        "signature": SIGNATURE,
        "cases": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare, never write")
    text = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if ap.parse_args().check:
        if OUT.read_text() != text:
            print(f"{OUT} differs from a fresh run", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
