# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0

"""Differential fuzzing of canonical JSON across the three implementations.

Generates random documents from a grammar that deliberately reaches the corners
where JCS implementations drift - supplementary-plane member names, BMP
boundaries, control characters, escapes, deep nesting - and compares the bytes
produced by the Asqav cloud canonicalizer, the Python SDK, and the TypeScript
SDK. Any disagreement is a divergence, printed with the seed that produced it.

Numbers above 2**53 are excluded by default: the TypeScript number path rounds
them and Python keeps them exact, a documented open divergence pinned by the
doors parity test rather than rediscovered on every run. ``--unsafe-numbers``
puts them back for anyone working that case.

Usage:
    python verifier/differential_fuzz.py --iterations 500 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python" / "src"))

from asqav._jcs import canonical_json as sdk_canonical  # noqa: E402

    # The cloud emitter is optional: this repo does not depend on the platform.
try:
    sys.path.insert(0, str(_ROOT.parent / "asqav" / "src"))
    from asqav_cloud.core.canonical import canonical_json as cloud_canonical
except Exception:  # pragma: no cover - exercised only outside the workspace
    cloud_canonical = None

    # Member names chosen to straddle every ordering boundary that matters.
KEY_ALPHABET = [
    "a", "b", "z", "A", "Z", "_", "0", "9",
    "\u00e9",          # BMP Latin-1
    "\u4e2d",          # BMP CJK
    "\uff20",          # U+FF20, sorts above D800 by code point
    "\ue000",          # first private-use BMP char, the divergence boundary
    "\uffff",          # BMP maximum
    "\U0001f600",      # astral: surrogate pair D83D DE00
    "\U00010000",      # astral: first supplementary code point
    "\U0010ffff",      # astral: last code point
    "k\U0001f600",     # astral in a non-leading position
    "\u0000", "\u001f", "\"", "\\", "\n",
]

VALUE_LEAVES = [None, True, False, 0, 1, -1, 42, "", "x", "\U0001f600", "￿", "a\\b\"c\n"]


def _rand_value(rng: random.Random, depth: int, unsafe_numbers: bool):
    if depth <= 0 or rng.random() < 0.45:
        leaf = rng.choice(VALUE_LEAVES)
        if unsafe_numbers and rng.random() < 0.1:
            return rng.choice([2**53 + 1, 2**53 + 3, 2**63, -(2**53) - 1])
        return leaf
    if rng.random() < 0.5:
        return [_rand_value(rng, depth - 1, unsafe_numbers) for _ in range(rng.randint(0, 3))]
    n = rng.randint(0, 5)
    keys = rng.sample(KEY_ALPHABET, min(n, len(KEY_ALPHABET)))
    return {k: _rand_value(rng, depth - 1, unsafe_numbers) for k in keys}


def generate(rng: random.Random, unsafe_numbers: bool = False):
    """Return one random document rooted at an object, as receipts always are."""
    doc = _rand_value(rng, depth=4, unsafe_numbers=unsafe_numbers)
    return doc if isinstance(doc, dict) else {"root": doc}


_TS_DRIVER = """
const {canonicalize} = require(process.argv[2]);
const docs = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const out = docs.map((d) => Buffer.from(canonicalize(d)).toString('base64'));
process.stdout.write(JSON.stringify(out));
"""


def ts_canonical_batch(docs: list) -> list[bytes] | None:
    """Canonicalize every document with the TypeScript SDK, or None when it cannot run."""
    dist = _ROOT / "typescript" / "dist" / "index.js"
    if not dist.exists():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        docs_path = Path(tmp) / "docs.json"
        docs_path.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
        driver = Path(tmp) / "driver.cjs"
        driver.write_text(_TS_DRIVER, encoding="utf-8")
        proc = subprocess.run(
            ["node", str(driver), str(dist), str(docs_path)],
            capture_output=True, text=True,
        )
    if proc.returncode != 0:
        return None
    import base64
    return [base64.b64decode(x) for x in json.loads(proc.stdout)]


def engines_available() -> list[str]:
    """Name the canonicalizers this environment can actually compare."""
    names = ["sdk"]
    if cloud_canonical is not None:
        names.append("cloud")
    if ts_canonical_batch([{}]) is not None:
        names.append("typescript")
    return names


def run(iterations: int, seed: int, unsafe_numbers: bool) -> list[dict]:
    """Return one entry per divergence found; an empty list means the engines agree."""
    rng = random.Random(seed)
    docs = [generate(rng, unsafe_numbers) for _ in range(iterations)]
    divergences: list[dict] = []

    sdk = [sdk_canonical(d) for d in docs]
    cloud = [cloud_canonical(d) for d in docs] if cloud_canonical else None
    ts = ts_canonical_batch(docs)

    for i, doc in enumerate(docs):
        seen = {"sdk": sdk[i]}
        if cloud is not None:
            seen["cloud"] = cloud[i]
        if ts is not None:
            seen["typescript"] = ts[i]
        if len(set(seen.values())) > 1:
            divergences.append({
                "index": i,
                "seed": seed,
                "document": doc,
                "bytes": {k: v.decode("utf-8", "replace") for k, v in seen.items()},
            })
    return divergences


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unsafe-numbers", action="store_true")
    args = ap.parse_args()

    divergences = run(args.iterations, args.seed, args.unsafe_numbers)
    engines = engines_available()
    print(f"engines compared: {', '.join(engines)}")
    print(f"iterations: {args.iterations}  seed: {args.seed}")
    if len(engines) < 2:
        print("REFUSING: one engine is not a differential test")
        return 2

    if not divergences:
        print("no divergence")
        return 0
    print(f"DIVERGENCES: {len(divergences)}")
    for d in divergences[:5]:
        print(json.dumps(d, ensure_ascii=False, indent=2)[:1200])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
