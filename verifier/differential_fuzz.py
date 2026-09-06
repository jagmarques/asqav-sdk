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
import base64
import copy
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python" / "src"))

from asqav._jcs import canonical_json as sdk_canonical  # noqa: E402
from asqav.verifier.verify_receipt import canonical_json as standalone_canonical  # noqa: E402
from asqav.counterparty import compute_envelope_hash, verify_counterparty_binding  # noqa: E402
from asqav.verifier.verify_receipt import _envelope_hash, check_counterparty_binding  # noqa: E402

    # The cloud emitter is optional: this repo does not depend on the platform.
try:
    sys.path.insert(0, str(_ROOT.parent / "asqav" / "src"))
    from asqav_cloud.core.canonical import canonical_json as cloud_canonical
    from asqav_cloud.core.envelope import compute_envelope_hash as cloud_binding_hash
    from asqav_cloud.core.envelope import verify_counterparty_binding as cloud_binding_check
except Exception:  # pragma: no cover - exercised only outside the workspace
    cloud_canonical = None
    cloud_binding_hash = cloud_binding_check = None

REQUIRED_ENGINES = {"sdk", "standalone", "cloud", "typescript", "typescript-verifier"}
BINDING_CASES = ("anchors_absent", "anchors_present", "anchors_upgraded", "signature_base64url",
                 "signature_reencoded", "anchors_included", "scope_absent", "scope_null",
                 "scope_unknown", "origin_missing", "kid_mismatch", "binding_malformed",
                 "signature_base64url_anchors_present", "signature_base64url_anchors_upgraded",
                 "scope_absent_no_origin", "scope_null_no_origin", "scope_unknown_no_origin",
                 "distinct_issuer_matching_kid", "distinct_issuer_wrong_kid", "digest_overpadded", "digest_unpadded")

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


_TS_VERIFIER_DRIVER = """
const {asqavJcs} = require(process.argv[2]);
const docs = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const out = docs.map((d) => Buffer.from(asqavJcs(d)).toString('base64'));
process.stdout.write(JSON.stringify(out));
"""


def _ts_batch(docs: list, dist: Path, driver_src: str) -> list[bytes] | None:
    if not dist.exists():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        docs_path = Path(tmp) / "docs.json"
        docs_path.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
        driver = Path(tmp) / "driver.cjs"
        driver.write_text(driver_src, encoding="utf-8")
        proc = subprocess.run(
            ["node", str(driver), str(dist), str(docs_path)],
            capture_output=True, text=True,
        )
    if proc.returncode != 0:
        return None
    import base64
    return [base64.b64decode(x) for x in json.loads(proc.stdout)]


def ts_canonical_batch(docs: list) -> list[bytes] | None:
    """Canonicalize every document with the TypeScript SDK emitter, or None when it cannot run."""
    return _ts_batch(docs, _ROOT / "typescript" / "dist" / "index.js", _TS_DRIVER)


def ts_verifier_canonical_batch(docs: list) -> list[bytes] | None:
    """Canonicalize with the TypeScript VERIFIER's asqav dialect (the bytes it checks signatures over)."""
    return _ts_batch(docs, _ROOT / "typescript" / "dist" / "verifier" / "index.js", _TS_VERIFIER_DRIVER)


def engines_available() -> list[str]:
    """Name the canonicalizers this environment can actually compare."""
    names = ["sdk", "standalone"]
    if cloud_canonical is not None:
        names.append("cloud")
    if ts_canonical_batch([{}]) is not None:
        names.append("typescript")
    if ts_verifier_canonical_batch([{}]) is not None:
        names.append("typescript-verifier")
    return names


def run(iterations: int, seed: int, unsafe_numbers: bool) -> list[dict]:
    """Return one entry per divergence found; an empty list means the engines agree."""
    rng = random.Random(seed)
    docs = [generate(rng, unsafe_numbers) for _ in range(iterations)]
    divergences: list[dict] = []

    sdk = [sdk_canonical(d) for d in docs]
    standalone = [standalone_canonical(d) for d in docs]
    cloud = [cloud_canonical(d) for d in docs] if cloud_canonical else None
    ts = ts_canonical_batch(docs)
    ts_verifier = ts_verifier_canonical_batch(docs)

    for i, doc in enumerate(docs):
        seen = {"sdk": sdk[i], "standalone": standalone[i]}
        if cloud is not None:
            seen["cloud"] = cloud[i]
        if ts is not None:
            seen["typescript"] = ts[i]
        if ts_verifier is not None:
            seen["typescript-verifier"] = ts_verifier[i]
        if len(set(seen.values())) > 1:
            divergences.append({
                "index": i,
                "seed": seed,
                "document": doc,
                "bytes": {k: v.decode("utf-8", "replace") for k, v in seen.items()},
            })
    return divergences


def binding_cases(iterations: int, seed: int) -> list[dict]:
    """Exercise each case on each randomized payload; generation never rewrites a defect."""
    rng = random.Random(seed)
    cases = []
    for index in range(iterations):
        origin = {"payload": generate(rng), "signature": {"alg": "Ed25519", "kid": "origin", "sig": "+/8="}}
        digest = base64.b64encode(hashlib.sha256(sdk_canonical(origin)).digest()).decode()
        for name in BINDING_CASES:
            peer = copy.deepcopy(origin)
            binding = {"scope": "envelope_minus_anchors", "receipt_ref": "sig_origin", "envelope_hash": digest,
                       "expect_ack_from": "ack"}
            if "anchors_present" in name or "anchors_upgraded" in name or name == "anchors_included":
                peer["anchors"] = [{"type": "rfc3161", "value": "first"}, {"type": "opentimestamps", "value": "pending"}]
            if "anchors_upgraded" in name:
                peer["anchors"][1]["value"] = "upgraded"
                peer["export_metadata"] = {"ignored": index}
            if name == "anchors_included":
                binding["envelope_hash"] = base64.b64encode(hashlib.sha256(sdk_canonical(peer)).digest()).decode()
            elif name.startswith("signature_base64url") or name == "signature_reencoded":
                peer["signature"]["sig"] = "-_8"
                if name.startswith("signature_base64url"):
                    projection = {key: peer[key] for key in ("payload", "signature")}
                    binding["envelope_hash"] = base64.urlsafe_b64encode(hashlib.sha256(sdk_canonical(projection)).digest()).decode().rstrip("=")
            elif name.startswith("scope_absent"):
                binding.pop("scope")
            elif name.startswith("scope_null"):
                binding["scope"] = None
            elif name.startswith("scope_unknown"):
                binding["scope"] = "other"
            elif name == "origin_missing":
                peer = None
            elif name == "kid_mismatch":
                binding["expect_ack_from"] = "another-ack"
            elif name == "binding_malformed":
                binding = None
            elif name == "distinct_issuer_wrong_kid":
                binding["expect_ack_from"] = "issuer"
            elif name == "digest_overpadded":
                binding["envelope_hash"] += "="
            elif name == "digest_unpadded":
                binding["envelope_hash"] = binding["envelope_hash"].rstrip("=")
            if name.endswith("_no_origin"):
                peer = None
            expected = (True, "matches")
            if name.startswith("scope_absent"):
                expected = (None, "legacy_scope")
            elif name.startswith(("scope_null", "scope_unknown")):
                expected = (None, "unrecognised_scope")
            elif name == "origin_missing":
                expected = (None, "unresolved")
            elif name in {"signature_reencoded", "anchors_included"}:
                expected = (False, "mismatch")
            elif name in {"kid_mismatch", "distinct_issuer_wrong_kid"}:
                expected = (False, "kid_mismatch")
            elif name in {"binding_malformed", "digest_overpadded"}:
                expected = (False, "malformed")
            cases.append({"name": name, "index": index, "origin": peer, "expected": expected,
                          "acknowledging_kid": "ack", "payload": {"issuer_id": "issuer" if name.startswith("distinct_issuer") else "ack", "counterparty_binding": binding}})
    return cases


_TS_BINDING_DRIVER = r"""
const api = require(process.argv[2]);
const cases = JSON.parse(require('node:fs').readFileSync(process.argv[3], 'utf8'));
const verifier = process.argv[4] === 'verifier';
const labelOf = (state, note) => {
  for (const label of ['legacy_scope', 'unrecognised_scope', 'kid_mismatch']) if (note.startsWith(label + ':')) return label;
  if (state === 'PASS') return 'matches';
  if (state === 'SKIPPED') return 'unresolved';
  return note.startsWith('counterparty_mismatch:') ? 'mismatch' : 'malformed';
};
const out = cases.map(c => {
  let digest = null;
  if (c.origin !== null) {
    digest = verifier ? api.counterpartyEnvelopeHash(c.origin)
      : api.computeCounterpartyBinding(c.origin, {receiptRef:'sig_origin'}).envelope_hash;
  }
  let label, valid;
  const ack = {payload:c.payload, signature:{alg:'Ed25519', kid:c.acknowledging_kid, sig:'AQID'}};
  if (verifier) {
    const axis = api.ADAPTERS.find(a => a.name === 'asqav-native').extraAxesWithContext(ack, null, {originatingEnvelope:c.origin}).find(a => a[0] === 'counterparty');
    valid = axis[1] === 'PASS' ? true : axis[1] === 'FAIL' ? false : null;
    label = labelOf(axis[1], axis[2]);
  } else ({valid, label} = api.verifyCounterpartyBinding(ack, c.origin));
  return {digest, valid, label};
});
process.stdout.write(JSON.stringify(out));
"""


def _ts_bindings(cases: list[dict], verifier: bool) -> list[dict]:
    dist = _ROOT / "typescript" / "dist" / ("verifier/index.js" if verifier else "index.js")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        (path / "cases.json").write_text(json.dumps(cases))
        (path / "driver.cjs").write_text(_TS_BINDING_DRIVER)
        result = subprocess.run(["node", str(path / "driver.cjs"), str(dist), str(path / "cases.json"),
                                 "verifier" if verifier else "sdk"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"required TypeScript binding engine failed: {result.stderr[:1000]}")
    values = json.loads(result.stdout)
    if len(values) != len(cases):
        raise RuntimeError("required TypeScript engine omitted binding cases")
    return values


def _standalone_binding(case: dict) -> tuple[bool | None, str]:
    state, note = check_counterparty_binding(case["payload"], case["origin"], acknowledging_kid=case["acknowledging_kid"])
    valid = {"PASS": True, "FAIL": False, "SKIPPED": None}[state]
    for label in ("legacy_scope", "unrecognised_scope", "kid_mismatch"):
        if note.startswith(label + ":"):
            return valid, label
    if state == "PASS":
        return True, "matches"
    if state == "SKIPPED":
        return None, "unresolved"
    return False, "mismatch" if note.startswith("counterparty_mismatch:") else "malformed"


def run_bindings(iterations: int, seed: int) -> dict:
    """Require real participation from all five engines, with no lower success threshold."""
    available = set(engines_available())
    if available != REQUIRED_ENGINES or cloud_binding_check is None:
        raise RuntimeError(f"five binding engines required; available: {sorted(available)}")
    cases = binding_cases(iterations, seed)
    counts = Counter(case["name"] for case in cases)
    if set(counts) != set(BINDING_CASES) or any(counts[name] < 1 for name in BINDING_CASES):
        raise RuntimeError("every required binding case needs a nonzero count")
    ts = _ts_bindings(cases, False)
    ts_verifier = _ts_bindings(cases, True)
    divergences = []
    for index, case in enumerate(cases):
        origin = case["origin"]
        ack = {"payload": case["payload"], "signature": {"kid": case["acknowledging_kid"]}}
        sdk = verify_counterparty_binding(ack, origin)
        outcomes = {"sdk": (sdk.valid, sdk.label), "standalone": _standalone_binding(case),
                    "cloud": cloud_binding_check(case["payload"]["counterparty_binding"], origin, case["acknowledging_kid"])}
        hashes = {"sdk": compute_envelope_hash, "standalone": _envelope_hash, "cloud": cloud_binding_hash}
        seen = {engine: {"digest": hashes[engine](origin) if origin is not None else None,
                         "valid": outcome[0], "label": outcome[1]} for engine, outcome in outcomes.items()}
        seen.update({"typescript": ts[index], "typescript-verifier": ts_verifier[index]})
        if (len({json.dumps(value, sort_keys=True) for value in seen.values()}) != 1 or
                any([value["valid"], value["label"]] != list(case["expected"]) for value in seen.values())):
            divergences.append({"case": case, "engines": seen, "seed": seed})
    return {"engines": sorted(available), "counts": dict(counts), "divergences": divergences}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unsafe-numbers", action="store_true")
    args = ap.parse_args()

    try:
        bindings = run_bindings(args.iterations, args.seed)
    except RuntimeError as exc:
        print(f"REFUSING: {exc}")
        return 2
    divergences = run(args.iterations, args.seed, args.unsafe_numbers) + bindings["divergences"]
    engines = engines_available()
    print(f"engines compared: {', '.join(engines)}")
    print(f"iterations: {args.iterations}  seed: {args.seed}")
    print("binding case counts: " + json.dumps(bindings["counts"], sort_keys=True))

    if not divergences:
        print("no divergence")
        return 0
    print(f"DIVERGENCES: {len(divergences)}")
    for d in divergences[:5]:
        print(json.dumps(d, ensure_ascii=False, indent=2)[:1200])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
