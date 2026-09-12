# Copyright 2026 Asqav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Recompute every re-derivable member of corpus B from each vector's own input.

Corpus B is ``verifier/conformance-vectors``. This is the corpus-B half of
what ``regen_fingerprint_vectors.py`` does for corpus A (criterion 693;
criterion 672's audit is the method this mechanizes).

Derived members this script owns, per vector directory:

* ``previousReceiptHash`` — from ``predecessor.json`` under the vector's OWN
  chain rule: asqav-native chains ``sha256(JCS(predecessor.payload))``, acta
  chains ``sha256(JCS({payload, signature}))`` of the full signed
  predecessor. One rule for all reports a false positive on
  ``acta-02-chain-link``; the 672 audit proved that. A published value may
  carry a ``sha256:`` prefix (the ACTA -03 form); the hex is derived, the
  form is authored and is preserved. No ``predecessor.json`` means genesis:
  the hash must be all zeros.
* ``payload_digest`` — ``{hash, size}`` over the JCS of the context carried
  in the same receipt. No carried context means unrecomputable, not drift.
* ``counterparty_binding.envelope_hash`` — standard-base64
  ``sha256(JCS({payload, signature}))`` of ``originating_envelope.json``.
  A binding that names no originating envelope file is a failure: a digest
  with no stated origin is the exact shape that let corpus A go stale.

Everything else is authored (or pinned, not recomputed): ``sig`` needs the
private keys, and ``action_ref``/``policy_digest``/``key_thumbprint`` have no
preimage in the corpus. Those are pinned literally by
``python/tests/test_corpus_b_integrity.py``.

Tamper vectors pin intentional breakage, and this job knows it
mechanically: ``expected.json``'s ``reason_code`` maps to the member the
vector exists to break (``chain`` -> previousReceiptHash,
``payload_digest_mismatch`` -> payload_digest, ``counterparty_mismatch`` ->
envelope_hash). A skip that re-derives cleanly is a STALE SKIP and fails:
the list must stay honest. Reasons that name no recomputed member
(``issuer_signature``, ``counterparty_legacy_scope`` — whose digest is
correct and only the scope is wrong — and the rest) skip nothing.

Run ``python verifier/regen_conformance_vectors.py`` after an intentional
corpus edit, then ``python verifier/freeze_corpus_lock.py`` to move the pins
— the re-freeze is the LAST step. ``--check`` exits nonzero on any drift
without writing, which is what CI runs. Write mode swaps published value
bytes surgically and refuses any value that does not occur exactly once, so
it can neither reformat the twelve nonstandard receipts nor collapse
``asqav-11``'s deliberate duplicate payload.
"""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
ROOT = _HERE / "conformance-vectors"

GENESIS_ZEROS = "0" * 64

#: reason_code -> the recomputed member that vector exists to break.
SKIP_BY_REASON = {
    "chain": "previousReceiptHash",
    "payload_digest_mismatch": "payload_digest",
    "counterparty_mismatch": "envelope_hash",
}


def canonical_json(obj: Any) -> bytes:
    """RFC 8785 canonical bytes; keys sort by UTF-16 code unit.

    Vendored like the fingerprint regen's copy on purpose: this job must not
    import the SDK code whose outputs it checks.
    """
    return _serialize(obj).encode("utf-8")


def _serialize(obj: Any) -> str:
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: kv[0].encode("utf-16-be"))
        return "{" + ",".join(f"{_serialize(k)}:{_serialize(v)}" for k, v in items) + "}"
    if isinstance(obj, list):
        return "[" + ",".join(_serialize(v) for v in obj) + "]"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if obj is None:
        return "null"
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError("NaN and Infinity are not JSON")
        return repr(obj)
    raise TypeError(f"not canonicalizable: {type(obj).__name__}")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Finding:
    def __init__(self, vector: str, member: str, published: Any, recomputed: Any):
        self.vector = vector
        self.member = member
        self.published = published
        self.recomputed = recomputed

    def __str__(self) -> str:
        return (
            f"{self.vector}: {self.member}\n"
            f"    was {self.published!r}\n"
            f"    now {self.recomputed!r}"
        )


def _chain_hex(vector: str, fmt: str, predecessor: dict) -> str:
    if fmt == "acta":
        obj = {"payload": predecessor["payload"], "signature": predecessor["signature"]}
    elif fmt == "asqav-native":
        obj = predecessor["payload"]
    else:
        raise ValueError(f"{vector}: chain rule unknown for format {fmt!r}")
    return _sha256_hex(canonical_json(obj))


def _audit_vector(vector_dir: Path) -> tuple[list[Finding], list[str], list[str], int]:
    """Returns (drifts, applied_skips, unrecomputable_notes, rederived) for one vector."""
    name = vector_dir.name
    receipt = json.loads((vector_dir / "receipt.json").read_text())
    expected = json.loads((vector_dir / "expected.json").read_text())
    reason = expected.get("reason_code") or ""
    fmt = expected.get("format")
    drifts: list[Finding] = []
    skips: list[str] = []
    unrecomputable: list[str] = []
    rederived = 0

    def skipped(member: str) -> bool:
        return SKIP_BY_REASON.get(reason) == member

    def note(member: str, published: Any, recomputed: Any) -> None:
        nonlocal rederived
        if published == recomputed:
            if skipped(member):
                drifts.append(Finding(name, member + " (stale skip)",
                                      published, recomputed))
            else:
                rederived += 1
            return
        if skipped(member):
            skips.append(f"{name}: {member} (reason {reason})")
            return
        drifts.append(Finding(name, member, published, recomputed))

    payload = receipt.get("payload")
    if isinstance(payload, dict) and payload.get("previousReceiptHash") is not None:
        published = payload["previousReceiptHash"]
        pred_path = vector_dir / "predecessor.json"
        if pred_path.exists():
            predecessor = json.loads(pred_path.read_text())
            recomputed_hex = _chain_hex(name, fmt, predecessor)
            published_hex = published[7:] if published.startswith("sha256:") else published
            if published_hex == recomputed_hex:
                if skipped("previousReceiptHash"):
                    drifts.append(Finding(name, "previousReceiptHash (stale skip)",
                                          published, published))
                else:
                    rederived += 1
            else:
                new = ("sha256:" if published.startswith("sha256:") else "") + recomputed_hex
                note("previousReceiptHash", published, new)
        elif published == GENESIS_ZEROS:
            if skipped("previousReceiptHash"):
                drifts.append(Finding(name, "previousReceiptHash (stale skip)",
                                      published, published))
            else:
                rederived += 1
        elif skipped("previousReceiptHash"):
            skips.append(f"{name}: previousReceiptHash (reason {reason})")
        else:
            drifts.append(Finding(
                name, "previousReceiptHash (non-genesis hash, no predecessor.json)",
                published, "<unrecomputable>"))

    if isinstance(payload, dict) and isinstance(payload.get("payload_digest"), dict):
        published = payload["payload_digest"]
        context = payload.get("context")
        if context is None:
            unrecomputable.append(f"{name}: payload_digest (no context carried)")
        else:
            encoded = canonical_json(context)
            recomputed = {"hash": _sha256_hex(encoded), "size": len(encoded)}
            note("payload_digest", published, recomputed)

    binding = payload.get("counterparty_binding") if isinstance(payload, dict) else None
    if isinstance(binding, dict) and "envelope_hash" in binding:
        published = binding["envelope_hash"]
        origin_path = vector_dir / "originating_envelope.json"
        if not origin_path.exists():
            drifts.append(Finding(name, "envelope_hash (no originating_envelope.json)",
                                  published, "<unrecomputable>"))
        else:
            origin = json.loads(origin_path.read_text())
            scoped = {"payload": origin["payload"], "signature": origin["signature"]}
            recomputed = base64.b64encode(
                hashlib.sha256(canonical_json(scoped)).digest()).decode()
            note("envelope_hash", published, recomputed)

    return drifts, skips, unrecomputable, rederived


def _vector_dirs() -> list[Path]:
    return sorted(
        (Path(p).parent for p in glob.glob(str(ROOT / "*" / "receipt.json"))),
        key=lambda d: d.name,
    )


def _published_text(vector: str, member: str, published: Any) -> str:
    """The exact bytes write mode looks for: the JSON rendering of a value."""
    if member == "payload_digest":
        raw = json.dumps(published, separators=(",", ":"))
        spaced = json.dumps(published, separators=(", ", ": "))
        text = (ROOT / vector / "receipt.json").read_text()
        if text.count(spaced) == 1:
            return spaced
        return raw
    return json.dumps(published, ensure_ascii=False)


def _apply_surgical(vector: str, member: str, published: Any, recomputed: Any) -> str:
    """Swap one published value's bytes; refuse anything but one occurrence."""
    path = ROOT / vector / "receipt.json"
    text = path.read_text()
    if member == "payload_digest":
        old = _published_text(vector, member, published)
        new_candidates = [
            json.dumps(recomputed, separators=(",", ":")),
            json.dumps(recomputed, separators=(", ", ": ")),
        ]
        new = new_candidates[1] if ", " in old else new_candidates[0]
    else:
        old = _published_text(vector, member, published)
        new = json.dumps(recomputed, ensure_ascii=False)
    hits = text.count(old)
    if hits != 1:
        raise ValueError(
            f"{vector}: {member} occurs {hits} times; refusing a blind rewrite "
            "(fix the receipt by hand, then re-freeze)"
        )
    path.write_text(text.replace(old, new, 1))
    return f"{vector}: {member}\n    was {published!r}\n    now {recomputed!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if any derived member differs from the published file; write nothing",
    )
    args = parser.parse_args()

    all_drifts: list[Finding] = []
    all_skips: list[str] = []
    all_unrecomputable: list[str] = []
    rederived = 0
    for vector_dir in _vector_dirs():
        drifts, skips, unrecomputable, clean = _audit_vector(vector_dir)
        all_drifts.extend(drifts)
        all_skips.extend(skips)
        all_unrecomputable.extend(unrecomputable)
        rederived += clean

    if args.check:
        if all_drifts:
            print("corpus-B integrity: DRIFT between the published files and their own inputs",
                  file=sys.stderr)
            for finding in all_drifts:
                print(f"  {finding}", file=sys.stderr)
            print("\nregenerate with: python verifier/regen_conformance_vectors.py"
                  "\nthen re-freeze:  python verifier/freeze_corpus_lock.py",
                  file=sys.stderr)
            return 1
        print(f"corpus-B integrity: {rederived} members re-derive, "
              f"{len(all_skips)} reason-coded skips, "
              f"{len(all_unrecomputable)} unrecomputable without their input")
        for line in all_skips:
            print(f"  skip: {line}")
        for line in all_unrecomputable:
            print(f"  unrecomputable: {line}")
        return 0

    applied = []
    for finding in all_drifts:
        if finding.member.endswith("(stale skip)"):
            print(f"STALE SKIP, fix the skip list, not the corpus: {finding}", file=sys.stderr)
            return 1
        if finding.recomputed == "<unrecomputable>":
            print(f"CANNOT REGENERATE: {finding}", file=sys.stderr)
            return 1
        applied.append(_apply_surgical(finding.vector, finding.member,
                                       finding.published, finding.recomputed))
    if applied:
        print(f"regenerated {len(applied)} derived member(s):")
        for line in applied:
            print(f"  {line}")
        print("\nnow re-freeze the pins: python verifier/freeze_corpus_lock.py")
    else:
        print("corpus B already consistent; nothing changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
