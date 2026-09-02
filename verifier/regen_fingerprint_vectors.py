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

"""Regenerate EVERY derived member of conformance/vectors.json from each vector's input.

The corpus is a contract with third-party implementers. Before this script the
regeneration covered ``canonical`` and ``sha256`` only, so a derived member that
no test recomputed could go stale in the published file and stay stale: the
counterparty ``envelope_hash`` family did exactly that across five vectors,
carrying a digest taken with a genesis ``previousReceiptHash`` form the corpus
had already stopped using.

Derived members this script owns, per vector:

* ``canonical``  - the RFC 8785 (JCS) bytes of ``input``, as a UTF-8 string
* ``sha256``     - SHA-256 of those bytes, hex
* ``input.counterparty_binding.envelope_hash`` and the ``expected``
  ``envelope_hash_*`` renderings - the digest of the ORIGINATING envelope named
  by ``expected.originating_envelope_ref``, taken under the scope declared in
  ``input.counterparty_binding.scope``

Everything else in a vector is authored, not derived, and is left untouched.

Run ``python verifier/regen_fingerprint_vectors.py`` after any authored edit, then
``python verifier/freeze_corpus_lock.py`` to move the pins. ``--check`` exits
nonzero on any drift without writing, which is what CI runs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
VECTORS_PATH = _ROOT / "conformance" / "vectors.json"

#: The only counterparty digest scope this revision emits. An absent scope member
#: means a legacy -04..-08 three-key binding and is reported, never silently retried.
SCOPE_MINUS_ANCHORS = "envelope_minus_anchors"

#: Envelope members the minus-anchors scope covers, in the order the draft states them.
MINUS_ANCHORS_MEMBERS = ("payload", "signature")


def canonical_json(obj: Any) -> bytes:
    """RFC 8785 canonical bytes.

    Object keys sort by UTF-16 code unit, NOT by code point: the two orders agree
    across the BMP and diverge above U+FFFF. ``json.dumps(sort_keys=True)`` is code
    point order and is not conformant here.
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


def envelope_digest(envelope: dict[str, Any], scope: str) -> bytes:
    """Raw SHA-256 of the originating envelope under ``scope``."""
    if scope != SCOPE_MINUS_ANCHORS:
        raise ValueError(f"unknown counterparty binding scope: {scope!r}")
    missing = [m for m in MINUS_ANCHORS_MEMBERS if m not in envelope]
    if missing:
        raise ValueError(f"originating envelope is missing {missing}")
    scoped = {m: envelope[m] for m in MINUS_ANCHORS_MEMBERS}
    return hashlib.sha256(canonical_json(scoped)).digest()


def _vectors_by_name(vectors: list[dict]) -> dict[str, dict]:
    return {v["name"]: v for v in vectors}


def regenerate(doc: dict) -> list[str]:
    """Rewrite every derived member in place. Returns one line per change."""
    vectors = doc["vectors"]
    by_name = _vectors_by_name(vectors)
    changes: list[str] = []

    def record(vector_name: str, member: str, old: Any, new: Any) -> None:
        if old != new:
            changes.append(f"{vector_name}: {member}\n    was {old!r}\n    now {new!r}")

    # Pass 1: the counterparty bindings, because they sit INSIDE `input` and
    # therefore change the canonical bytes that pass 2 derives.
    for vector in vectors:
        expected = vector.get("expected")
        if not isinstance(expected, dict):
            continue
        ref = expected.get("originating_envelope_ref")
        if not ref:
            continue
        origin = by_name.get(ref)
        if origin is None:
            raise KeyError(f"{vector['name']}: originating_envelope_ref {ref!r} names no vector")

        binding = vector.get("input", {}).get("counterparty_binding")
        if not isinstance(binding, dict) or "envelope_hash" not in binding:
            # A vector that deliberately omits envelope_hash still names its origin
            # so the renderings below stay derivable; nothing to re-pin on the wire.
            digest = envelope_digest(origin["input"], SCOPE_MINUS_ANCHORS)
        else:
            scope = binding.get("scope", SCOPE_MINUS_ANCHORS)
            digest = envelope_digest(origin["input"], scope)
            # The wire value keeps whichever alphabet the vector is exercising.
            old_hash = binding["envelope_hash"]
            urlsafe = "-" in old_hash or "_" in old_hash
            new_hash = (
                base64.urlsafe_b64encode(digest) if urlsafe else base64.b64encode(digest)
            ).decode()
            record(vector["name"], "input.counterparty_binding.envelope_hash", old_hash, new_hash)
            binding["envelope_hash"] = new_hash

        renderings = {
            "envelope_hash_base64": base64.b64encode(digest).decode(),
            "envelope_hash_base64url": base64.urlsafe_b64encode(digest).decode(),
            "envelope_hash_hex": digest.hex(),
        }
        for member, value in renderings.items():
            if member in expected:
                record(vector["name"], f"expected.{member}", expected[member], value)
                expected[member] = value

    # Pass 2: canonical bytes and their digest, for every vector.
    for vector in vectors:
        if "input" not in vector:
            continue
        canonical = canonical_json(vector["input"]).decode("utf-8")
        if "canonical" in vector:
            record(vector["name"], "canonical", f"<{len(vector['canonical'])} chars>",
                   f"<{len(canonical)} chars>")
        vector["canonical"] = canonical
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if "sha256" in vector:
            record(vector["name"], "sha256", vector["sha256"], digest)
        vector["sha256"] = digest

    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if any derived member differs from the published file; write nothing",
    )
    args = parser.parse_args()

    original_text = VECTORS_PATH.read_text()
    doc = json.loads(original_text)
    changes = regenerate(doc)
    regenerated = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if regenerated != original_text:
            print("corpus integrity: DRIFT between the published file and its own inputs", file=sys.stderr)
            for line in changes:
                print(f"  {line}", file=sys.stderr)
            if not changes:
                print("  (formatting-only drift; run without --check to normalise)", file=sys.stderr)
            print(
                "\nregenerate with: python verifier/regen_fingerprint_vectors.py"
                "\nthen re-freeze:  python verifier/freeze_corpus_lock.py",
                file=sys.stderr,
            )
            return 1
        print(f"corpus integrity: every derived member of {len(doc['vectors'])} vectors re-derives")
        return 0

    VECTORS_PATH.write_text(regenerated)
    if changes:
        print(f"regenerated {VECTORS_PATH.name}: {len(changes)} derived member(s) changed")
        for line in changes:
            print(f"  {line}")
        print("\nnow re-freeze the pins: python verifier/freeze_corpus_lock.py")
    else:
        print(f"{VECTORS_PATH.name} already consistent; nothing changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
