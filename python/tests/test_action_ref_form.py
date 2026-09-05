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

"""One `action_ref` form across both corpora, and the three named exceptions.

Every asqav-native `action_ref` carries the one wire form: ``sha256:<64
lowercase hex>``. That is the whole rule - an opaque producer-chosen identifier
is not a digest and does not join.

NAMED EXCEPTIONS, deliberately not collapsed to ``sha256:`` +
``payload_digest.hash`` (the form is the rule; equality with the payload digest
is not - `action_ref` is a producer-asserted pointer, and nothing in the
profile binds it to the payload digest):

* ``asqav-06-mldsa65-payload-prod`` - a prod-shaped capture carrying a genuine
  caller-supplied digest; the wire-legal case the corpus should contain.
* ``asqav-12-time-edge-expiry`` - carries abab... against payload_digest
  cdcd... precisely so a verifier confusing the two members fails; collapsing
  them would delete that coverage while the vector still passed.
* ``asqav-13-dup-member-nested`` - a duplicate-member refusal vector whose
  empty-digest ref is deliberate.

Every OTHER directory vector's `action_ref` equals ``sha256:`` +
``payload_digest.hash``, which the generators now bind by construction; the
second test keeps that invariant from drifting without enshrining it as a wire
rule the format does not have.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

FORM = re.compile(r"^sha256:[0-9a-f]{64}$")

FINGERPRINT_PATH = Path(__file__).parent.parent.parent / "conformance" / "vectors.json"
_VECTOR_ROOT = Path(__file__).parent.parent.parent / "verifier" / "conformance-vectors"

#: Directory vectors whose authored action_ref deliberately differs from
#: payload_digest.hash (see the module docstring for the reasons).
AUTHORED_REF_VECTORS = frozenset(
    {
        "asqav-06-mldsa65-payload-prod",
        "asqav-12-time-edge-expiry",
        "asqav-13-dup-member-nested",
    }
)


def _iter_action_refs():
    """Yield (location, action_ref, payload_digest_hash_or_None) over both corpora."""
    for path in sorted(_VECTOR_ROOT.glob("asqav-*/**/*.json")):
        if path.name not in ("receipt.json", "predecessor.json"):
            continue
        payload = json.loads(path.read_text()).get("payload") or {}
        ref = payload.get("action_ref")
        if ref is None:
            continue
        digest = payload.get("payload_digest")
        yield (
            f"{path.parent.name}/{path.name}",
            ref,
            digest.get("hash") if isinstance(digest, dict) else None,
            path.parent.name,
        )
    for vec in json.loads(FINGERPRINT_PATH.read_text())["vectors"]:
        inp = vec.get("input", {})
        nested = inp.get("payload") if isinstance(inp.get("payload"), dict) else {}
        refs = [r for r in (inp.get("action_ref"), nested.get("action_ref")) if r]
        digests = [
            d.get("hash")
            for d in (inp.get("payload_digest"), nested.get("payload_digest"))
            if isinstance(d, dict)
        ]
        for ref in refs:
            yield (f"vectors.json:{vec['name']}", ref, digests[0] if digests else None, vec["name"])


def test_every_action_ref_carries_the_one_wire_form() -> None:
    offenders = [(loc, ref) for loc, ref, _, _ in _iter_action_refs() if not FORM.match(ref)]
    assert not offenders, (
        "action_ref admits exactly one form, sha256:<64 lowercase hex>; "
        "an opaque producer-chosen identifier is not a digest and does not join: "
        f"{offenders}"
    )


def test_only_the_named_exceptions_differ_from_payload_digest() -> None:
    unexpected = []
    for loc, ref, digest_hash, vector in _iter_action_refs():
        if digest_hash and ref != f"sha256:{digest_hash}" and vector not in AUTHORED_REF_VECTORS:
            unexpected.append((loc, ref))
    assert not unexpected, (
        "every generator-owned action_ref equals sha256:+payload_digest.hash; "
        "a new divergence belongs in AUTHORED_REF_VECTORS with its reason, "
        f"never in silence: {unexpected}"
    )
