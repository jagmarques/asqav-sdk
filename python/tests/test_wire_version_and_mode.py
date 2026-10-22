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

"""`v` and `mode` are REQUIRED on every asqav-native vector and forbidden elsewhere; the draft requires both
on the payload (5.1.1 ``v``, 5.1.2 ``mode``). Only a broken signature catches a missing member: an EDITED
vector, not a REGENERATED one, so a re-signing generator yields a valid receipt. Presence is asserted against
the bytes, not a verdict. 5.3 makes the ABSENCE of ``v`` the discriminator from a bare upstream receipt, so
non-asqav families MUST NOT gain it: that deletes the only negative evidence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from asqav.verifier.oracle import ADAPTERS, crypto, verify

_VECTOR_ROOT = Path(__file__).parent.parent.parent / "verifier" / "conformance-vectors"

#: Recognised versions. asqav-08/09 pin ``v: 2`` as a VERIFIED signer canary:
#: at least {1, 2}; widening it is a registry decision, not a test edit.
RECOGNISED_V = frozenset({1, 2})

#: The two capture modes the corpus uses: ``hash`` for the flat shape whose
#: ``payload`` is null; ``payload`` for a real signed payload object.
RECOGNISED_MODE = frozenset({"hash", "payload"})

#: Families that MUST NOT carry ``v``. Their receipts are bare upstream-format
#: receipts and their lack of ``v`` is what 5.3 discriminates on.
FOREIGN_PREFIXES = ("acta-", "agentreceipts-", "aerf-", "pipelock-", "w3c-", "authproof-", "dsse-")

#: Negative vectors for 5.2.1 (criterion 674): a missing or unrecognised ``v``
#: under a valid signature, unverified by both oracles; exempt from the census.
VERSION_NEGATIVE_VECTORS = frozenset(
    {
        "asqav-39-absent-version",
        "asqav-40-unrecognised-version",
        "asqav-41-hash-unrecognised-version",
    }
)


def _receipt_files(prefix: str):
    """Yield (label, parsed json) for every receipt-shaped file under a family."""
    for path in sorted(_VECTOR_ROOT.glob(f"{prefix}*/**/*.json")):
        if path.name not in ("receipt.json", "predecessor.json"):
            continue
        yield f"{path.parent.name}/{path.name}", json.loads(path.read_text())


def _signed_members(doc: dict) -> tuple[dict, str]:
    """Return the object the signed members live in, and which position that is. Payload-mode receipts carry
    them inside ``payload``; the flat hash-mode shape has ``payload`` null and carries them at the top level
    (5.1.1 states both positions), so a checker that only looks inside ``payload`` reports a false absence on
    exactly the production-captured vectors.
    """
    payload = doc.get("payload")
    if isinstance(payload, dict):
        return payload, "payload"
    return doc, "top-level"


def _asqav_receipts():
    return list(_receipt_files("asqav-"))


def test_every_asqav_vector_carries_v_and_mode() -> None:
    files = _asqav_receipts()
    assert files, "no asqav-* receipts found - the glob is wrong, not the corpus"

    offenders = []
    for label, doc in files:
        if label.split("/")[0] in VERSION_NEGATIVE_VECTORS:
            continue
        members, position = _signed_members(doc)
        v, mode = members.get("v"), members.get("mode")
        if v is None or mode is None:
            offenders.append((label, position, v, mode))

    assert not offenders, (
        "v (5.1.1) and mode (5.1.2) are REQUIRED on every asqav-native signed payload. "
        "A vector missing one is not a Compliance Receipt of this document, and a "
        "verifier MUST NOT report it verified - so publishing it as `verified` is the "
        f"defect this file exists to prevent: {offenders}"
    )


def test_v_and_mode_carry_only_recognised_values() -> None:
    bad = []
    for label, doc in _asqav_receipts():
        if label.split("/")[0] in VERSION_NEGATIVE_VECTORS:
            continue
        members, _ = _signed_members(doc)
        v, mode = members.get("v"), members.get("mode")
        if v is not None and (isinstance(v, bool) or v not in RECOGNISED_V):
            bad.append((label, "v", v))
        if mode is not None and mode not in RECOGNISED_MODE:
            bad.append((label, "mode", mode))

    assert not bad, (
        f"v must be one of {sorted(RECOGNISED_V)} and mode one of {sorted(RECOGNISED_MODE)}. "
        "Widening either set is a registry decision that belongs in the draft first, "
        f"never a quiet edit here: {bad}"
    )


def test_a_flat_shaped_receipt_is_always_hash_mode() -> None:
    """A receipt whose ``payload`` is null cannot be payload-mode. Only this ONE direction holds, and the
    corpus is the reason the converse is not asserted here: ``asqav-08``, ``asqav-09`` and ``asqav-24`` carry
    a real payload OBJECT and ``mode`` ``hash``. That is correct, not a defect: ``mode`` records how the
    ACTION was captured (content or digest), not the envelope shape, so a hash-mode receipt can still carry a
    payload object holding metadata and the digest.
    """
    wrong = [
        (label, members.get("mode"))
        for label, doc in _asqav_receipts()
        for members, position in [_signed_members(doc)]
        if position == "top-level" and members.get("mode") != "hash"
    ]

    assert not wrong, (
        "a receipt whose payload member is null carried no payload content, so it "
        f"cannot be payload-mode: {wrong}"
    )


@pytest.mark.parametrize("prefix", FOREIGN_PREFIXES)
def test_foreign_families_never_gain_v(prefix: str) -> None:
    """The ABSENCE of `v` is 5.3's discriminator, so these must stay without it."""
    offenders = []
    for label, doc in _receipt_files(prefix):
        members, _ = _signed_members(doc)
        if "v" in members:
            offenders.append((label, members["v"]))

    assert not offenders, (
        f"a {prefix}* receipt gained a v member. 5.3 discriminates this profile from the "
        "bare upstream format on exactly that absence, so adding v here does not tidy the "
        "corpus, it deletes the only negative evidence the rule has: "
        f"{offenders}"
    )


_ED25519_AVAILABLE = crypto.verify_ed25519(b"\x00" * 32, b"m", b"\x00" * 64)[0] != crypto.SKIPPED
requires_ed25519 = pytest.mark.skipif(
    not _ED25519_AVAILABLE, reason="cryptography not installed; Ed25519 verify SKIPs"
)


def _41() -> tuple[dict, dict]:
    base = _VECTOR_ROOT / "asqav-41-hash-unrecognised-version"
    return json.loads((base / "receipt.json").read_text()), json.loads(
        (base / "jwks.json").read_text()
    )


@requires_ed25519
def test_displayed_v_binds_hash_signature() -> None:
    """674 C3: asqav-41 displays v=99 with a signature over v=1 bytes; the
    signature axis must FAIL, not PASS beside structure SKIPPED."""
    receipt, jwks = _41()
    res = verify(receipt, ADAPTERS, key_provider=jwks)
    assert res.axis("signature").result == crypto.FAIL
    assert res.verdict == "unverified"


@requires_ed25519
def test_relabelled_v2_does_not_verify() -> None:
    """One-char relabel v=99->2: structure may PASS, but the signature over
    the displayed v must FAIL; the verdict stays unverified."""
    receipt, jwks = _41()
    receipt = copy.deepcopy(receipt)
    receipt["v"] = 2
    res = verify(receipt, ADAPTERS, key_provider=jwks)
    assert res.axis("signature").result == crypto.FAIL
    assert res.verdict == "unverified"
