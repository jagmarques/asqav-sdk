# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Independent failure classification for the oracle (criterion 574 D5a).

Derived from criterion 418, not ported from any verifier's table: INVALID
means a resolved key and a cryptographic binding failed; UNVERIFIABLE
means the recompute could not complete. Each axis entry below cites the
code contract proving its FAIL implies resolved inputs (unresolvable
inputs SKIP upstream and never reach FAIL). The oracle's ``_axis``
constructor uses ``classify``; a standing corpus gate proves behavior
is preserved. Agreement of two independent derivations is evidence.

Derivation per axis (418 invalid limb: resolved key + binding failed;
anything unresolvable SKIPs upstream): signature SKIPs when no key
resolves; key_status and issuer_bind only weigh a resolved entry;
key_binding SKIPs when unrecomputable; counterparty SKIPs what it
cannot resolve; payload_digest FAILs only on carried-context
disagreement; outer_copies FAILs on disagreement with signed bytes;
nonce FAILs on a proven duplicate pair; seq FAILs on withheld or
malformed counters; parent/pdp signatures FAIL on absent-required or
failed verification with resolved keys, SKIP otherwise; anchors FAILs
on malformed/untrusted material, SKIPs when absent. Note-dependent
axes (chain/skew/structure/expiry) split proven breaks from stops
by note; anything unlisted never proves a break.
"""
from __future__ import annotations

from . import crypto

#: Failure classes (criterion 418). Literals, not logic: pinned equal
#: to the oracle core's by the vocabulary test below in spirit.
FAILURE_INVALID = "invalid"
FAILURE_UNVERIFIABLE = "unverifiable"

#: Axes whose FAIL proves a binding failed (418 invalid; see docstring).
_ALWAYS_INVALID_AXES = frozenset(
    {
        "signature",
        "key_status",
        "issuer_bind",
        "key_binding",
        "counterparty",
        "payload_digest",
        "outer_copies",
        "nonce",
        "seq",
        "parent_signature",
        "pdp_signature",
        "anchors",
    }
)


def classify(axis: str, result: str, note: str) -> str | None:
    """Map one axis outcome to its failure class (criterion 418).

    PASS carries none; SKIPPED means the recompute could not complete.
    A FAIL on a table axis is a proven binding failure (invalid); the
    note-dependent axes below split proven breaks (invalid) from stops
    (unverifiable); anything unlisted never proves a break.
    """
    if result == crypto.PASS:
        return None
    if result == crypto.SKIPPED:
        return FAILURE_UNVERIFIABLE
    if axis in _ALWAYS_INVALID_AXES:
        return FAILURE_INVALID
    if axis == "chain":
        # A mismatched link or a cross-format predecessor is a proven
        # break; a predecessor the canonicaliser cannot walk is a stop.
        if note.startswith("chain break:") or note == "predecessor is a different receipt format":
            return FAILURE_INVALID
        return FAILURE_UNVERIFIABLE
    if axis == "skew":
        if note.startswith("unparseable issued_at"):
            return FAILURE_UNVERIFIABLE
        return FAILURE_INVALID
    if axis == "structure":
        if note.startswith("unsupported ACTA alg") or note.startswith(
            "unsupported signature algorithm"
        ):
            return FAILURE_INVALID
        if note.startswith("key_purpose mismatch"):
            return FAILURE_INVALID
        if "signing-key DID != issuer DID" in note:
            return FAILURE_INVALID
        if note.startswith("proof @context is not a prefix"):
            return FAILURE_INVALID
        if note.startswith("unregistered type ") or note.startswith(
            "malformed type "
        ):
            return FAILURE_INVALID
        return FAILURE_UNVERIFIABLE
    if axis == "expiry":
        if note.startswith("unreadable expires_at"):
            return FAILURE_UNVERIFIABLE
        return FAILURE_INVALID
    # issuer_key / ingest and anything unlisted: never a proven break.
    return FAILURE_UNVERIFIABLE
