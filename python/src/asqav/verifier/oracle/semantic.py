# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
"""Independent verdict semantics for the oracle (criterion 574 D5).

Derived from the normative rules, not by porting ``fold_verdict``: 418
(invalid vs unverifiable never collapse), 438 (keyed digests report
verified_keyed, never plain verified), 426 (expiry reports but never
folds), and the chain tolerance (a skipped chain is the expected shape
of a genesis receipt, not a gap). The table below is the whole
aggregation; the oracle's ``verify`` uses it, and a standing corpus
gate proves it agrees with every verifier the project ships.
Agreement of two independent derivations is evidence; agreement of one
derivation with itself is not.
"""
from __future__ import annotations

from typing import Any

from . import crypto

#: Public verdict vocabulary (criteria 418/438). Literals, not logic:
#: pinned equal to the oracle core's by the vocabulary test, so the
#: surfaces keep reporting alike while the aggregation stays separate.
VERDICT_VERIFIED = "verified"
VERDICT_VERIFIED_KEYED = "verified_keyed"
VERDICT_UNVERIFIED = "unverified"
FAILURE_INVALID = "invalid"
FAILURE_UNVERIFIABLE = "unverifiable"

#: Axes that report but never fold the verdict (criterion 426: expiry).
_NON_FOLDING_AXES = frozenset({"expiry"})
#: A skipped check blocks, except the chain axis (see module docstring).
_TOLERATED_SKIP_AXES = frozenset({"chain"})


def fold_independent(axes: list[Any], keyed: bool) -> tuple[str, str | None]:
    """Fold per-axis outcomes into verdict + failure class (418/438/426).

    A proven binding failure dominates (invalid); any other failure only
    shows the recompute could not complete (unverifiable). A blocked
    check dominates a keyed digest: keying never rescues a failure.
    """
    fails = [
        a for a in axes if a.result == crypto.FAIL and a.axis not in _NON_FOLDING_AXES
    ]
    if any(a.failure_class == FAILURE_INVALID for a in fails):
        return VERDICT_UNVERIFIED, FAILURE_INVALID
    if fails:
        return VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE
    blocked = any(
        a.result == crypto.SKIPPED and a.axis not in _TOLERATED_SKIP_AXES for a in axes
    )
    if blocked:
        return VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE
    if keyed:
        return VERDICT_VERIFIED_KEYED, None
    return VERDICT_VERIFIED, None
