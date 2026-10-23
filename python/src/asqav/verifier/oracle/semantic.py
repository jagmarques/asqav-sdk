"""Independent verdict semantics for the oracle (criterion 574 D5).
418, 438 and 426 applied from the normative rules, not ported from a fold."""
from __future__ import annotations

from typing import Any

from . import crypto

#: Shared literals, separate logic: pinned equal to the oracle core's
#: by the vocabulary test; the aggregation below stays independent.
VERDICT_VERIFIED = "verified"
VERDICT_VERIFIED_KEYED = "verified_keyed"
VERDICT_UNVERIFIED = "unverified"
FAILURE_INVALID = "invalid"
FAILURE_UNVERIFIABLE = "unverifiable"

#: Axes that report but never fold the verdict (criterion 426: expiry).
_NON_FOLDING_AXES = frozenset({"expiry"})
#: A skipped check blocks, except the chain axis (see module docstring).
_TOLERATED_SKIP_AXES = frozenset({"chain"})

#: (invalid, other, skip, keyed) -> (verdict, class); rows cite rules.
#: invalid = proved binding failure (418); keyed never rescues a failure.
_TABLE: dict[tuple[bool, bool, bool, bool], tuple[str, str | None]] = {
    (False, False, False, False): ("verified", None),  # clean receipt
    (False, False, False, True): ("verified_keyed", None),  # 438
    (False, False, True, False): ("unverified", "unverifiable"),  # gap
    (False, False, True, True): ("unverified", "unverifiable"),  # gap > keyed
    (False, True, False, False): ("unverified", "unverifiable"),  # 418
    (False, True, False, True): ("unverified", "unverifiable"),  # 418 > keyed
    (False, True, True, False): ("unverified", "unverifiable"),  # 418
    (False, True, True, True): ("unverified", "unverifiable"),  # 418 > keyed
    (True, False, False, False): ("unverified", "invalid"),  # 418 dominance
    (True, False, False, True): ("unverified", "invalid"),  # 418 > keyed
    (True, False, True, False): ("unverified", "invalid"),  # 418 dominance
    (True, False, True, True): ("unverified", "invalid"),  # 418 > keyed
    (True, True, False, False): ("unverified", "invalid"),  # 418 dominance
    (True, True, False, True): ("unverified", "invalid"),  # 418 > keyed
    (True, True, True, False): ("unverified", "invalid"),  # 418 dominance
    (True, True, True, True): ("unverified", "invalid"),  # 418 > keyed
}


def fold_independent(axes: list[Any], keyed: bool) -> tuple[str, str | None]:
    """Fold per-axis outcomes by table lookup (see _TABLE for the rules)."""
    invalid = False
    other = False
    for a in axes:
        if a.axis in _NON_FOLDING_AXES:
            continue
        if a.result == crypto.FAIL:
            if a.failure_class == FAILURE_INVALID:
                invalid = True
            else:
                other = True
    blocked = any(
        a.result == crypto.SKIPPED and a.axis not in _TOLERATED_SKIP_AXES for a in axes
    )
    return _TABLE[(invalid, other, blocked, keyed)]
