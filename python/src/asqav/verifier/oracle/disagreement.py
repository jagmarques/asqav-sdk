"""Disagreement classification over normalized comparanda (488 C13, C2 core).

Labels only what the comparanda support: a verdict difference with no
input difference is unexplained (investigate the code, never assume a
spec bug); a difference with input differences attributes to those
inputs. No new taxonomy beyond the criterion's own distinction."""
from __future__ import annotations

from dataclasses import dataclass

from asqav.verifier.oracle.agreement import COMPARANDA, compare_assessments

#: Comparanda that vary legitimately across trust environments / time.
TRUST_TIME_DIMS = frozenset({"trust_roots", "revocation_snapshot", "verification_time"})


@dataclass(frozen=True)
class Disagreement:
    """Outcome of classifying two retained assessments of one subject."""

    label: str
    differing: tuple[str, ...] = ()


def classify(first: dict, second: dict, verdict_a: str, verdict_b: str) -> Disagreement:
    """Classify a verdict pair against normalized input comparanda."""
    compared = compare_assessments(first, second)
    if not compared["evidence_bytes"]:
        return Disagreement(label="different_subject")
    differing = tuple(name for name in COMPARANDA if not compared[name])
    if verdict_a == verdict_b:
        return Disagreement(label="agreement", differing=differing)
    if not differing:
        return Disagreement(label="unexplained_needs_investigation")
    return Disagreement(label="attributed_to_inputs", differing=differing)


def attributed_to_trust_time(result: Disagreement) -> bool:
    """True when every differing dim is a trust/time input (C2's own line)."""
    return bool(result.differing) and set(result.differing) <= TRUST_TIME_DIMS
