"""Disagreement classification over normalized comparanda (488 C13, C2 core).

Labels only what the comparanda support: a verdict difference with no
input difference is unexplained (investigate the code, never assume a
spec bug); a difference with input differences attributes to those
inputs. No new taxonomy beyond the criterion's own distinction."""
from __future__ import annotations

from dataclasses import dataclass, replace

from asqav.verifier.oracle.agreement import COMPARANDA, _norm_agreement, compare_assessments

#: Comparanda varying legitimately across trust environments and time; the
#: cutoff is a time horizon (488 C9), policy_revision is a ruleset change
TRUST_TIME_DIMS = frozenset({"trust_roots", "revocation_snapshot", "verification_time",
                             "evidence_knowledge_cutoff"})


@dataclass(frozen=True)
class Disagreement:
    """Outcome of classifying two retained assessments of one subject."""

    label: str
    differing: tuple[str, ...] = ()
    # Conflict shape for one subject (488 C39b); None when the
    # pair carries no conflict call (different bytes or no shape).
    conflict: str | None = None


def classify(first: dict, second: dict, verdict_a: str, verdict_b: str) -> Disagreement:
    """Classify a verdict pair against normalized input comparanda."""
    compared = compare_assessments(first, second)
    conflict = conflict_kind(first, second)
    if not compared["evidence_bytes"]:
        return Disagreement(label="different_subject", conflict=conflict)
    differing = tuple(name for name in COMPARANDA if not compared[name])
    if verdict_a == verdict_b:
        return Disagreement(label="agreement", differing=differing, conflict=conflict)
    if not differing:
        return Disagreement(label="unexplained_needs_investigation", conflict=conflict)
    return Disagreement(label="attributed_to_inputs", differing=differing,
                        conflict=conflict)


def attributed_to_trust_time(result: Disagreement) -> bool:
    """True when every differing dim is a trust/time input (C2's own line)."""
    return bool(result.differing) and set(result.differing) <= TRUST_TIME_DIMS


CONTRADICTORY_TERMS = "contradictory_terms"
MISSING_EVIDENCE = "missing_evidence"


def conflict_kind(first: dict, second: dict) -> str | None:
    """Name the conflict shape for one subject (488 C7).
    Same bytes with both sides carrying differing signed terms is a
    contradictory-terms conflict; same bytes with evidence on one side
    only is missing evidence. Different bytes carry no conflict call.
    Terms win only when both sides are evidenced; an empty side reads as missing first."""
    if first.get("doc_sha256") != second.get("doc_sha256"):
        return None
    first_axes = first.get("axes") or {}
    second_axes = second.get("axes") or {}
    if (not first_axes) != (not second_axes):
        return MISSING_EVIDENCE
    if not first_axes:
        return None
    first_terms = _norm_agreement(first.get("agreement")).get("terms_hash")
    second_terms = _norm_agreement(second.get("agreement")).get("terms_hash")
    if first_terms and second_terms and first_terms != second_terms:
        return CONTRADICTORY_TERMS
    return None


def mark_legacy_gap(disagreement: Disagreement, original_ref: object | None,
                     live_ref: object | None) -> Disagreement:
    """Name a storage-generation ref gap on a classified pair (488 C39b).

    Same digest means same derivability, so a ref-less retained
    original beside a live ref is a legacy store, never differing
    input; the caller already compared the pair ref-less. Name the
    gap missing-evidence unless the pair already carries a call."""
    if original_ref is None and live_ref is not None and disagreement.conflict is None:
        return replace(disagreement, conflict=MISSING_EVIDENCE)
    return disagreement


@dataclass(frozen=True)
class SpecCase:
    """Evidence-bar outcome for an unexplained verdict split (488 C23).

    Only a deterministically reproduced split on identical inputs is a
    spec_bug_candidate; every other outcome names exactly what is missing."""

    label: str
    note: str
    pinned: tuple[str, ...] = ()


def spec_case(
    first: dict,
    second: dict,
    verdict_a: str,
    verdict_b: str,
    rerun_a: dict | None = None,
    rerun_b: dict | None = None,
) -> SpecCase:
    """Apply the spec-bug evidence bar to a classified verdict pair.

    Reruns are re-executions from retained bytes (retain.replay +
    verify); a split is attributable only when both reruns reproduce
    the original axes on the original digests."""
    if classify(first, second, verdict_a, verdict_b).label != "unexplained_needs_investigation":
        return SpecCase("not_applicable", "no unexplained split to attribute")
    if not first.get("doc_sha256") or not second.get("doc_sha256"):
        return SpecCase("evidence_insufficient", "no digest to pin the split to")
    if rerun_a is None or rerun_b is None:
        return SpecCase("evidence_insufficient", "re-execute both inputs first")
    for tag, orig, rerun in (("first", first, rerun_a), ("second", second, rerun_b)):
        if rerun.get("doc_sha256") != orig.get("doc_sha256"):
            return SpecCase("evidence_insufficient", f"{tag} rerun is not the same inputs")
        if not compare_assessments(orig, rerun)["content_hashes"]:
            return SpecCase("nondeterministic_flake", f"{tag} axes moved between runs")
    return SpecCase("spec_bug_candidate", "split reproduces on identical inputs",
                    (str(first.get("doc_sha256")),))
