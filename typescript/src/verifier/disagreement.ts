/** Disagreement classification over normalized comparanda (488 C13). */

import { COMPARANDA, compareAssessments, type AssessmentView } from "./agreement.js";

export const TRUST_TIME_DIMS = new Set(["trustRoots", "revocationSnapshot", "verificationTime"]);

export interface Disagreement {
  readonly label: string;
  readonly differing: string[];
}

export function classify(
  first: AssessmentView,
  second: AssessmentView,
  verdictA: string,
  verdictB: string,
): Disagreement {
  const compared = compareAssessments(first, second);
  if (!compared.evidenceBytes) return { label: "different_subject", differing: [] };
  const differing = COMPARANDA.filter((name) => !compared[name]);
  if (verdictA === verdictB) return { label: "agreement", differing: [...differing] };
  if (differing.length === 0) return { label: "unexplained_needs_investigation", differing: [] };
  return { label: "attributed_to_inputs", differing: [...differing] };
}

export function attributedToTrustTime(result: Disagreement): boolean {
  return result.differing.length > 0 && result.differing.every((d) => TRUST_TIME_DIMS.has(d));
}

export interface SpecCase {
  readonly label: string;
  readonly note: string;
  readonly pinned: string[];
}

export function specCase(
  first: AssessmentView,
  second: AssessmentView,
  verdictA: string,
  verdictB: string,
  rerunA: AssessmentView | null = null,
  rerunB: AssessmentView | null = null,
): SpecCase {
  if (classify(first, second, verdictA, verdictB).label !== "unexplained_needs_investigation") {
    return { label: "not_applicable", note: "no unexplained split to attribute", pinned: [] };
  }
  if (!first.docSha256 || !second.docSha256) {
    return { label: "evidence_insufficient", note: "no digest to pin the split to", pinned: [] };
  }
  if (rerunA === null || rerunB === null) {
    return { label: "evidence_insufficient", note: "re-execute both inputs first", pinned: [] };
  }
  for (const [tag, orig, rerun] of [["first", first, rerunA], ["second", second, rerunB]] as const) {
    if (rerun.docSha256 !== orig.docSha256) {
      return { label: "evidence_insufficient", note: `${tag} rerun is not the same inputs`, pinned: [] };
    }
    if (!compareAssessments(orig, rerun).contentHashes) {
      return { label: "nondeterministic_flake", note: `${tag} axes moved between runs`, pinned: [] };
    }
  }
  return { label: "spec_bug_candidate", note: "split reproduces on identical inputs", pinned: [first.docSha256] };
}
