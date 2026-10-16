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
