/** Agreement comparanda, TS half (488 C12). */

import { describe, expect, it } from "vitest";

import {
  COMPARANDA,
  agreementOfTerms,
  compareAssessments,
  type AssessmentView,
} from "../src/verifier/agreement.js";

const CTX = {
  evaluator: "c12-probe",
  trustRoots: ["root-a", "root-b"],
  revocationSnapshot: "snap-1",
  evaluationTime: "2029-06-01T00:00:00Z",
};
const AXES = { expiry: "PASS", skew: "PASS" };
const AGR = agreementOfTerms("agr-1", "2029-01", { clause: "pay on delivery" });

function view(over: Partial<AssessmentView> = {}): AssessmentView {
  return {
    docSha256: "dd".repeat(32),
    axes: { ...AXES },
    context: { ...CTX },
    agreement: AGR,
    ...over,
  };
}

describe("agreement comparanda", () => {
  it("matches on all seven for identical views", () => {
    expect(compareAssessments(view(), view())).toEqual(
      Object.fromEntries(COMPARANDA.map((k) => [k, true])),
    );
  });

  it("flips only agreementVersion on changed terms", () => {
    const other = view({ agreement: agreementOfTerms("agr-1", "2029-01", { clause: "pay upfront" }) });
    const result = compareAssessments(view(), other);
    expect(result.agreementVersion).toBe(false);
    expect(Object.entries(result).filter(([k]) => k !== "agreementVersion").every(([, v]) => v)).toBe(true);
  });

  it("flips only verificationTime on a later evaluation", () => {
    const other = view({ context: { ...CTX, evaluationTime: "2031-01-01T00:00:00Z" } });
    const result = compareAssessments(view(), other);
    expect(result.verificationTime).toBe(false);
    expect(Object.entries(result).filter(([k]) => k !== "verificationTime").every(([, v]) => v)).toBe(true);
  });

  it("treats a missing agreement as mismatch, not a crash", () => {
    const result = compareAssessments(view({ agreement: null }), view());
    expect(result.agreementVersion).toBe(false);
    expect(result.evidenceBytes).toBe(true);
  });

  it("reads snake_case contexts from the other language", () => {
    const snake = view({
      context: {
        evaluator: "c12-probe",
        trust_roots: ["root-b", "root-a"],
        revocation_snapshot: "snap-1",
        evaluation_time: "2029-06-01T00:00:00Z",
      },
    });
    expect(compareAssessments(view(), snake)).toEqual(
      Object.fromEntries(COMPARANDA.map((k) => [k, true])),
    );
  });
});
