/** Conflict-kind vocabulary tests, TS half (488 C7). */

import { describe, expect, it } from "vitest";

import { agreementOfTerms, type AssessmentView } from "../src/verifier/agreement.js";
import { conflictKind } from "../src/verifier/disagreement.js";

const CTX = { evaluator: "c34-probe" };
const AXES = { expiry: "PASS", skew: "PASS" };
const TERMS_A = agreementOfTerms("agr-1", "2029-01", { clause: "pay on delivery" });
const TERMS_B = agreementOfTerms("agr-1", "2029-01", { clause: "refund in full" });

function view(over: Partial<AssessmentView> = {}): AssessmentView {
  return {
    docSha256: "ee".repeat(32),
    axes: { ...AXES },
    context: { ...CTX },
    agreement: TERMS_A,
    ...over,
  };
}

describe("conflict kind", () => {
  it("names contradictory signed terms", () => {
    expect(conflictKind(view(), view({ agreement: TERMS_B }))).toBe("contradictory_terms");
  });

  it("names missing evidence when the first side is empty", () => {
    expect(conflictKind(view({ axes: {} }), view())).toBe("missing_evidence");
  });

  it("names missing evidence when the second side is empty", () => {
    expect(conflictKind(view(), view({ axes: {} }))).toBe("missing_evidence");
  });

  it("makes no call when both sides are empty", () => {
    expect(conflictKind(view({ axes: {} }), view({ axes: {} }))).toBeNull();
  });

  it("makes no call across subjects", () => {
    expect(conflictKind(view(), view({ docSha256: "ff".repeat(32) }))).toBeNull();
  });
});
