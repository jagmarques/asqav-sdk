/** C7 scenario families, TS half (488 C14). */

import { describe, expect, it } from "vitest";

import { agreementOfTerms, type AssessmentView } from "../src/verifier/agreement.js";
import { attributedToTrustTime, classify } from "../src/verifier/disagreement.js";

const DOC = "ff".repeat(32);
const T1 = "2029-06-01T00:00:00Z";
const T2 = "2031-01-01T00:00:00Z";
const CTX = {
  evaluator: "c14-probe",
  trustRoots: ["root-a"],
  revocationSnapshot: "snap-1",
  evaluationTime: T1,
  policyRevision: "2029-01",
  evidenceKnowledgeCutoff: T1,
};
const AXES_OK = { expiry: "PASS", skew: "PASS" };
const AGR = agreementOfTerms("agr-1", "2029-01", { clause: "pay on delivery" });

function view(over: Partial<AssessmentView> = {}): AssessmentView {
  return { docSha256: DOC, axes: { ...AXES_OK }, context: { ...CTX }, agreement: AGR, ...over };
}

describe("C7 scenario families", () => {
  it("family 1: success is agreement", () => {
    expect(classify(view(), view(), "verified", "verified")).toEqual({ label: "agreement", differing: [] });
  });

  it("family 2: contradictory terms are unexplained", () => {
    expect(classify(view(), view(), "verified", "unverified").label).toBe("unexplained_needs_investigation");
  });

  it("family 3: missing agreement attributes", () => {
    const out = classify(view({ agreement: null }), view(), "verified", "unverified");
    expect(out.label).toBe("attributed_to_inputs");
    expect(out.differing).toEqual(["agreementVersion"]);
    expect(attributedToTrustTime(out)).toBe(false);
  });

  it("family 4: trust/time drift is legitimate", () => {
    const later = view({ context: { ...CTX, evaluationTime: T2, evidenceKnowledgeCutoff: T2 } });
    const out = classify(view(), later, "verified", "unverified");
    expect(out.label).toBe("attributed_to_inputs");
    expect(attributedToTrustTime(out)).toBe(true);
  });
});
