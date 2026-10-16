/** Disagreement classifier tests, TS half (488 C13). */

import { describe, expect, it } from "vitest";

import { agreementOfTerms, type AssessmentView } from "../src/verifier/agreement.js";
import { attributedToTrustTime, classify } from "../src/verifier/disagreement.js";

const CTX = {
  evaluator: "c13-probe",
  trustRoots: ["root-a"],
  revocationSnapshot: "snap-1",
  evaluationTime: "2029-06-01T00:00:00Z",
};
const AXES = { expiry: "PASS", skew: "PASS" };
const AGR = agreementOfTerms("agr-1", "2029-01", { clause: "pay on delivery" });

function view(over: Partial<AssessmentView> = {}): AssessmentView {
  return {
    docSha256: "ee".repeat(32),
    axes: { ...AXES },
    context: { ...CTX },
    agreement: AGR,
    ...over,
  };
}

describe("disagreement classifier", () => {
  it("calls the same verdict agreement", () => {
    expect(classify(view(), view(), "verified", "verified")).toEqual({ label: "agreement", differing: [] });
  });

  it("attributes a time-only difference to trust/time inputs", () => {
    const later = view({ context: { ...CTX, evaluationTime: "2031-01-01T00:00:00Z" } });
    const out = classify(view(), later, "verified", "unverified");
    expect(out.label).toBe("attributed_to_inputs");
    expect(out.differing).toEqual(["verificationTime"]);
    expect(attributedToTrustTime(out)).toBe(true);
  });

  it("attributes a terms difference outside trust/time", () => {
    const other = view({ agreement: agreementOfTerms("agr-1", "2029-02", { clause: "pay on delivery" }) });
    const out = classify(view(), other, "verified", "unverified");
    expect(out.label).toBe("attributed_to_inputs");
    expect(out.differing).toEqual(["agreementVersion"]);
    expect(attributedToTrustTime(out)).toBe(false);
  });

  it("marks same-input verdict splits unexplained", () => {
    expect(classify(view(), view(), "verified", "unverified")).toEqual({
      label: "unexplained_needs_investigation",
      differing: [],
    });
  });

  it("treats different bytes as a different subject", () => {
    const other = view({ docSha256: "ff".repeat(32) });
    expect(classify(view(), other, "verified", "unverified").label).toBe("different_subject");
  });

  it("keeps agreement when inputs drift but verdicts hold", () => {
    const later = view({ context: { ...CTX, evaluationTime: "2031-01-01T00:00:00Z" } });
    expect(classify(view(), later, "verified", "verified")).toEqual({
      label: "agreement",
      differing: ["verificationTime"],
    });
  });
});
