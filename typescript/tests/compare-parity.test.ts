/** Classifier comparison parity, TS half (488 C20). */

import { describe, expect, it } from "vitest";

import { COMPARANDA, compareAssessments, type AssessmentView } from "../src/verifier/agreement.js";
import { attributedToTrustTime, classify } from "../src/verifier/disagreement.js";

const SHA = "aa".repeat(32);

function view(over: Partial<AssessmentView> = {}): AssessmentView {
  return {
    docSha256: SHA,
    axes: { expiry: "PASS" },
    context: { evaluator: "c20", evaluationTime: "T1" },
    ...over,
  };
}

describe("compare parity", () => {
  it("reports single-dimension differences exactly", () => {
    const base = view();
    const onlyEval = classify(base, view({ context: { evaluator: "other", evaluationTime: "T1" } }), "v", "v");
    expect(onlyEval.differing).toEqual(["evaluator"]);
    const onlyAxes = classify(base, view({ axes: { expiry: "FAIL" } }), "v", "v");
    expect(onlyAxes.differing).toEqual(["contentHashes"]);
    const onlyTrust = classify(
      base,
      view({ context: { evaluator: "c20", evaluationTime: "T1", trustRoots: ["root-b"] } }),
      "v",
      "v",
    );
    expect(onlyTrust.differing).toEqual(["trustRoots"]);
  });

  it("orders multi-dimension differences by comparanda", () => {
    const moved = view({
      axes: { expiry: "FAIL" },
      context: { evaluator: "other", evaluationTime: "T2", trustRoots: ["r"] },
    });
    const out = classify(view(), moved, "v", "v");
    expect(out.label).toBe("agreement");
    expect(out.differing).toEqual(["contentHashes", "evaluator", "trustRoots", "verificationTime"]);
    expect(out.differing).toEqual([...COMPARANDA].filter((n) => out.differing.includes(n)));
  });

  it("reads absent, null and sibling null the same", () => {
    const missing = view({ context: {} });
    const noneView = view({ context: { evaluationTime: null } });
    expect(compareAssessments(missing, noneView)["verificationTime"]).toBe(true);
    const nulled = view({ context: { evaluationTime: null, evaluation_time: "T1" } });
    expect(compareAssessments(nulled, missing)["verificationTime"]).toBe(false);
    const camelOnly = view({ context: { evaluation_time: "T1" } });
    expect(Object.values(compareAssessments(nulled, camelOnly)).every(Boolean)).toBe(true);
  });

  it("compares a view without axes as empty", () => {
    const base = view();
    const noAxes = { docSha256: SHA, context: base.context } as unknown as AssessmentView;
    expect(compareAssessments(noAxes, noAxes)["contentHashes"]).toBe(true);
    expect(compareAssessments(noAxes, base)["contentHashes"]).toBe(false);
  });
});

/** Identical-facts/different-interpretation vector (488 C9), TS half. */
const C9_CTX = {
  evaluator: "c9-probe",
  trustRoots: ["root-a"],
  revocationSnapshot: "snap-1",
  evaluationTime: "2029-06-01T00:00:00Z",
  policyRevision: "pol-7",
  evidenceKnowledgeCutoff: "cut-3",
};
const C9_AXES = { expiry: "PASS", skew: "PASS" };
const C9_SHA = "cc".repeat(32);

function c9view(context: Record<string, unknown> = C9_CTX): AssessmentView {
  return { docSha256: C9_SHA, axes: { ...C9_AXES }, context: { ...context } };
}

describe("identical facts vector", () => {
  it("matches on all comparanda when identical", () => {
    expect(compareAssessments(c9view(), c9view())).toEqual(
      Object.fromEntries(COMPARANDA.map((n) => [n, true])),
    );
  });

  it("attributes a policy-revision flip, never unexplained", () => {
    const other = c9view({ ...C9_CTX, policyRevision: "pol-9" });
    expect(compareAssessments(c9view(), other)["policyRevision"]).toBe(false);
    const out = classify(c9view(), other, "verified", "unverified");
    expect(out.label).toBe("attributed_to_inputs");
    expect(out.differing).toEqual(["policyRevision"]);
    expect(attributedToTrustTime(out)).toBe(false);
  });

  it("attributes a cutoff flip under the same verdict", () => {
    const other = c9view({ ...C9_CTX, evidenceKnowledgeCutoff: "cut-4" });
    const out = classify(c9view(), other, "verified", "verified");
    expect(out.label).toBe("agreement");
    expect(out.differing).toEqual(["evidenceKnowledgeCutoff"]);
    expect(attributedToTrustTime(out)).toBe(true);
  });

  it("reads absent fields equal for old entries", () => {
    const bare = { docSha256: C9_SHA, axes: { ...C9_AXES }, context: { evaluator: "c9-probe" } };
    expect(compareAssessments(bare, { ...bare })["policyRevision"]).toBe(true);
    const out = classify(bare, { ...bare }, "verified", "verified");
    expect(out.label).toBe("agreement");
    expect(out.differing).toEqual([]);
  });

  it("reads dual-spelled contexts camel-first", () => {
    const both = { ...C9_CTX, policy_revision: "pol-7", policyRevision: "pol-7" };
    const camelFlip = { ...both, policyRevision: "pol-9" };
    expect(compareAssessments(c9view(both), c9view(camelFlip))["policyRevision"]).toBe(false);
    const snakeFlip = { ...both, policy_revision: "pol-9" };
    expect(compareAssessments(c9view(both), c9view(snakeFlip))["policyRevision"]).toBe(true);
    const out = classify(c9view(both), c9view(camelFlip), "verified", "unverified");
    expect(out.label).toBe("attributed_to_inputs");
    expect(out.differing).toEqual(["policyRevision"]);
  });
});
