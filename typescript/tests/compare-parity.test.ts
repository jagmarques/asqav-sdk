/** Classifier comparison parity, TS half (488 C20). */

import { describe, expect, it } from "vitest";

import { COMPARANDA, compareAssessments, type AssessmentView } from "../src/verifier/agreement.js";
import { classify } from "../src/verifier/disagreement.js";

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
