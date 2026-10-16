/** Spec-bug evidence bar, TS half (488 C23). */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { axesDict, contextDict } from "../src/verifier/retain.js";
import { classify, specCase, type SpecCase } from "../src/verifier/disagreement.js";
import type { AssessmentView } from "../src/verifier/agreement.js";

const CTX = {
  evaluator: "c23-probe",
  trustRoots: ["root-a"],
  revocationSnapshot: "snap-1",
  evaluationTime: "2029-06-01T00:00:00Z",
};
const AXES = { expiry: "PASS", skew: "PASS" };
const DOC = "ee".repeat(32);
const FIX = join(__dirname, "..", "..", "verifier", "signed-conflict-fixtures");

function view(over: Partial<AssessmentView> = {}): AssessmentView {
  return { docSha256: DOC, axes: { ...AXES }, context: { ...CTX }, agreement: null, ...over };
}

describe("spec case", () => {
  it("agreement is not applicable", () => {
    const out = specCase(view(), view(), "verified", "verified", view(), view());
    expect(out.label).toBe("not_applicable");
    expect(out.pinned).toEqual([]);
  });

  it("different subject is not applicable", () => {
    const first = view();
    const second = view({ docSha256: "ff".repeat(32) });
    expect(classify(first, second, "verified", "unverified").label).toBe("different_subject");
    expect(specCase(first, second, "verified", "unverified", first, second).label).toBe("not_applicable");
  });

  it("unexplained without reruns is insufficient", () => {
    const out: SpecCase = specCase(view(), view(), "verified", "unverified");
    expect(out.label).toBe("evidence_insufficient");
  });

  it("rerun on other bytes is insufficient", () => {
    const out = specCase(view(), view(), "verified", "unverified", view({ docSha256: "ff".repeat(32) }), view());
    expect(out.label).toBe("evidence_insufficient");
  });

  it("moved axes are a flake, not a spec bug", () => {
    const moved = view({ axes: { expiry: "FAIL", skew: "PASS" } });
    const out = specCase(view(), view(), "verified", "unverified", moved, view());
    expect(out.label).toBe("nondeterministic_flake");
    expect(out.pinned).toEqual([]);
  });

  it("reproduced split is a candidate with pinned bytes", () => {
    const out = specCase(view(), view(), "verified", "unverified", view(), view());
    expect(out.label).toBe("spec_bug_candidate");
    expect(out.pinned).toEqual([DOC]);
  });

  it("live re-execution reproduces the view", () => {
    const raw = readFileSync(join(FIX, "conflict-allow.json"));
    const digest = createHash("sha256").update(raw).digest("hex");
    const doc = JSON.parse(raw.toString("utf-8")) as Record<string, unknown>;
    const views: AssessmentView[] = [0, 1].map(() => {
      const res = verify(doc, ADAPTERS, null, null, { evaluationTime: "2026-10-16T12:00:00Z", evaluator: "c23-probe" });
      return { docSha256: digest, axes: axesDict(res), context: contextDict(res), agreement: null };
    });
    expect(Object.keys(views[0].axes).length).toBeGreaterThan(0);
    expect(views[0]).toEqual(views[1]);
  });
});
