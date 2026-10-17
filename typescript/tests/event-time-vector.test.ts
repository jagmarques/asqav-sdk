/** Event/effective-time vector, TS half (488 C9). */

import { describe, expect, it } from "vitest";

import { COMPARANDA, compareAssessments, type AssessmentView } from "../src/verifier/agreement.js";
import { AssessmentStore } from "../src/verifier/assessmentStore.js";
import { attributedToTrustTime, classify } from "../src/verifier/disagreement.js";

const SHA = "dd".repeat(32);
const CTX = {
  evaluator: "c9t-probe",
  trustRoots: ["root-a"],
  revocationSnapshot: "snap-1",
  evaluationTime: "2029-06-01T00:00:00Z",
  eventTime: "2029-05-01T00:00:00Z",
  effectiveTime: "2029-05-02T00:00:00Z",
};
const AXES = { expiry: "PASS", skew: "PASS" };

function view(context: Record<string, unknown> = CTX): AssessmentView {
  return { docSha256: SHA, axes: { ...AXES }, context: { ...context } };
}

describe("event/effective-time vector", () => {
  it("matches on all comparanda when identical", () => {
    expect(compareAssessments(view(), view())).toEqual(
      Object.fromEntries(COMPARANDA.map((n) => [n, true])),
    );
  });

  it("attributes an event-time flip, never unexplained", () => {
    const other = view({ ...CTX, eventTime: "2029-07-01T00:00:00Z" });
    expect(compareAssessments(view(), other)["eventTime"]).toBe(false);
    const out = classify(view(), other, "verified", "unverified");
    expect(out.label).toBe("attributed_to_inputs");
    expect(out.differing).toEqual(["eventTime"]);
    expect(attributedToTrustTime(out)).toBe(false);
  });

  it("attributes an effective-time flip, never unexplained", () => {
    const other = view({ ...CTX, effectiveTime: "2029-05-03T00:00:00Z" });
    expect(compareAssessments(view(), other)["effectiveTime"]).toBe(false);
    const out = classify(view(), other, "verified", "verified");
    expect(out.label).toBe("agreement");
    expect(out.differing).toEqual(["effectiveTime"]);
    expect(attributedToTrustTime(out)).toBe(false);
  });

  it("reads absent fields equal for old entries", () => {
    const bare = { docSha256: SHA, axes: { ...AXES }, context: { evaluator: "c9t-probe" } };
    expect(compareAssessments(bare, { ...bare })["eventTime"]).toBe(true);
    expect(compareAssessments(bare, { ...bare })["effectiveTime"]).toBe(true);
  });

  it("replays the retained contemporaneous inputs", () => {
    const store = new AssessmentStore();
    store.record(SHA, { ...CTX }, { ...AXES }, "verified", null, "2029-06-01T00:00:00Z");
    const original = store.replay(SHA);
    expect(original?.context["eventTime"]).toBe("2029-05-01T00:00:00Z");
    expect(original?.context["effectiveTime"]).toBe("2029-05-02T00:00:00Z");
    const live = view({ ...CTX, eventTime: "2029-07-01T00:00:00Z" });
    const out = classify(
      { docSha256: SHA, axes: { ...(original?.axes ?? {}) }, context: { ...(original?.context ?? {}) } },
      live,
      "verified",
      "unverified",
    );
    expect(out.label).toBe("attributed_to_inputs");
    expect(out.differing).toEqual(["eventTime"]);
  });
});
