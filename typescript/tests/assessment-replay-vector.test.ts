/** Identical-facts replay vector, TS half (488 C11). */

import { describe, expect, it } from "vitest";

import { AssessmentStore } from "../src/verifier/assessmentStore.js";

const DOC = "cc".repeat(32);
const T1 = "2029-06-01T00:00:00Z";
const T2 = "2031-01-01T00:00:00Z";
const CTX_T1 = {
  policyRevision: "2029-01",
  eventTime: T1,
  evaluationTime: T1,
  evidenceKnowledgeCutoff: T1,
};
const CTX_T2 = {
  policyRevision: "2031-01",
  eventTime: T1,
  evaluationTime: T2,
  evidenceKnowledgeCutoff: T2,
};
const AXES_OK = { expiry: "PASS", skew: "PASS" };
const AXES_BAD = { expiry: "FAIL", skew: "PASS" };

function twoEras(): AssessmentStore {
  const store = new AssessmentStore();
  store.record(DOC, CTX_T1, AXES_OK, "verified", null, T1);
  store.reassess(DOC, CTX_T2, AXES_BAD, "unverified", "invalid", T2);
  return store;
}

describe("identical-facts replay vector", () => {
  it("replays contemporaneous inputs, not current policy", () => {
    const replayed = twoEras().replay(DOC);
    expect(replayed?.kind).toBe("original");
    expect(replayed?.context).toEqual(CTX_T1);
    expect(replayed?.verdict).toBe("verified");
  });

  it("labels the later reassessment and links it", () => {
    const store = twoEras();
    const original = store.replay(DOC);
    const second = (store as unknown as { entries: { kind: string; supersedes: string | null }[] }).entries[1];
    expect(second.kind).toBe("reassessment");
    expect(second.supersedes).toBe(original?.entryHash);
    expect(store.verifyChain()).toBe(true);
  });
});
