/** Assessment store behaviour, TS half (488 C10). */

import { describe, expect, it } from "vitest";

import { AssessmentStore } from "../src/verifier/assessmentStore.js";

const CTX = { evaluationTime: "2029-06-01T00:00:00Z", evaluator: "c10-probe" };
const AXES = { expiry: "PASS", skew: "PASS" };
const DOC = "aa".repeat(32);

function filled(): AssessmentStore {
  const store = new AssessmentStore();
  store.record(DOC, CTX, AXES, "verified", null, "2029-06-01T00:00:00Z");
  return store;
}

describe("assessment store", () => {
  it("replays the retained original", () => {
    const entry = filled().replay(DOC);
    expect(entry?.kind).toBe("original");
    expect(entry?.context).toEqual(CTX);
    expect(entry?.supersedes).toBeNull();
  });

  it("misses unknown bytes", () => {
    expect(filled().replay("bb".repeat(32))).toBeNull();
  });

  it("labels reassessments without revising", () => {
    const store = filled();
    const before = store.replay(DOC);
    const second = store.reassess(DOC, { ...CTX }, { expiry: "FAIL", skew: "PASS" }, "unverified", "invalid");
    expect(second.kind).toBe("reassessment");
    expect(second.supersedes).toBe(before?.entryHash);
    expect(store.replay(DOC)).toEqual(before);
    expect(store.verifyChain()).toBe(true);
  });

  it("detects mutation", () => {
    const store = filled();
    const tampered = { ...store.replay(DOC)! };
    (tampered as { verdict: string }).verdict = "x";
    (store as unknown as { entries: unknown[] }).entries[0] = tampered;
    expect(store.verifyChain()).toBe(false);
  });

  it("round-trips jsonl and rejects breaks", () => {
    const store = filled();
    store.reassess(DOC, CTX, AXES, "verified", null);
    const restored = AssessmentStore.fromJsonl(store.toJsonl());
    expect(restored.size).toBe(2);
    expect(restored.verifyChain()).toBe(true);
    expect(() => AssessmentStore.fromJsonl(store.toJsonl().replaceAll("verified", "x"))).toThrow();
  });
});
