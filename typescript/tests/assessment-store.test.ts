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

  it("hashes non-ascii inputs as utf-8 and round-trips", () => {    const store = new AssessmentStore();
    const entry = store.record(
      "ee".repeat(32),
      { note: "páy ✓", evaluationTime: "2029-06-01T00:00:00Z" },
      { expiry: "PASS" },
      "verified",
      null,
      "2029-06-01T00:00:00Z",
    );
    expect(entry.entryHash).toBe("ee47aa42ad88edbf5fd76dd1cc319336a8a440f4e3f28acf0bbadb4ac636df27");
    expect(store.verifyChain()).toBe(true);
    expect(AssessmentStore.fromJsonl(store.toJsonl()).replay("ee".repeat(32))).toEqual(entry);
  });
});

/** Compromise-driven confidence revision (488 C10), TS half. */
const C10_DOC = "cc".repeat(32);
const C10_ORIG = {
  evaluationTime: "2029-06-01T00:00:00Z",
  evaluator: "c10-probe",
  revocationSnapshot: "snap-1",
  policyRevision: "pol-7",
};
const C10_NEW = {
  evaluationTime: "2031-01-01T00:00:00Z",
  evaluator: "c10-probe",
  revocationSnapshot: "snap-2-key-revoked",
  policyRevision: "pol-9",
};

function c10store(): AssessmentStore {
  const store = new AssessmentStore();
  store.record(C10_DOC, C10_ORIG, { signature: "PASS", keyStatus: "PASS" }, "verified", null);
  return store;
}

describe("compromise revision", () => {
  it("serves current, never the frozen original", () => {
    const store = c10store();
    expect(store.current(C10_DOC)?.kind).toBe("original");
    const snap = store.replay(C10_DOC);
    const second = store.reassess(C10_DOC, C10_NEW, { signature: "PASS", keyStatus: "FAIL" },
      "unverified", "invalid", "2031-01-01T00:00:00Z", "key-compromised");
    expect(second.kind).toBe("reassessment");
    expect(second.reason).toBe("key-compromised");
    expect(store.replay(C10_DOC)).toEqual(snap);
    expect(store.current(C10_DOC)).toEqual(second);
    expect(store.verifyChain()).toBe(true);
  });

  it("records applied inputs on the reassessment", () => {
    const entry = c10store().reassess(C10_DOC, C10_NEW,
      { signature: "PASS", keyStatus: "FAIL" }, "unverified", "invalid", null, "key-compromised");
    expect(entry.context["revocationSnapshot"]).toBe("snap-2-key-revoked");
    expect(entry.context["policyRevision"]).toBe("pol-9");
  });

  it("round-trips reason and loads logs written without it", () => {
    const store = c10store();
    store.reassess(C10_DOC, C10_NEW, { signature: "PASS" }, "unverified", "invalid", null, "policy-updated");
    const restored = AssessmentStore.fromJsonl(store.toJsonl());
    expect(restored.current(C10_DOC)?.reason).toBe("policy-updated");
    expect(restored.verifyChain()).toBe(true);
    const legacy = store.toJsonl().replace(',"reason":"policy-updated"', "");
    const old = AssessmentStore.fromJsonl(legacy);
    expect(old.verifyChain()).toBe(true);
    expect(old.current(C10_DOC)?.reason).toBeNull();
  });

  it("misses unknown bytes and rejects reassessment without original", () => {
    expect(c10store().current("dd".repeat(32))).toBeNull();
    expect(() => new AssessmentStore().reassess(C10_DOC, C10_NEW, {}, "unverified", "invalid"))
      .toThrow();
  });
});
