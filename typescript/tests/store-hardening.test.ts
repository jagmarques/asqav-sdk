/** Store hardening, TS half (488 C18): dup guard, orphan raise, link checks. */

import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import { AssessmentStore } from "../src/verifier/assessmentStore.js";

const CTX = { evaluationTime: "2029-06-01T00:00:00Z", evaluator: "c18-probe" };
const AXES = { expiry: "PASS", skew: "PASS" };
const SHA = "cc".repeat(32);

// Mirrors assessmentStore.ts canonical bytes to craft chain-valid fixtures
function rehash(mutated: Record<string, unknown>): string {
  const { entryHash: _drop, ...body } = mutated;
  return createHash("sha256").update(JSON.stringify(sortDeep(body)), "utf-8").digest("hex");
}

function sortDeep(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortDeep);
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(value).sort()) {
      out[k] = sortDeep((value as Record<string, unknown>)[k]);
    }
    return out;
  }
  return value;
}

function relinked(lines: string[], index: number, over: Record<string, unknown>): string {
  const raw = { ...JSON.parse(lines[index]), ...over };
  return JSON.stringify({ ...raw, entryHash: rehash(raw) });
}

function recorded(): AssessmentStore {
  const store = new AssessmentStore();
  store.record(SHA, CTX, AXES, "verified", null);
  return store;
}

describe("store hardening", () => {
  it("rejects a duplicate original and keeps the first", () => {
    const store = recorded();
    expect(() => store.record(SHA, CTX, { expiry: "FAIL" }, "unverified", "invalid")).toThrow();
    expect(store.size).toBe(1);
    expect(store.replay(SHA)?.axes).toEqual(AXES);
    expect(store.verifyChain()).toBe(true);
  });

  it("rejects an orphan reassessment and appends nothing", () => {
    const store = new AssessmentStore();
    expect(() => store.reassess(SHA, CTX, AXES, "verified", null)).toThrow();
    expect(store.size).toBe(0);
    expect(store.replay(SHA)).toBeNull();
  });

  it("rejects dangling and foreign links in a restored log", () => {
    const store = recorded();
    const original = store.replay(SHA);
    const good = store.reassess(SHA, CTX, { expiry: "FAIL" }, "unverified", "invalid");
    const lines = store.toJsonl().split("\n");
    expect(() =>
      AssessmentStore.fromJsonl([lines[0], relinked(lines, 1, { supersedes: "ff".repeat(32) })].join("\n")),
    ).toThrow();
    const foreign = {
      docSha256: "dd".repeat(32),
      context: { ...CTX },
      axes: { ...AXES },
      verdict: "unverified",
      failureClass: "invalid",
      recordedAt: null,
      kind: "reassessment",
      supersedes: original!.entryHash,
      prevHash: good.entryHash,
    };
    const blob = [...lines, JSON.stringify({ ...foreign, entryHash: rehash(foreign) })].join("\n");
    expect(() => AssessmentStore.fromJsonl(blob)).toThrow();
  });

  it("rejects bad kinds and superseding originals in a restored log", () => {
    const store = recorded();
    const lines = store.toJsonl().split("\n");
    expect(() => AssessmentStore.fromJsonl(relinked(lines, 0, { kind: "snapshot" }))).toThrow();
    expect(() => AssessmentStore.fromJsonl(relinked(lines, 0, { supersedes: "ff".repeat(32) }))).toThrow();
    const restored = AssessmentStore.fromJsonl(store.toJsonl());
    expect(restored.size).toBe(1);
    expect(restored.verifyChain()).toBe(true);
  });
});
