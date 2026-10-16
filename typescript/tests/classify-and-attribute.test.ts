/** Reassessment review, TS half (488 C27): the caller, never the internals. */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { AssessmentStore } from "../src/verifier/assessmentStore.js";
import { classifyAndAttribute } from "../src/verifier/liveCompare.js";
import { recordVerifyResult } from "../src/verifier/retain.js";

const PINNED = "2026-10-16T00:00:00Z";
const T1 = "2019-06-01T00:00:00Z";
const T2 = "2026-09-21T00:00:00Z";
const FIX = join(__dirname, "..", "..", "verifier", "signed-conflict-fixtures");
const RECEIPT = join(__dirname, "..", "..", "verifier", "conformance-vectors", "asqav-01-genesis-permit", "receipt.json");

function load(name: string): { raw: Buffer; digest: string; doc: Record<string, unknown> } {
  const raw = readFileSync(join(FIX, `${name}.json`));
  return { raw, digest: createHash("sha256").update(raw).digest("hex"), doc: JSON.parse(raw.toString("utf-8")) };
}

function run(doc: Record<string, unknown>) {
  return verify(doc, ADAPTERS, null, null, { evaluationTime: PINNED, evaluator: "c27-probe" });
}

function expiring(): { doc: Record<string, unknown>; digest: string } {
  const doc = JSON.parse(readFileSync(RECEIPT, "utf-8")) as Record<string, unknown>;
  (doc["payload"] as Record<string, unknown>)["expires_at"] = "2020-01-01T00:00:00Z";
  return { doc, digest: createHash("sha256").update(JSON.stringify(doc), "utf-8").digest("hex") };
}

describe("classify and attribute", () => {
  it("retained then live agrees without mutating the store", () => {
    const { digest, doc } = load("conflict-allow");
    const store = new AssessmentStore();
    recordVerifyResult(store, digest, run(doc));
    const out = classifyAndAttribute(store, digest, run(doc), run(doc), run(doc));
    expect(out.disagreement.label).toBe("agreement");
    expect(out.disagreement.differing).toEqual([]);
    expect(out.specCase.label).toBe("not_applicable");
    expect(store.size).toBe(1);
  });

  it("missing original throws instead of classifying", () => {
    const { digest, doc } = load("conflict-allow");
    expect(() => classifyAndAttribute(new AssessmentStore(), digest, run(doc))).toThrow(/no retained original/);
  });

  it("reassessment split attributes through the caller", () => {
    const { digest, doc } = load("conflict-allow");
    const store = new AssessmentStore();
    recordVerifyResult(store, digest, run(doc));
    const split = { ...run(doc), verdict: "unverified" };
    const out = classifyAndAttribute(store, digest, split, run(doc), run(doc));
    expect(out.disagreement.label).toBe("unexplained_needs_investigation");
    expect(out.specCase.label).toBe("spec_bug_candidate");
    expect(out.specCase.pinned).toEqual([digest]);
  });

  it("moved rerun reports flake through the caller", () => {
    const { doc, digest } = expiring();
    const era = (t: string) => verify(doc, ADAPTERS, null, null, { evaluationTime: t });
    const store = new AssessmentStore();
    recordVerifyResult(store, digest, era(T1));
    const first = era(T1);
    const flipped = first.verdict === "verified" ? "unverified" : "verified";
    const split = { ...era(T1), verdict: flipped };
    const out = classifyAndAttribute(store, digest, split, era(T1), era(T2));
    expect(out.disagreement.label).toBe("unexplained_needs_investigation");
    expect(out.specCase.label).toBe("nondeterministic_flake");
    expect(out.specCase.note).toContain("second");
  });

  it("honest drift surfaces through the retained original", () => {
    const { doc, digest } = expiring();
    const era = (t: string) => verify(doc, ADAPTERS, null, null, { evaluationTime: t });
    const store = new AssessmentStore();
    recordVerifyResult(store, digest, era(T1));
    const out = classifyAndAttribute(store, digest, era(T2), era(T2), era(T2));
    expect(out.disagreement.label).toBe("agreement");
    expect(out.disagreement.differing).toContain("verificationTime");
    expect(out.specCase.label).toBe("not_applicable");
  });
});
