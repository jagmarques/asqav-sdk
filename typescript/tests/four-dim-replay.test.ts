/** Four-dimension replay with declared per-dim effects, TS half (488 C17). */

import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { AssessmentStore } from "../src/verifier/assessmentStore.js";
import { axesDict, recordVerifyResult } from "../src/verifier/retain.js";
import { classify } from "../src/verifier/disagreement.js";

const T1 = "2019-06-01T00:00:00Z";
const T2 = "2026-09-21T00:00:00Z";
const RECEIPT = join(__dirname, "..", "..", "verifier", "conformance-vectors", "asqav-01-genesis-permit", "receipt.json");

function frozenDoc(issuedAt: string | null = null): { doc: Record<string, unknown>; digest: string } {
  const doc = JSON.parse(readFileSync(RECEIPT, "utf-8")) as Record<string, unknown>;
  (doc["payload"] as Record<string, unknown>)["expires_at"] = "2020-01-01T00:00:00Z";
  if (issuedAt !== null) (doc["payload"] as Record<string, unknown>)["issued_at"] = issuedAt;
  expect(doc["payload"]).toHaveProperty("issued_at");
  return { doc, digest: createHash("sha256").update(JSON.stringify(doc), "utf-8").digest("hex") };
}

function ctx(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    evaluationTime: T1,
    trustRoots: ["root-a"],
    revocationSnapshot: "snap-1",
    policyRevision: "2029-01",
    evidenceKnowledgeCutoff: T1,
    evaluator: "c17-probe",
    ...over,
  };
}

describe("four-dim replay", () => {
  it("declares per-dim effects and replays the baseline", () => {
    const { doc, digest } = frozenDoc();
    const store = new AssessmentStore();
    const baseline = recordVerifyResult(store, digest, verify(doc, ADAPTERS, null, null, ctx()));
    expect(baseline.axes["expiry"]).toBe("PASS");

    const moved = recordVerifyResult(
      store,
      digest,
      verify(doc, ADAPTERS, null, null, ctx({ evaluationTime: T2 })),
    );
    expect(moved.axes["expiry"]).toBe("FAIL");

    for (const [label, over] of [
      ["policyRevision", { policyRevision: "2031-01" }],
      ["cutoff", { evidenceKnowledgeCutoff: T2 }],
      ["trustRoots", { trustRoots: ["root-b"] }],
    ] as const) {
      const same = verify(doc, ADAPTERS, null, null, ctx(over));
      expect(axesDict(same), label).toEqual(baseline.axes);
      expect(same.verdict, label).toBe(baseline.verdict);
    }

    const eventVariant = frozenDoc("2018-01-01T00:00:00Z");
    expect(eventVariant.digest).not.toBe(digest);
    const eventRun = verify(eventVariant.doc, ADAPTERS, null, null, ctx());
    expect(eventRun.axes.find((a) => a.axis === "skew")?.result).toBe("PASS");
    expect(baseline.axes["skew"]).toBe("FAIL");

    const replayed = verify(doc, ADAPTERS, null, null, store.replay(digest)?.context ?? {});
    expect(axesDict(replayed)).toEqual(baseline.axes);
    expect(replayed.verdict).toBe(baseline.verdict);

    const out = classify({ ...baseline }, { ...moved }, baseline.verdict, moved.verdict);
    expect(out.label).toBe("agreement");
    expect(out.differing).toEqual(["contentHashes", "verificationTime"]);
  });
});
