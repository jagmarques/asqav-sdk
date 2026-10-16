/** Retain live oracle results, TS half (488 C16). */

import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { AssessmentStore } from "../src/verifier/assessmentStore.js";
import { axesDict, contextDict, recordVerifyResult } from "../src/verifier/retain.js";
import { classify } from "../src/verifier/disagreement.js";

const RECEIPT = join(__dirname, "..", "..", "verifier", "conformance-vectors", "asqav-01-genesis-permit", "receipt.json");

function expiringDoc(): { doc: Record<string, unknown>; digest: string } {
  const doc = JSON.parse(readFileSync(RECEIPT, "utf-8")) as Record<string, unknown>;
  (doc["payload"] as Record<string, unknown>)["expires_at"] = "2020-01-01T00:00:00Z";
  const digest = createHash("sha256").update(JSON.stringify(doc), "utf-8").digest("hex");
  return { doc, digest };
}

describe("retain live results", () => {
  it("freezes Date contexts to ISO strings with distinct hashes", () => {
    const a = recordVerifyResult(new AssessmentStore(), "dd".repeat(32), {
      axes: [{ axis: "expiry", result: "PASS" }],
      verdict: "verified",
      failureClass: null,
      context: { evaluationTime: new Date("2019-06-01T00:00:00Z") },
    });
    const b = recordVerifyResult(new AssessmentStore(), "dd".repeat(32), {
      axes: [{ axis: "expiry", result: "PASS" }],
      verdict: "verified",
      failureClass: null,
      context: { evaluationTime: new Date("2025-01-01T00:00:00Z") },
    });
    expect(a.context["evaluationTime"]).toBe("2019-06-01T00:00:00.000Z");
    expect(b.context["evaluationTime"]).toBe("2025-01-01T00:00:00.000Z");
    expect(a.entryHash).not.toBe(b.entryHash);
  });

  it("maps attribution fields from a live result", () => {
    const { doc, digest } = expiringDoc();
    const res = verify(doc, ADAPTERS, null, null, {
      evaluationTime: "2019-06-01T00:00:00Z",
      evaluator: "c16-probe",
      trustRoots: ["root-a"],
      revocationSnapshot: "snap-1",
    });
    expect(res.axes.find((a) => a.axis === "expiry")?.result).toBe("PASS");
    const ctx = contextDict(res);
    expect(ctx["evaluator"]).toBe("c16-probe");
    expect(ctx["evaluationTime"]).toBe("2019-06-01T00:00:00Z");
    const axes = axesDict(res);
    expect(axes["expiry"]).toBe("PASS");
    expect(Object.keys(axes)).toHaveLength(res.axes.length);
    const entry = recordVerifyResult(new AssessmentStore(), digest, res);
    expect(entry.docSha256).toBe(digest);
    expect(entry.verdict).toBe(res.verdict);
  });

  it("retains and replays a live two-era run", () => {
    const { doc, digest } = expiringDoc();
    const store = new AssessmentStore();
    const first = recordVerifyResult(
      store,
      digest,
      verify(doc, ADAPTERS, null, null, { evaluationTime: "2019-06-01T00:00:00Z" }),
    );
    const second = recordVerifyResult(
      store,
      digest,
      verify(doc, ADAPTERS, null, null, { evaluationTime: "2026-09-21T00:00:00Z" }),
    );
    expect(first.axes["expiry"]).toBe("PASS");
    expect(second.axes["expiry"]).toBe("FAIL");
    expect(first.verdict).toBe(second.verdict);
    expect(store.replay(digest)).toEqual(first);
    const out = classify({ ...first }, { ...second }, first.verdict, second.verdict);
    expect(out.label).toBe("agreement");
    expect(out.differing).toContain("verificationTime");
    expect(out.differing).toContain("contentHashes");
  });
});
