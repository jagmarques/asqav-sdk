/** Live compare-before-classify over real oracle output, TS half (488 C19). */

import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { agreementOfTerms } from "../src/verifier/agreement.js";
import { attributedToTrustTime } from "../src/verifier/disagreement.js";
import { compareLive } from "../src/verifier/liveCompare.js";

const T1 = "2019-06-01T00:00:00Z";
const T2 = "2026-09-21T00:00:00Z";
const RECEIPT = join(__dirname, "..", "..", "verifier", "conformance-vectors", "asqav-01-genesis-permit", "receipt.json");

function expiringDoc(): { doc: Record<string, unknown>; digest: string } {
  const doc = JSON.parse(readFileSync(RECEIPT, "utf-8")) as Record<string, unknown>;
  (doc["payload"] as Record<string, unknown>)["expires_at"] = "2020-01-01T00:00:00Z";
  const digest = createHash("sha256").update(JSON.stringify(doc), "utf-8").digest("hex");
  return { doc, digest };
}

function run(doc: Record<string, unknown>, evaluationTime: string) {
  return verify(doc, ADAPTERS, null, null, {
    evaluationTime,
    trustRoots: ["root-a"],
    revocationSnapshot: "snap-1",
    evaluator: "c19-probe",
  });
}

describe("live compare", () => {
  it("classifies two-era live runs as agreement with drift", () => {
    const { doc, digest } = expiringDoc();
    const out = compareLive(digest, run(doc, T1), digest, run(doc, T2));
    expect(out.label).toBe("agreement");
    expect(out.differing).toContain("verificationTime");
    expect(out.differing).toContain("contentHashes");
    expect(attributedToTrustTime(out)).toBe(false);
  });

  it("fully agrees on same-era live runs", () => {
    const { doc, digest } = expiringDoc();
    const out = compareLive(digest, run(doc, T1), digest, run(doc, T1));
    expect(out.label).toBe("agreement");
    expect(out.differing).toEqual([]);
  });

  it("treats different input bytes as a different subject", () => {
    const { doc, digest } = expiringDoc();
    const other = JSON.parse(JSON.stringify(doc)) as Record<string, unknown>;
    (other["payload"] as Record<string, unknown>)["expires_at"] = "2021-01-01T00:00:00Z";
    const otherDigest = createHash("sha256").update(JSON.stringify(other), "utf-8").digest("hex");
    const out = compareLive(digest, run(doc, T1), otherDigest, run(other, T1));
    expect(out.label).toBe("different_subject");
  });

  it("binds predicate terms by id, version and hash", () => {
    const { doc, digest } = expiringDoc();
    const terms = agreementOfTerms("pred-1", "2026-01", doc["payload"]);
    const out = compareLive(digest, run(doc, T1), digest, run(doc, T1), terms, terms);
    expect(out.label).toBe("agreement");
    expect(out.differing).not.toContain("agreementVersion");
    const moved = agreementOfTerms("pred-1", "2026-02", doc["payload"]);
    const out2 = compareLive(digest, run(doc, T1), digest, run(doc, T1), terms, moved);
    expect(out2.label).toBe("agreement");
    expect(out2.differing).toContain("agreementVersion");
  });
});
