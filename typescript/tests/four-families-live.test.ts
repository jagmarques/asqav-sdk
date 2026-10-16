/** Four disagreement families through the live oracle, TS half (488 C22). */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { compareLive } from "../src/verifier/liveCompare.js";

const FIX = join(__dirname, "..", "..", "verifier", "signed-conflict-fixtures");
const T1 = "2026-10-16T12:00:00Z";
const T2 = "2031-01-01T00:00:00Z";

function load(name: string): { doc: Record<string, unknown>; digest: string } {
  const raw = readFileSync(join(FIX, `${name}.json`));
  return { doc: JSON.parse(raw.toString("utf-8")), digest: createHash("sha256").update(raw).digest("hex") };
}

function run(doc: Record<string, unknown>, evaluationTime: string = T1) {
  return verify(doc, ADAPTERS, null, null, {
    evaluationTime,
    trustRoots: ["root-a"],
    revocationSnapshot: "snap-1",
    evaluator: "c22-probe",
  });
}

function failingAxes(res: { axes: { axis: string; result: string }[] }): string[] {
  return res.axes.filter((a) => a.result !== "PASS").map((a) => a.axis);
}

describe("four families live", () => {
  it("success agrees", () => {
    const { doc, digest } = load("conflict-allow");
    const out = compareLive(digest, run(doc), digest, run(doc));
    expect(out.label).toBe("agreement");
    expect(out.differing).toEqual([]);
  });

  it("contradictory signed terms stay split", () => {
    const allow = load("conflict-allow");
    const deny = load("conflict-deny");
    expect(allow.digest).not.toBe(deny.digest);
    const ra = run(allow.doc);
    const rd = run(deny.doc);
    expect([ra.verdict, rd.verdict]).toEqual(["verified", "verified"]);
    const out = compareLive(allow.digest, ra, deny.digest, rd);
    expect(out.label).toBe("different_subject");
    expect(out.differing).toEqual([]);
  });

  it("dropped evidence fails only the signature", () => {
    const { doc, digest } = load("conflict-allow");
    const dropped = JSON.parse(JSON.stringify(doc)) as Record<string, unknown>;
    delete (dropped["credentialSubject"] as Record<string, unknown>)["outcome"];
    const raw = Buffer.from(JSON.stringify(dropped));
    const missing = run(JSON.parse(raw.toString("utf-8")));
    expect(missing.verdict).toBe("unverified");
    expect(failingAxes(missing)).toEqual(["signature"]);
    const out = compareLive(digest, run(doc), createHash("sha256").update(raw).digest("hex"), missing);
    expect(out.label).toBe("different_subject");
    expect(out.differing).toEqual([]);
  });

  it("trust/time drift agrees with the drift recorded", () => {
    const { doc, digest } = load("conflict-allow");
    const early = run(doc, T1);
    const late = run(doc, T2);
    expect([early.verdict, late.verdict]).toEqual(["verified", "verified"]);
    const out = compareLive(digest, early, digest, late);
    expect(out.label).toBe("agreement");
    expect(out.differing).toEqual(["verificationTime"]);
  });
});
