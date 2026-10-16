/** Wire specCase into the live path, TS half (488 C25). */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import type { AssessmentView } from "../src/verifier/agreement.js";
import { axesDict, contextDict } from "../src/verifier/retain.js";
import { attributeLiveSplit, attributeSplit } from "../src/verifier/liveCompare.js";

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
  return verify(doc, ADAPTERS, null, null, { evaluationTime: PINNED, evaluator: "c25-probe" });
}

function viewOf(res: ReturnType<typeof run>, digest: string): AssessmentView {
  return { docSha256: digest, axes: axesDict(res), context: contextDict(res), agreement: null };
}

function expiring(): { doc: Record<string, unknown>; digest: string } {
  const doc = JSON.parse(readFileSync(RECEIPT, "utf-8")) as Record<string, unknown>;
  (doc["payload"] as Record<string, unknown>)["expires_at"] = "2020-01-01T00:00:00Z";
  return { doc, digest: createHash("sha256").update(JSON.stringify(doc), "utf-8").digest("hex") };
}

describe("spec case wiring", () => {
  it("live agreement needs no attribution", () => {
    const { raw, digest, doc } = load("conflict-allow");
    expect(run(doc).verdict).toBe("verified");
    const out = attributeLiveSplit(digest, run(doc), digest, run(doc), run(doc), run(doc), null, null, raw);
    expect(out.disagreement.label).toBe("agreement");
    expect(out.disagreement.differing).toEqual([]);
    expect(out.specCase.label).toBe("not_applicable");
  });

  it("live drift pair needs no attribution", () => {
    const { doc, digest } = expiring();
    const era = (t: string) => verify(doc, ADAPTERS, null, null, {
      evaluationTime: t, trustRoots: ["root-a"], revocationSnapshot: "snap-1", evaluator: "c19-probe",
    });
    const out = attributeLiveSplit(digest, era(T1), digest, era(T2), era(T1), era(T2));
    expect(out.disagreement.label).toBe("agreement");
    expect(out.disagreement.differing).toContain("verificationTime");
    expect(out.specCase.label).toBe("not_applicable");
  });

  it("live conflict stays split without fusion", () => {
    const allow = load("conflict-allow");
    const deny = load("conflict-deny");
    const out = attributeLiveSplit(
      allow.digest, run(allow.doc), deny.digest, run(deny.doc), run(allow.doc), run(deny.doc));
    expect(out.disagreement.label).toBe("different_subject");
    expect(out.disagreement.differing).toEqual([]);
    expect(out.specCase.label).toBe("not_applicable");
  });

  it("live rerun bytes must match", () => {
    const { digest, doc } = load("conflict-allow");
    expect(() => attributeLiveSplit(digest, run(doc), digest, run(doc), run(doc), run(doc),
      null, null, null, null, Buffer.from("not the receipt"))).toThrow("rerunA digest");
  });

  it("candidate pins the live digest through the pipeline", () => {
    const { digest, doc } = load("conflict-allow");
    const first = run(doc);
    const second = run(doc);
    expect(viewOf(first, digest)).toEqual(viewOf(second, digest));
    expect(Object.keys(first.axes).length).toBeGreaterThan(0);
    const out = attributeSplit(viewOf(first, digest), viewOf(second, digest), first.verdict, "unverified",
      viewOf(run(doc), digest), viewOf(run(doc), digest));
    expect(out.disagreement.label).toBe("unexplained_needs_investigation");
    expect(out.specCase.label).toBe("spec_bug_candidate");
    expect(out.specCase.pinned).toEqual([digest]);
  });

  it("pipeline without the split stays not applicable", () => {
    const { digest, doc } = load("conflict-allow");
    const first = run(doc);
    const second = run(doc);
    const out = attributeSplit(viewOf(first, digest), viewOf(second, digest), first.verdict, second.verdict,
      viewOf(run(doc), digest), viewOf(run(doc), digest));
    expect(out.disagreement.label).toBe("agreement");
    expect(out.specCase.label).toBe("not_applicable");
  });

  it("pipeline with moved rerun axes reports flake", () => {
    const { digest, doc } = load("conflict-allow");
    const first = run(doc);
    const second = run(doc);
    const rerun = viewOf(run(doc), digest);
    const moved: AssessmentView = { ...rerun, axes: { ...rerun.axes, expiry: "FAIL" } };
    const out = attributeSplit(viewOf(first, digest), viewOf(second, digest), first.verdict, "unverified",
      viewOf(run(doc), digest), moved);
    expect(out.disagreement.label).toBe("unexplained_needs_investigation");
    expect(out.specCase.label).toBe("nondeterministic_flake");
  });
});
