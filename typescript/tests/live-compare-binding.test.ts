/** Digest-to-bytes binding over live compare, TS half (488 C24). */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { compareLive } from "../src/verifier/liveCompare.js";
import { AssessmentStore } from "../src/verifier/assessmentStore.js";
import { recordVerifyResult } from "../src/verifier/retain.js";

const FIX = join(__dirname, "..", "..", "verifier", "signed-conflict-fixtures");
const CTX = { evaluationTime: "2026-10-16T12:00:00Z", evaluator: "c24-probe" };

function raw(name: string): Buffer {
  return readFileSync(join(FIX, `${name}.json`));
}

function digest(b: Buffer): string {
  return createHash("sha256").update(b).digest("hex");
}

function run(b: Buffer, ctx: Record<string, unknown> = CTX) {
  return verify(JSON.parse(b.toString("utf-8")), ADAPTERS, null, null, ctx);
}

describe("live compare binding", () => {
  it("bound bytes agree", () => {
    const allow = raw("conflict-allow");
    const hex = digest(allow);
    const out = compareLive(hex, run(allow), hex, run(allow), null, null, allow, allow);
    expect(out.label).toBe("agreement");
    expect(out.differing).toEqual([]);
  });

  it("bound conflict stays split", () => {
    const allow = raw("conflict-allow");
    const deny = raw("conflict-deny");
    const out = compareLive(digest(allow), run(allow), digest(deny), run(deny), null, null, allow, deny);
    expect(out.label).toBe("different_subject");
    expect(out.differing).toEqual([]);
  });

  it("mismatched digest throws instead of classifying", () => {
    const allow = raw("conflict-allow");
    const deny = raw("conflict-deny");
    const good = digest(allow);
    expect(() => compareLive("00".repeat(32), run(allow), good, run(allow), null, null, allow, allow)).toThrow(
      /first digest/,
    );
    expect(() => compareLive(good, run(allow), good, run(deny), null, null, allow, deny)).toThrow(/second digest/);
  });

  it("compare runs over retained inputs", () => {
    const allow = raw("conflict-allow");
    const deny = raw("conflict-deny");
    const store = new AssessmentStore();
    for (const b of [allow, deny]) recordVerifyResult(store, digest(b), run(b));
    const replayed = [allow, deny].map((b) => {
      const entry = store.replay(digest(b));
      if (entry === null) throw new Error("retained entry missing");
      return run(b, entry.context as Record<string, unknown>);
    });
    expect(replayed.map((r) => r.verdict)).toEqual(["verified", "verified"]);
    const out = compareLive(digest(allow), replayed[0], digest(deny), replayed[1], null, null, allow, deny);
    expect(out.label).toBe("different_subject");
  });
});
