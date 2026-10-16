/** Per-side trust contexts in the strict entry (488 C33, TS half). */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { compareLiveStrict } from "../src/verifier/liveCompare.js";

const FIX = join(__dirname, "..", "..", "verifier", "signed-conflict-fixtures");
const CTX = { evaluationTime: "2026-10-16T12:00:00Z", evaluator: "c33-probe" };

function raw(name: string): Buffer {
  return readFileSync(join(FIX, `${name}.json`));
}

describe("strict per-side trust", () => {
  it("shared context unchanged", () => {
    const allow = raw("conflict-allow");
    const out = compareLiveStrict(allow, allow, null, null, CTX);
    expect(out.label).toBe("agreement");
    expect(out.differing).toEqual([]);
  });

  it("per-side trust roots surface in differing", () => {
    const allow = raw("conflict-allow");
    const out = compareLiveStrict(allow, allow, null, null, CTX,
      { ...CTX, trustRoots: ["root-a"] }, { ...CTX, trustRoots: ["root-b"] });
    expect(out.differing).toContain("trustRoots");
  });

  it("omitted side falls back to shared context", () => {
    const allow = raw("conflict-allow");
    const out = compareLiveStrict(allow, allow, null, null, CTX,
      { ...CTX, trustRoots: ["root-a"] });
    expect(out.differing).toContain("trustRoots");
  });
});
