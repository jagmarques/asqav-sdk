/** Conflicting signed terms through the real oracle, TS half (488 C21). */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { compareLive } from "../src/verifier/liveCompare.js";

const FIX = join(__dirname, "..", "..", "verifier", "signed-conflict-fixtures");

function load(name: string): { doc: Record<string, unknown>; digest: string } {
  const raw = readFileSync(join(FIX, `${name}.json`));
  return { doc: JSON.parse(raw.toString("utf-8")), digest: createHash("sha256").update(raw).digest("hex") };
}

function signatureOf(result: { axes: { axis: string; result: string }[] }): string | undefined {
  return result.axes.find((a) => a.axis === "signature")?.result;
}

describe("signed conflict", () => {
  it("verifies both conflicting terms with valid signatures", () => {
    for (const name of ["conflict-allow", "conflict-deny"]) {
      const { doc } = load(name);
      const res = verify(doc, ADAPTERS, null, null, {});
      expect(signatureOf(res), name).toBe("PASS");
      expect(res.verdict, name).toBe("verified");
    }
  });

  it("preserves the conflict instead of fusing agreement", () => {
    const allow = load("conflict-allow");
    const deny = load("conflict-deny");
    expect(allow.digest).not.toBe(deny.digest);
    const out = compareLive(
      allow.digest,
      verify(allow.doc, ADAPTERS, null, null, {}),
      deny.digest,
      verify(deny.doc, ADAPTERS, null, null, {}),
    );
    expect(out.label).toBe("different_subject");
    expect(out.differing).toEqual([]);
  });

  it("fails the signature on a tampered copy", () => {
    const { doc } = load("conflict-allow");
    const tampered = JSON.parse(JSON.stringify(doc)) as Record<string, unknown>;
    ((tampered["credentialSubject"] as Record<string, unknown>)["outcome"] as Record<string, unknown>)["status"] =
      "success!";
    expect(signatureOf(verify(tampered, ADAPTERS, null, null, {}))).toBe("FAIL");
  });

  it("fails the signature when evidence is dropped", () => {
    const { doc } = load("conflict-allow");
    const dropped = JSON.parse(JSON.stringify(doc)) as Record<string, unknown>;
    delete (dropped["credentialSubject"] as Record<string, unknown>)["outcome"];
    const res = verify(dropped, ADAPTERS, null, null, {});
    expect(signatureOf(res)).toBe("FAIL");
    expect(res.axes.find((a) => a.axis === "structure")?.result).toBe("PASS");
  });
});
