/** Strict live compare, TS half (C26; C29 binds envelope from the bytes). */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { envelopeOf } from "../src/verifier/agreement.js";
import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { compareLive, compareLiveStrict } from "../src/verifier/liveCompare.js";

const FIX = join(__dirname, "..", "..", "verifier", "signed-conflict-fixtures");
const CTX = { evaluationTime: "2026-10-16T12:00:00Z", evaluator: "c26-probe" };

function raw(name: string): Buffer {
  return readFileSync(join(FIX, `${name}.json`));
}

function digest(b: Buffer): string {
  return createHash("sha256").update(b).digest("hex");
}

function run(b: Buffer, ctx: Record<string, unknown> = CTX) {
  return verify(JSON.parse(b.toString("utf-8")), ADAPTERS, null, null, ctx);
}

describe("live compare strict", () => {
  it("agrees on the same bytes", () => {
    const allow = raw("conflict-allow");
    const out = compareLiveStrict(allow, allow, null, null, CTX);
    expect(out.label).toBe("agreement");
    expect(out.differing).toEqual([]);
  });

  it("strict conflict stays split", () => {
    const allow = raw("conflict-allow");
    const deny = raw("conflict-deny");
    const out = compareLiveStrict(allow, deny, null, null, CTX);
    expect(out.label).toBe("different_subject");
  });

  it("matches the live pipeline", () => {
    const allow = raw("conflict-allow");
    const deny = raw("conflict-deny");
    const strict = compareLiveStrict(allow, deny, null, null, CTX);
    const piped = compareLive(digest(allow), run(allow), digest(deny), run(deny), null, null, allow, deny);
    expect([strict.label, strict.differing]).toEqual([piped.label, piped.differing]);
  });

  it("a forged extra result cannot steer the entry", () => {
    const allow = raw("conflict-allow");
    const deny = raw("conflict-deny");
    expect(compareLiveStrict.length).toBe(2);
    const forged = { ...run(allow), verdict: "unverified" };
    const plain = compareLiveStrict(allow, deny, null, null, CTX);
    const withForged = (compareLiveStrict as (...args: unknown[]) => unknown)(
      allow, deny, null, null, CTX, forged,
    );
    expect(withForged).toEqual(plain);
  });

  it("derives the envelope from the verified bytes", () => {
    const allow = raw("conflict-allow");
    const out = compareLiveStrict(allow, allow, null, null, CTX);
    expect(out.label).toBe("agreement");
    expect(out.differing).toEqual([]);
  });

  it("throws on a forged stated envelope", () => {
    const allow = raw("conflict-allow");
    expect(() =>
      compareLiveStrict(allow, allow, { kid: "attacker-key", signature: "00".repeat(64) }, null, CTX),
    ).toThrow();
    const env = envelopeOf(JSON.parse(allow.toString("utf-8")), ADAPTERS);
    const honest = compareLiveStrict(allow, allow, { kid: env.kid, signature: env.signature }, null, CTX);
    expect(honest.label).toBe("agreement");
    expect(honest.differing).toEqual([]);
  });

  it("keeps the conflict label past envelope dims", () => {
    const allow = raw("conflict-allow");
    const deny = raw("conflict-deny");
    const out = compareLiveStrict(allow, deny, null, null, CTX);
    expect(out.label).toBe("different_subject");
    expect(out.differing).toEqual([]);
  });
});
