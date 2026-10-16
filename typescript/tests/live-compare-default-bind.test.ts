/** Default envelope binding in compareLive (488 C30, TS half). */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { compareLive } from "../src/verifier/liveCompare.js";
import { agreementOfTerms, envelopeOf } from "../src/verifier/agreement.js";

const FIX = join(__dirname, "..", "..", "verifier", "signed-conflict-fixtures");
const CTX = { evaluationTime: "2026-10-16T12:00:00Z", evaluator: "c30-probe" };

function raw(name: string): Buffer {
  return readFileSync(join(FIX, `${name}.json`));
}

function digest(b: Buffer): string {
  return createHash("sha256").update(b).digest("hex");
}

function run(b: Buffer, ctx: Record<string, unknown> = CTX) {
  return verify(JSON.parse(b.toString("utf-8")), ADAPTERS, null, null, ctx);
}

describe("live compare default bind", () => {
  it("stated envelope without bytes throws", () => {
    const allow = raw("conflict-allow");
    const hex = digest(allow);
    const res = run(allow);
    const env = envelopeOf(JSON.parse(allow.toString("utf-8")));
    const honest = { kid: env.kid, signature: env.signature };
    expect(() => compareLive(hex, res, hex, res, honest, null)).toThrow(/first envelope without bytes/);
    expect(() => compareLive(hex, res, hex, res, null, honest)).toThrow(/second envelope without bytes/);
    expect(() => compareLive(hex, res, hex, res, { kid: env.kid }, null)).toThrow(/first envelope without bytes/);
  });

  it("bound envelope matches and contradiction throws", () => {
    const allow = raw("conflict-allow");
    const hex = digest(allow);
    const res = run(allow);
    const env = envelopeOf(JSON.parse(allow.toString("utf-8")));
    const honest = { kid: env.kid, signature: env.signature };
    const out = compareLive(hex, res, hex, res, honest, honest, allow, allow);
    expect(out.label).toBe("agreement");
    const forged = { kid: "attacker-key", signature: "00".repeat(64) };
    expect(() => compareLive(hex, res, hex, res, forged, null, allow, allow)).toThrow(/contradicts the verified envelope/);
  });

  it("terms only without bytes still classifies", () => {
    const allow = raw("conflict-allow");
    const hex = digest(allow);
    const res = run(allow);
    const terms = agreementOfTerms("agree-1", "v1", { scope: "c30" });
    const out = compareLive(hex, res, hex, res, terms, terms);
    expect(out.label).toBe("agreement");
  });
});
