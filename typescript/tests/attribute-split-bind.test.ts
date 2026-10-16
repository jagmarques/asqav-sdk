/** Envelope binding in attributeLiveSplit (488 C31, TS half). */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { attributeLiveSplit } from "../src/verifier/liveCompare.js";
import { envelopeOf } from "../src/verifier/agreement.js";

const FIX = join(__dirname, "..", "..", "verifier", "signed-conflict-fixtures");
const CTX = { evaluationTime: "2026-10-16T12:00:00Z", evaluator: "c31-probe" };

function raw(name: string): Buffer {
  return readFileSync(join(FIX, `${name}.json`));
}

function digest(b: Buffer): string {
  return createHash("sha256").update(b).digest("hex");
}

function run(b: Buffer, ctx: Record<string, unknown> = CTX) {
  return verify(JSON.parse(b.toString("utf-8")), ADAPTERS, null, null, ctx);
}

describe("attribute split bind", () => {
  it("stated envelope without bytes throws", () => {
    const allow = raw("conflict-allow");
    const hex = digest(allow);
    const res = run(allow);
    const env = envelopeOf(JSON.parse(allow.toString("utf-8")));
    const honest = { kid: env.kid, signature: env.signature };
    expect(() => attributeLiveSplit(hex, res, hex, res, null, null, honest, null)).toThrow(/first envelope without bytes/);
    expect(() => attributeLiveSplit(hex, res, hex, res, res, null, honest, null, allow, allow)).toThrow(/rerunA envelope without bytes/);
  });

  it("bound rerun bytes bind and contradiction throws", () => {
    const allow = raw("conflict-allow");
    const hex = digest(allow);
    const res = run(allow);
    const env = envelopeOf(JSON.parse(allow.toString("utf-8")));
    const honest = { kid: env.kid, signature: env.signature };
    const out = attributeLiveSplit(hex, res, hex, res, res, res, honest, honest, allow, allow, allow, allow);
    expect(out.disagreement.label).toBe("agreement");
    const forged = { kid: "attacker-key", signature: "00".repeat(64) };
    expect(() => attributeLiveSplit(hex, res, hex, res, null, null, forged, null, allow, allow)).toThrow(/contradicts the verified envelope/);
  });
});
