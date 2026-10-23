// The counterparty hash of float-preserving input equals the hash Python's
// canonicaliser gives, at any depth

import { describe, expect, it } from "vitest";

import { computeEnvelopeHash, verifyCounterpartyBinding } from "../src/counterparty.js";
import { canonicalJson } from "../src/jcs.js";
import { parseJsonPreservingFloats, RawFloat } from "../src/verifier/index.js";
import {
  acknowledgment,
  envelopeText,
  FLOAT_TABLE,
  type FloatCase,
} from "./counterparty-float-cases.js";

const UNRELATED_HASH = Buffer.alloc(32, 7).toString("base64");

function originator(c: FloatCase): Record<string, unknown> {
  return parseJsonPreservingFloats(envelopeText(c)) as Record<string, unknown>;
}

const hashed = FLOAT_TABLE.cases.filter((c) => c.envelope_hash !== null);
const refused = FLOAT_TABLE.cases.filter((c) => c.envelope_hash === null);

describe("computeEnvelopeHash on float-preserving input", () => {
  it("covers decimal, exponent, depth and refused rows, so the loops are not vacuous", () => {
    const deep = (c: FloatCase): boolean => c.wrappers !== undefined;
    expect(hashed.filter(deep).some((c) => /^[0-9.]+$/.test(c.leaf as string))).toBe(true);
    expect(hashed.filter(deep).some((c) => /e/i.test(c.leaf as string))).toBe(true);
    expect(hashed.filter((c) => !deep(c) && /e/i.test(c.name)).length).toBeGreaterThan(5);
    expect(refused.some((c) => c.error?.includes("canonical bytes unavailable"))).toBe(true);
    expect(refused.some((c) => c.error?.includes("nesting exceeds 200 levels"))).toBe(true);
  });

  for (const c of hashed) {
    it(`matches Python: ${c.name}`, () => {
      expect(computeEnvelopeHash(originator(c))).toBe(c.envelope_hash);
    });
  }

  for (const c of refused) {
    it(`refuses as Python does: ${c.name}`, () => {
      expect(() => computeEnvelopeHash(originator(c))).toThrow(c.error as string);
    });
  }

  it("keeps a float apart from the integer of the same value", () => {
    const one = FLOAT_TABLE.cases.find((c) => c.name === "integer one") as FloatCase;
    const decimal = FLOAT_TABLE.cases.find((c) => c.name === "decimal one") as FloatCase;
    expect(computeEnvelopeHash(originator(one))).not.toBe(computeEnvelopeHash(originator(decimal)));
  });

  it("hashes a hand-built RawFloat at the depth cap as a leaf", () => {
    const leafAt = (levels: number): Record<string, unknown> => {
      let node: unknown = new RawFloat(1.5, "1.5");
      for (let i = 0; i < levels; i++) node = { n: node };
      return { payload: { junk: node }, signature: FLOAT_TABLE.signature };
    };
    expect(() => computeEnvelopeHash(leafAt(198))).not.toThrow();
    expect(() => computeEnvelopeHash(leafAt(199))).toThrow("nesting exceeds 200 levels");
  });
});

describe("verifyCounterpartyBinding on float-preserving input", () => {
  for (const c of hashed) {
    it(`matches when the binding carries Python's hash: ${c.name}`, () => {
      const r = verifyCounterpartyBinding(acknowledgment(c.envelope_hash as string), originator(c));
      expect(r.label).toBe("matches");
      expect(r.valid).toBe(true);
    });
  }

  for (const c of refused) {
    it(`reports unresolved where Python cannot hash: ${c.name}`, () => {
      const r = verifyCounterpartyBinding(acknowledgment(UNRELATED_HASH), originator(c));
      expect(r.label).toBe("unresolved");
      expect(r.valid).toBeNull();
    });
  }

  it("reports a mismatch when the decimal leaf was changed", () => {
    const base = FLOAT_TABLE.cases.find((c) => c.name === "decimal 1.5") as FloatCase;
    const other = FLOAT_TABLE.cases.find((c) => c.name === "decimal 0.1") as FloatCase;
    const r = verifyCounterpartyBinding(acknowledgment(base.envelope_hash as string), originator(other));
    expect(r.label).toBe("mismatch");
    expect(r.valid).toBe(false);
  });
});

describe("canonicalJson of a float literal", () => {
  for (const [lexeme, expected] of FLOAT_TABLE.float_reprs) {
    it(`writes ${lexeme} as Python does`, () => {
      const parsed = parseJsonPreservingFloats(`{"v":${lexeme}}`);
      expect(new TextDecoder().decode(canonicalJson(parsed))).toBe(expected);
    });
  }
});
