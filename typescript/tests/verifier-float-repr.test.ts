// Float spelling parity with Python repr (criterion 747), table in verifier/
// float-repr-cases.json

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  asqavJcs,
  counterpartyEnvelopeHash,
  jcs,
  parseJsonPreservingFloats,
  RawFloat,
} from "../src/verifier/index.js";
import { computeEnvelopeHash, verifyCounterpartyBinding } from "../src/index.js";

interface FloatCase {
  zone: string;
  lexeme: string;
  repr: string;
  canonical: string;
  envelope_hash: string;
  binding: string;
}

const TABLE = JSON.parse(
  readFileSync(resolve(__dirname, "..", "..", "verifier", "float-repr-cases.json"), "utf8"),
) as { signature: Record<string, string>; cases: FloatCase[] };
const decode = (b: Uint8Array): string => new TextDecoder().decode(b);

describe("float spelling matches Python repr", () => {
  it("the table covers every split zone", () => {
    const count = (zone: string) => TABLE.cases.filter((c) => c.zone === zone).length;
    for (const zone of ["small", "large", "mid", "bits", "edge"]) expect(count(zone)).toBeGreaterThanOrEqual(40);
    expect(TABLE.cases.some((c) => c.repr === "-0.0")).toBe(true);
  });

  it.each(["asqavJcs", "jcs"] as const)("%s writes each decimal as Python does", (name) => {
    const write = name === "asqavJcs" ? asqavJcs : jcs;
    const wrong: string[] = [];
    for (const c of TABLE.cases) {
      const got = decode(write(parseJsonPreservingFloats(`{"v":${c.lexeme}}`)));
      if (got !== c.canonical) wrong.push(`${c.zone} ${c.lexeme}: got ${got}, want ${c.canonical}`);
    }
    expect({ wrong: wrong.length, sample: wrong.slice(0, 3) }).toEqual({ wrong: 0, sample: [] });
  });

  it("counterpartyEnvelopeHash equals the Python envelope hash", () => {
    const wrong: string[] = [];
    for (const c of TABLE.cases) {
      const text = `{"payload":{"v":${c.lexeme}},"signature":${JSON.stringify(TABLE.signature)}}`;
      const got = counterpartyEnvelopeHash(parseJsonPreservingFloats(text));
      if (got !== c.envelope_hash) wrong.push(`${c.zone} ${c.lexeme}: ${got} != ${c.envelope_hash}`);
    }
    expect({ wrong: wrong.length, sample: wrong.slice(0, 3) }).toEqual({ wrong: 0, sample: [] });
  });

  // JSON.parse reads 500.0 as 500, so only an integer-valued float can differ
  const parseLoses = (v: number): boolean => Number.isInteger(v) && !Object.is(v, -0) && Math.abs(v) <= 2 ** 53 - 1;
  const origin = (c: FloatCase): string => `{"payload":{"v":${c.lexeme}},"signature":${JSON.stringify(TABLE.signature)}}`;
  const ack = (hash: string) => ({
    payload: { counterparty_binding: { scope: "envelope_minus_anchors", receipt_ref: "r1", envelope_hash: hash } },
    signature: { kid: "kid-B" },
  });

  it.each(["JSON.parse", "parseJsonPreservingFloats"] as const)(
    "verifyCounterpartyBinding on a %s origin gives the Python label",
    (parser) => {
      const wrong: string[] = [];
      for (const c of TABLE.cases) {
        const parsed = parser === "JSON.parse" ? JSON.parse(origin(c)) : parseJsonPreservingFloats(origin(c));
        const lost = parser === "JSON.parse" && parseLoses(JSON.parse(c.lexeme) as number);
        const want = lost ? "mismatch" : c.binding;
        const got = verifyCounterpartyBinding(ack(c.envelope_hash), parsed).label;
        if (got !== want) wrong.push(`${c.zone} ${c.lexeme}: ${got} != ${want}`);
      }
      expect({ wrong: wrong.length, sample: wrong.slice(0, 3) }).toEqual({ wrong: 0, sample: [] });
    },
  );

  it("computeEnvelopeHash on a JSON.parse origin is the Python hash for -0.0 and 1e-9..1e-4", () => {
    const rows = TABLE.cases.filter((c) => c.zone === "small" || c.repr === "-0.0");
    expect(rows.length).toBeGreaterThan(300);
    for (const c of rows) expect(computeEnvelopeHash(JSON.parse(origin(c))), c.lexeme).toBe(c.envelope_hash);
  });

  it("a plain -0 hashes as -0.0, so an integer -0 Python reads as 0 no longer matches", () => {
    const text = (n: string) => `{"payload":{"v":${n}},"signature":${JSON.stringify(TABLE.signature)}}`;
    const negZero = TABLE.cases.find((c) => c.lexeme === "-0.0")!.envelope_hash;
    expect(computeEnvelopeHash(parseJsonPreservingFloats(text("-0")))).toBe(negZero);
    expect(computeEnvelopeHash(JSON.parse(text("-0")))).toBe(negZero);
  });

  it("a RawFloat built by hand, with no lexeme, is written the same way", () => {
    for (const c of TABLE.cases) {
      expect(decode(asqavJcs({ v: new RawFloat(Number(c.lexeme)) }))).toBe(c.canonical);
    }
  });

  it("the named failing spellings", () => {
    const one = (lexeme: string) => decode(asqavJcs(parseJsonPreservingFloats(`[${lexeme}]`)));
    expect(one("1e-7")).toBe("[1e-07]");
    expect(one("0.00001")).toBe("[1e-05]");
    expect(one("0.0001")).toBe("[0.0001]");
    expect(one("76355072847686510.0")).toBe("[7.635507284768651e+16]");
    expect(one("9999999999999998.0")).toBe("[9999999999999998.0]");
    expect(one("1e21")).toBe("[1e+21]");
    expect(one("-0.0")).toBe("[-0.0]");
    expect(one("500.0")).toBe("[500.0]");
  });

  it("a float Python refuses is refused here too", () => {
    expect(() => asqavJcs(parseJsonPreservingFloats("[1e999]"))).toThrow(/non-finite/);
  });
});
