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

interface FloatCase {
  zone: string;
  lexeme: string;
  repr: string;
  canonical: string;
  envelope_hash: string;
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
