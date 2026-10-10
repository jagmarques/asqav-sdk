// One verdict per signed float receipt across languages (criterion 747); the
// Python half, test_verdict_parity_cases.py, asserts the same manifest

import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { verifyReceiptOffline } from "../src/index.js";
import { runOne } from "../src/verifier/runner.js";

interface Outcome {
  outcome: string;
  failure_class: string | null;
}
interface ParityCase extends Outcome {
  dir: string;
  lexeme: string;
  parsed: Outcome;
}

const CASES_DIR = resolve(__dirname, "..", "..", "verifier", "verdict-parity-cases");
const MANIFEST = (JSON.parse(readFileSync(join(CASES_DIR, "manifest.json"), "utf8")) as { cases: ParityCase[] })
  .cases;
const read = (c: ParityCase, f: string): string | null =>
  existsSync(join(CASES_DIR, c.dir, f)) ? readFileSync(join(CASES_DIR, c.dir, f), "utf8") : null;
const parse = (t: string | null): Record<string, unknown> | null => (t === null ? null : JSON.parse(t));

describe("signed float receipts get the Python verdict", () => {
  it("the manifest covers every zone with tamper controls", () => {
    const lexemes = new Set(MANIFEST.map((c) => c.lexeme));
    for (const l of ["1e-07", "7.635507284768651e+16", "-0.0", "500.0", "0.5"]) expect(lexemes.has(l)).toBe(true);
    expect(MANIFEST.filter((c) => c.dir.endsWith("-tamper")).length).toBe(10);
  });

  it.each(MANIFEST.map((c) => [c.dir, c] as const))("%s: runner and raw-text API", (_dir, c) => {
    const want = [c.outcome, c.failure_class];
    const raw = runOne(join(CASES_DIR, c.dir), "asqav-native", c.outcome, "", c.failure_class ?? "");
    expect([raw.actualVerdict, raw.actualFailureClass || null], raw.detail.slice(0, 300)).toEqual(want);
    const text = verifyReceiptOffline(read(c, "receipt.json")!, read(c, "jwks.json")!, null, read(c, "originating_envelope.json"));
    expect([text.verdict, text.failureClass]).toEqual(want);
  });

  it.each(MANIFEST.map((c) => [c.dir, c] as const))("%s: JSON.parse object API", (_dir, c) => {
    const r = verifyReceiptOffline(
      parse(read(c, "receipt.json"))!,
      parse(read(c, "jwks.json"))!,
      null,
      parse(read(c, "originating_envelope.json")),
    );
    expect([r.verdict, r.failureClass]).toEqual([c.parsed.outcome, c.parsed.failure_class]);
    // The documented residual turns unverifiable into invalid, never verified
    if (c.outcome === "verified") expect(c.parsed).toEqual({ outcome: c.outcome, failure_class: c.failure_class });
  });
});
