// One verdict per signed float receipt across languages (criterion 747); the
// Python half, test_verdict_parity_cases.py, asserts the same manifest

import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { verifyCounterpartyBinding, verifyReceiptOffline } from "../src/index.js";
import { parseJsonPreservingFloats, UnsafeIntegerError } from "../src/verifier/canonical.js";
import { runOne } from "../src/verifier/runner.js";

interface Outcome {
  outcome: string;
  failure_class: string | null;
}
interface ParityCase extends Outcome {
  dir: string;
  lexeme: string;
  parsed: Outcome;
  binding?: string;
  parsed_binding?: string;
  unsafe_integer?: boolean;
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
    for (const l of ["1e-07", "7.635507284768651e+16", "-0.0", "500.0", "0.5", "5e2", "-0"]) expect(lexemes.has(l)).toBe(true);
    expect(MANIFEST.filter((c) => c.dir.endsWith("-tamper")).length).toBe(10);
  });

  it.each(MANIFEST.map((c) => [c.dir, c] as const))("%s: runner and raw-text API", (_dir, c) => {
    const want = [c.outcome, c.failure_class];
    const raw = runOne(join(CASES_DIR, c.dir), "asqav-native", c.outcome, "", c.failure_class ?? "");
    expect([raw.actualVerdict, raw.actualFailureClass || null], raw.detail.slice(0, 300)).toEqual(want);
    const text = verifyReceiptOffline(read(c, "receipt.json")!, read(c, "jwks.json")!, null, read(c, "originating_envelope.json"));
    expect([text.verdict, text.failureClass]).toEqual(want);
  });

  it.each(MANIFEST.map((c) => [c.dir, c] as const))("%s: float-preserving object API", (_dir, c) => {
    const keep = (t: string | null) => (t === null ? null : parseJsonPreservingFloats(t));
    const names = ["receipt.json", "jwks.json", "originating_envelope.json"];
    // The parse itself refuses an integer past 2**53, as raw-text ingest does
    if (c.unsafe_integer) {
      expect(() => names.map((f) => keep(read(c, f)))).toThrow(UnsafeIntegerError);
      return;
    }
    const files = names.map((f) => keep(read(c, f)));
    const r = verifyReceiptOffline(files[0] as never, files[1] as never, null, files[2] as never);
    expect([r.verdict, r.failureClass]).toEqual([c.outcome, c.failure_class]);
  });

  it.each(MANIFEST.map((c) => [c.dir, c] as const))("%s: JSON.parse object API", (_dir, c) => {
    const r = verifyReceiptOffline(
      parse(read(c, "receipt.json"))!,
      parse(read(c, "jwks.json"))!,
      null,
      parse(read(c, "originating_envelope.json")),
    );
    expect([r.verdict, r.failureClass]).toEqual([c.parsed.outcome, c.parsed.failure_class]);
    // Past Python only for a respelled or rounded unsigned integer
    if (c.parsed.outcome === "verified" && c.outcome !== "verified") expect(respelledInteger(c) || unsignedUnsafe(c)).toBe(true);
  });

  const bindings = MANIFEST.filter((c) => c.binding !== undefined);
  const bind = (c: ParityCase, origin: unknown) => verifyCounterpartyBinding(parse(read(c, "receipt.json"))!, origin).label;

  it.each(bindings.map((c) => [c.dir, c] as const))("%s: verifyCounterpartyBinding", (_dir, c) => {
    const text = read(c, "originating_envelope.json")!;
    expect(bind(c, text)).toBe(c.binding);
    if (c.unsafe_integer && c.dir.includes("-origin-")) expect(() => parseJsonPreservingFloats(text)).toThrow(UnsafeIntegerError);
    else expect(bind(c, parseJsonPreservingFloats(text))).toBe(c.binding);
    expect(bind(c, JSON.parse(text))).toBe(c.parsed_binding);
    if (c.parsed_binding === "matches" && c.binding !== "matches") expect(respelledInteger(c) || unsignedUnsafe(c)).toBe(true);
  });
});

function respelledInteger(c: ParityCase): boolean {
  return c.dir.includes("-respelled-") && /[.eE]/.test(c.lexeme) && Number.isInteger(Number(c.lexeme));
}

// JSON.parse rounds an integer past 2**53; unsigned, it goes unseen
function unsignedUnsafe(c: ParityCase): boolean {
  return c.unsafe_integer === true && c.dir.includes("-unsigned") && !Number.isSafeInteger(Number(c.lexeme));
}

describe("a parsed -0 is refused only where it is signed", () => {
  const c = MANIFEST.find((m) => m.dir === "receipt-half-repr")!;
  const receipt = (): Record<string, unknown> => parse(read(c, "receipt.json"))!;
  const check = (r: Record<string, unknown>) => verifyReceiptOffline(r, parse(read(c, "jwks.json"))!);

  it("verifies with -0 in an unsigned member and an anchor entry, as the other verifiers do", () => {
    const r = receipt();
    r.note = -0;
    (r.anchors as Array<Record<string, unknown>>)[0].x = -0;
    expect([check(r).verdict, check(r).failureClass]).toEqual(["verified", null]);
  });

  it.each(["payload", "signature"])("refuses -0 in the signed %s as unverifiable", (part) => {
    const r = receipt();
    (r[part] as Record<string, unknown>).x = -0;
    const res = check(r);
    expect([res.verdict, res.failureClass]).toEqual(["unverified", "unverifiable"]);
    expect(res.axes[0].note).toContain("receipt carries -0");
  });
});
