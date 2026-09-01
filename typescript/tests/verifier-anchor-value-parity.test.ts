/**
 * Anchor value base64 tolerance, differential against Python. The anchors field is unsigned
 * and attacker-steerable, so the permissive direction (a false PASS) is asserted at zero.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { checkAnchors } from "../src/verifier/vrShim.js";

interface ValueCase {
  value: string;
  expect: { safe_b64: boolean; axis: string };
}

const CASES_FILE = resolve(__dirname, "..", "..", "verifier", "anchor-value-cases.json");
const VALUES = (JSON.parse(readFileSync(CASES_FILE, "utf-8")) as { values: ValueCase[] }).values;

const ENVELOPE = {
  payload: { type: "protectmcp:decision", issued_at: "2026-06-19T00:00:00+00:00" },
  signature: { alg: "ML-DSA-65", kid: "k1", sig: "AAAA" },
};

/** The axis result for one anchor value: FAIL on junk, SKIPPED (unverifiable) on
 * shape-valid non-tokens since the cryptographic anchor check landed. */
function axisFor(value: string): string {
  return checkAnchors({ ...ENVELOPE, anchors: [{ type: "rfc3161", value }] })[0];
}

describe("anchor value base64 parity", () => {
  it("corpus is populated and covers non-ASCII", () => {
    expect(VALUES.length).toBeGreaterThanOrEqual(900);
    const wide = VALUES.filter((c) => [...c.value].some((ch) => ch.codePointAt(0)! > 127));
    expect(wide.length).toBeGreaterThanOrEqual(350);
    // Both directions present: junk FAILs (invalid), shape-valid non-tokens
    // SKIP (unverifiable, never PASS on presence).
    expect(VALUES.filter((c) => c.expect.axis === "FAIL").length).toBeGreaterThanOrEqual(100);
    expect(VALUES.filter((c) => c.expect.axis === "SKIPPED").length).toBeGreaterThanOrEqual(100);
  });

  // Python's b64decode(validate=True) accepts this shape on 3.11 and raises on
  // 3.12, so a corpus without it cannot separate the two rules
  it("covers the surplus-padding class", () => {
    const excess = VALUES.filter((c) => {
      const s = c.value.replace(/-/g, "+").replace(/_/g, "/");
      const padded = s + "=".repeat(((-s.length % 4) + 4) % 4);
      return /^[A-Za-z0-9+/]+={3,}$/.test(padded);
    });
    expect(excess.length).toBeGreaterThanOrEqual(20);
    expect(excess.filter((c) => c.expect.axis !== "FAIL").map((c) => c.value)).toEqual([]);
  });

  it("has zero permissive disagreements with Python", () => {
    const permissive: string[] = [];
    for (const c of VALUES) {
      if (c.expect.axis === "FAIL" && axisFor(c.value) === "PASS") permissive.push(c.value);
    }
    expect(permissive, `TS reads these as base64-ok where Python refuses: ${JSON.stringify(permissive.slice(0, 8))}`).toEqual([]);
  });

  it("agrees with Python on every value", () => {
    const disagree: string[] = [];
    for (const c of VALUES) {
      if (axisFor(c.value) !== c.expect.axis) disagree.push(c.value);
    }
    expect(disagree, `disagreements: ${JSON.stringify(disagree.slice(0, 8))}`).toEqual([]);
  });

  it("refuses a value carrying any non-ASCII codepoint", () => {
    for (const value of ["é", "中文", "😀", "MTIzNA==é", " "]) {
      expect(axisFor(value), value).toBe("FAIL");
    }
  });
});
