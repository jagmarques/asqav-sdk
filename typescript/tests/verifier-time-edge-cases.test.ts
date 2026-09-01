// Time-edge vectors, TS half (criterion 422). One JSON table drives both verifiers and every
// case freezes the wall clock, so no verdict depends on the run date.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS, checkExpiry, checkSkew, SKEW_BOUND_SECONDS } from "../src/verifier/index.js";
import { runOne } from "../src/verifier/runner.js";

interface Probe {
  clock: string;
  result: string;
  note: string;
  exact: boolean;
}

interface InstantCase {
  name: string;
  stamp: string;
  utc: string;
  phenomenon: string;
  probes: Probe[];
}

interface SkewCase {
  name: string;
  issued_at: string;
  expect: { result: string; exact: boolean; note?: string; note_contains?: string };
}

interface ExpiryCase {
  name: string;
  payload: Record<string, unknown>;
  expect: { result: string; exact: boolean; note: string };
}

const CASES_FILE = resolve(__dirname, "..", "..", "verifier", "time-edge-cases.json");
const TABLE = JSON.parse(readFileSync(CASES_FILE, "utf-8")) as {
  frozen_clock: string;
  instants: InstantCase[];
  skew_bounds: SkewCase[];
  expiry: ExpiryCase[];
};

const TIME_EDGE_VECTOR = resolve(
  __dirname, "..", "..", "verifier", "conformance-vectors", "asqav-12-time-edge-expiry",
);

function loadVector(file: string): Record<string, unknown> {
  const raw = readFileSync(resolve(TIME_EDGE_VECTOR, file), "utf-8");
  return JSON.parse(raw) as Record<string, unknown>;
}

// The probes pin each instant to the second, so the clock must sit exactly there
beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("the time-edge table is populated", () => {
  it("keeps the Python bound", () => {
    expect(SKEW_BOUND_SECONDS).toBe(300);
  });

  it("carries both outcomes and enough cases", () => {
    expect(TABLE.instants.length).toBeGreaterThanOrEqual(4);
    expect(TABLE.skew_bounds.length).toBeGreaterThanOrEqual(6);
    expect(TABLE.expiry.length).toBeGreaterThanOrEqual(6);
    const skewOutcomes = new Set(TABLE.skew_bounds.map((c) => c.expect.result));
    expect(skewOutcomes).toEqual(new Set(["PASS", "FAIL"]));
  });
});

describe("DST and offset instants normalise to the pinned UTC instant", () => {
  for (const c of TABLE.instants) {
    it(c.name, () => {
      // At utc-301s the stamp is 301s ahead and FAILs; at utc-300s and utc it PASSes.
      // Only the true UTC instant draws all three, pinning the parse on the TS side.
      for (const p of c.probes) {
        vi.setSystemTime(Date.parse(p.clock));
        const [result, note] = checkSkew(c.stamp);
        expect(result, `${c.name} @ ${p.clock}: ${note}`).toBe(p.result);
        expect(note, `${c.name} @ ${p.clock}`).toBe(p.note);
      }
    });
  }

  it("resolves the ambiguous fall-back wall clock to two instants", () => {
    const first = TABLE.instants.find((c) => c.name === "north-fall-back-first-occurrence")!;
    const second = TABLE.instants.find((c) => c.name === "north-fall-back-second-occurrence")!;
    expect(first.stamp.slice(0, 16)).toBe(second.stamp.slice(0, 16));
    expect(Date.parse(second.utc) - Date.parse(first.utc)).toBe(3600_000);
  });
});

describe("the skew bound is future-only at the frozen clock", () => {
  for (const c of TABLE.skew_bounds) {
    it(c.name, () => {
      vi.setSystemTime(Date.parse(TABLE.frozen_clock));
      const [result, note] = checkSkew(c.issued_at);
      expect(result, `${c.name}: ${note}`).toBe(c.expect.result);
      if (c.expect.exact) expect(note, c.name).toBe(c.expect.note);
      else expect(note, c.name).toContain(c.expect.note_contains);
    });
  }

  it("passes a far-past stamp exactly like a fresh one", () => {
    vi.setSystemTime(Date.parse(TABLE.frozen_clock));
    const fresh = TABLE.skew_bounds.find((c) => c.name === "fresh-at-frozen-clock")!;
    const old = TABLE.skew_bounds.find((c) => c.name === "far-past-three-years")!;
    expect(checkSkew(fresh.issued_at)[0]).toBe("PASS");
    const [result, note] = checkSkew(old.issued_at);
    expect(result).toBe("PASS");
    expect(note).toContain("within bound");
  });
});

describe("the expiry axis at the frozen clock", () => {
  for (const c of TABLE.expiry) {
    it(c.name, () => {
      vi.setSystemTime(Date.parse(TABLE.frozen_clock));
      const [result, note] = checkExpiry(c.payload);
      expect(result, `${c.name}: ${note}`).toBe(c.expect.result);
      expect(note, c.name).toBe(c.expect.note);
    });
  }

  it("fails closed on an unreadable expires_at", () => {
    vi.setSystemTime(Date.parse(TABLE.frozen_clock));
    const [result, note] = checkExpiry({ expires_at: "not-a-stamp" });
    expect(result).toBe("FAIL");
    expect(note).toContain("refused rather than read as no expiry");
  });
});

describe("the time-edge corpus vector keeps expiry on its own axis", () => {
  it("verdict stays verified while the lapsed expires_at FAILs alone", () => {
    const result = verify(loadVector("receipt.json"), ADAPTERS, loadVector("jwks.json"));
    expect(result.fmt).toBe("asqav-native");
    const expiry = result.axes.find((a) => a.axis === "expiry");
    expect(expiry?.result).toBe("FAIL");
    expect(expiry?.note).toContain("lapsed");
    const nonExpiry = result.axes.filter((a) => a.axis !== "expiry");
    expect(nonExpiry.map((a) => a.result)).not.toContain("FAIL");
    expect(result.verdict).toBe("verified");
  });

  it("matches its expected outcome through the corpus runner", () => {
    const raw = readFileSync(resolve(TIME_EDGE_VECTOR, "expected.json"), "utf-8");
    const expected = JSON.parse(raw) as {
      format: string; outcome: string; reason_code: string;
    };
    const outcome = runOne(
      TIME_EDGE_VECTOR, expected.format, expected.outcome, expected.reason_code,
    );
    expect(outcome.ok, outcome.detail).toBe(true);
  });
});
