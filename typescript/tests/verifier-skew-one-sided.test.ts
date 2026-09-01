// The clock-skew axis is one-sided: a stamp past the future bound fails while a backdated one
// passes, since the wall clock cannot detect a lie about the past. Same bound as Python.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { checkSkew, SKEW_BOUND_SECONDS } from "../src/verifier/index.js";

function stampAt(offsetSeconds: number): string {
  return new Date(Date.now() + offsetSeconds * 1000).toISOString();
}

describe("the skew bound is one-sided", () => {
  // A frozen clock keeps the boundary cases exact under slow CI workers.
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(Date.parse("2026-08-04T12:00:00Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("keeps the Python bound", () => {
    expect(SKEW_BOUND_SECONDS).toBe(300);
  });

  it("passes a stamp in the past", () => {
    const [result] = checkSkew(stampAt(-3600));
    expect(result).toBe("PASS");
  });

  it("passes a stamp just inside the bound", () => {
    const [result] = checkSkew(stampAt(SKEW_BOUND_SECONDS - 5));
    expect(result).toBe("PASS");
  });

  it("fails a stamp past the bound", () => {
    const [result, note] = checkSkew(stampAt(SKEW_BOUND_SECONDS + 60));
    expect(result).toBe("FAIL");
    expect(note).toContain("ahead of wall clock");
  });
});
