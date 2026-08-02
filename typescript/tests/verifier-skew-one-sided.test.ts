// The clock-skew axis is one-sided: a stamp ahead of the wall clock past the bound
// fails, while a backdated stamp passes, because the wall clock cannot detect a lie
// about the past. The TypeScript half of the Python skew table pins the same bound

import { describe, expect, it } from "vitest";

import { checkSkew, SKEW_BOUND_SECONDS } from "../src/verifier/index.js";

function stampAt(offsetSeconds: number): string {
  return new Date(Date.now() + offsetSeconds * 1000).toISOString();
}

describe("the skew bound is one-sided", () => {
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
