// The offline verifier refuses a receipt whose signed expires_at is in the past, so
// the verdict does not depend on which verifier the reader runs. The window is read
// from inside the signed bytes; the TypeScript half of the Python control lives in
// python/tests/test_expiry_offline.py

import { describe, expect, it } from "vitest";

import { checkExpiry, checkSkew } from "../src/verifier/index.js";

function stampAt(offsetSeconds: number): string {
  return new Date(Date.now() + offsetSeconds * 1000).toISOString();
}

describe("the offline expiry axis", () => {
  it("fails a receipt whose signed expires_at lapsed", () => {
    const [result, note] = checkExpiry({ expires_at: stampAt(-3600) });
    expect(result).toBe("FAIL");
    expect(note).toContain("lapsed");
  });

  it("passes a non-expired control", () => {
    const [result] = checkExpiry({ expires_at: stampAt(3600) });
    expect(result).toBe("PASS");
  });

  it("passes a receipt that declares no expiry", () => {
    const [result] = checkExpiry({});
    expect(result).toBe("PASS");
  });

  it("fails closed on an unreadable expires_at", () => {
    const [result] = checkExpiry({ expires_at: "not-a-stamp" });
    expect(result).toBe("FAIL");
  });
});

// A stamp in the 59th second with a fractional part is ordinary: Date#toISOString
// always emits milliseconds, so roughly one receipt in sixty lands here. Range-
// checking Number("59.656") read it as out of range and failed the receipt closed,
// while the Python half range-checks a two-digit capture and passed the same bytes.
// A verdict that depends on which half you run, and on the second of the minute.
describe("a fractional second in the 59th second (cross-language parity)", () => {
  const STAMPS = [
    "2099-01-01T06:30:59.656Z",
    "2099-01-01T00:00:59.001Z",
    "2099-12-31T23:59:59.999Z",
  ];

  for (const stamp of STAMPS) {
    it(`reads ${stamp} as a real expiry, not as unreadable`, () => {
      const [result, note] = checkExpiry({ expires_at: stamp });
      expect(note).not.toContain("unreadable");
      expect(result).toBe("PASS");
    });
  }

  it("still refuses a genuinely out-of-range second", () => {
    const [result] = checkExpiry({ expires_at: "2099-01-01T06:30:60.500Z" });
    expect(result).toBe("FAIL");
  });

  it("applies to issued_at as well, where the class would be invalid", () => {
    const [result, note] = checkSkew("2026-01-01T00:00:59.500Z");
    expect(note).not.toContain("unparseable");
    expect(result).toBe("PASS");
  });
});
