// The offline verifier refuses a receipt whose signed expires_at has passed, read from inside
// the signed bytes. Python half: python/tests/test_expiry_offline.py

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

// toISOString always emits milliseconds, so ~1 receipt in 60 lands in the 59th second.
// Number("59.656") read as out of range failed it closed while Python passed the same bytes.
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
