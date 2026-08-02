// The offline verifier refuses a receipt whose signed expires_at is in the past, so
// the verdict does not depend on which verifier the reader runs. The window is read
// from inside the signed bytes; the TypeScript half of the Python control lives in
// python/tests/test_expiry_offline.py

import { describe, expect, it } from "vitest";

import { checkExpiry } from "../src/verifier/index.js";

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
