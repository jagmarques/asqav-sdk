// Criterion 625: an unregistered type namespace is REPORTED, never failed
import { describe, expect, it } from "vitest";

import { checkStructure } from "../src/verifier/vrShim.js";

describe("checkStructure reports an unregistered namespace", () => {
  it("returns SKIPPED naming the type", () => {
    const payload: Record<string, unknown> = {
      type: "protectmcp:lifecycle:oversight_ruling",
      issued_at: "2026-06-01T19:26:44Z",
      issuer_id: "org-1",
      action_ref: "sha256:" + "8".repeat(64),
      payload_digest: { hash: "8".repeat(64), size: 1 },
      policy_digest: "sha256:" + "3".repeat(64),
      previousReceiptHash: "0".repeat(64),
      decision: "observation",
    };
    const [res, note] = checkStructure(payload);
    expect(res).toBe("SKIPPED");
    expect(note).toContain("protectmcp:lifecycle:oversight_ruling");
  });

  it("keeps FAIL for genuinely missing required fields", () => {
    const payload: Record<string, unknown> = {
      type: "protectmcp:lifecycle:oversight_ruling",
    };
    const [res, note] = checkStructure(payload);
    expect(res).toBe("FAIL");
    expect(note).toContain("missing required fields");
  });
});
