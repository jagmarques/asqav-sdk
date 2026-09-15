// A bare revocation must not accuse forgery (criterion 276)
// TS half of test_verify_receipt_revoked_key.py; both engines must agree

import { describe, expect, it } from "vitest";

import { AsqavNativeAdapter } from "../src/verifier/adapters/asqavNative.js";
import { FAILURE_UNVERIFIABLE, axisFailureClass } from "../src/verifier/core.js";
import { checkKeyStatus } from "../src/verifier/vrShim.js";

const ISSUED_AT = "2026-06-01T00:00:00Z";

function jwks(status: string, revokedAt?: string): Record<string, unknown> {
  const entry: Record<string, unknown> = {
    kid: "agent-revoked-001",
    issuer_id: "agent-revoked-001",
    alg: "ML-DSA-65",
    public_key: "QUFBQQ==",
    status,
  };
  if (revokedAt !== undefined) entry.revoked_at = revokedAt;
  return { keys: [entry] };
}

function doc(): Record<string, unknown> {
  return {
    payload: {
      type: "protectmcp:decision",
      issued_at: ISSUED_AT,
      issuer_id: "agent-revoked-001",
      action_ref: `sha256:${"8".repeat(64)}`,
      payload_digest: { hash: "8".repeat(64), size: 512 },
      policy_digest: `sha256:${"3".repeat(64)}`,
      previousReceiptHash: "0".repeat(64),
      decision: "allow",
    },
    signature: { alg: "ML-DSA-65", kid: "agent-revoked-001", sig: "AAAA" },
    anchors: [],
  };
}

describe("revoked key without a published revoked_at (criterion 276)", () => {
  it("reports SKIPPED, not FAIL, when no revoked_at is published", () => {
    const [res, note] = checkKeyStatus("revoked", ISSUED_AT, null, false);
    expect(res).toBe("SKIPPED");
    expect(note.toLowerCase()).toContain("revoked_at");
  });

  it("folds that SKIPPED to unverifiable, never invalid", () => {
    const [res, note] = checkKeyStatus("revoked", ISSUED_AT, null, false);
    expect(axisFailureClass("key_status", res, note)).toBe(FAILURE_UNVERIFIABLE);
  });

  it("surfaces SKIPPED on the adapter axis too", () => {
    const axes = new AsqavNativeAdapter().extraAxes(doc(), jwks("revoked"));
    const found = axes.find(([name]) => name === "key_status");
    expect(found?.[1]).toBe("SKIPPED");
  });

  it("still FAILs a revocation dated on or before issuance", () => {
    // The control: a placed revocation keeps its current outcome
    const [res] = checkKeyStatus("revoked", ISSUED_AT, "2026-05-01T00:00:00Z", false);
    expect(res).toBe("FAIL");
  });

  it("still SKIPs a post-issuance revocation without an anchor", () => {
    // The control: the existing unplaceable-timing outcome is unchanged
    const [res] = checkKeyStatus("revoked", "2026-05-01T00:00:00Z", ISSUED_AT, false);
    expect(res).toBe("SKIPPED");
  });
});
