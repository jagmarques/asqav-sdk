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

describe("four-state key distinction (criterion 574/F5)", () => {
  it("never PASSes a decommissioned key as active", () => {
    const [res, note] = checkKeyStatus("decommissioned", ISSUED_AT, null, false);
    expect(res).toBe("SKIPPED");
    expect(note.toLowerCase()).toContain("decommissioned");
  });

  it("never PASSes an unknown status as active", () => {
    for (const unknown of [null, ""]) {
      const [res, note] = checkKeyStatus(unknown, ISSUED_AT, null, false);
      expect(res).toBe("SKIPPED");
      expect(note.toLowerCase()).toContain("unknown");
    }
  });

  it("names a compromise reason distinctly", () => {
    const [res, note] = checkKeyStatus("revoked", "2026-05-01T00:00:00Z", ISSUED_AT, false, "compromised");
    expect(res).toBe("FAIL");
    expect(note.toLowerCase()).toContain("compromised");
  });
});

describe("four-state hardening (criterion 574/F5 critic follow-ups)", () => {
  it("SKIPs the literal unknown status too", () => {
    const [res, note] = checkKeyStatus("unknown", ISSUED_AT, null, false);
    expect(res).toBe("SKIPPED");
    expect(note.toLowerCase()).toContain("unknown");
  });

  it("never crashes on a non-string revocation reason", () => {
    const [res, note] = checkKeyStatus("revoked", "2026-05-01T00:00:00Z", ISSUED_AT, false, 123 as unknown as string);
    expect(res).toBe("SKIPPED");
    expect(note.toLowerCase()).toContain("revoked");
  });
});

describe("compromise is decisive (criterion 810)", () => {
  it("FAILs a compromised key after issuance even with a trusted anchor", () => {
    const [res, note] = checkKeyStatus(
      "compromised", "2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z", true,
    );
    expect(res).toBe("FAIL");
    expect(note.toLowerCase()).toContain("compromised");
  });

  it("FAILs a revoked key whose reason is compromised, anchor or not", () => {
    const [res] = checkKeyStatus(
      "revoked", "2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z", true, "compromised",
    );
    expect(res).toBe("FAIL");
  });

  it("lets compromise decide even when the status label says active", () => {
    const [res, note] = checkKeyStatus(
      "active", "2026-05-01T00:00:00Z", null, false, "compromised",
    );
    expect(res).toBe("FAIL");
    expect(note.toLowerCase()).toContain("compromised");
  });

  it("lets compromise decide over a decommissioned label too", () => {
    const [res] = checkKeyStatus("decommissioned", "2026-05-01T00:00:00Z", null, false, "compromised");
    expect(res).toBe("FAIL");
  });

  it("keeps a retired (not compromised) key passing with a trusted anchor", () => {
    const [res] = checkKeyStatus("revoked", "2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z", true);
    expect(res).toBe("PASS");
  });
});
