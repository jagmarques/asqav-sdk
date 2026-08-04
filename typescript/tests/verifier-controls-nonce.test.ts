// Criterion 435: closed controls_evaluated key set and duplicate-nonce replay flag.
import { describe, expect, it } from "vitest";

import { AsqavNativeAdapter } from "../src/verifier/adapters/asqavNative.js";
import { checkNonce, checkStructure } from "../src/verifier/vrShim.js";

function basePayload(): Record<string, unknown> {
  return {
    type: "protectmcp:decision",
    issued_at: "2026-06-01T19:26:44Z",
    issuer_id: "org-1",
    action_ref: "sha256:" + "8".repeat(64),
    payload_digest: { hash: "8".repeat(64), size: 1 },
    policy_digest: "sha256:" + "3".repeat(64),
    previousReceiptHash: "0".repeat(64),
    decision: "allow",
  };
}

function doc(nonce: string): Record<string, unknown> {
  return {
    payload: { ...basePayload(), nonce },
    signature: { alg: "ML-DSA-65", kid: "k", sig: "QUFB" },
    anchors: [],
  };
}

describe("checkStructure enforces the controls_evaluated closed key set", () => {
  it("rejects a key outside the closed set", () => {
    const payload = { ...basePayload(), controls_evaluated: { bogus: {} } };
    const [res, note] = checkStructure(payload);
    expect(res).toBe("FAIL");
    expect(note).toContain("bogus");
    expect(note).toContain("false_control_attestation_guard");
  });

  it("requires quorum fired=true and a bare 64-hex attestation_hash", () => {
    const ok = {
      ...basePayload(),
      controls_evaluated: { quorum: { fired: true, attestation_hash: "e".repeat(64) } },
    };
    expect(checkStructure(ok)[0]).toBe("PASS");
    const prefixed = {
      ...basePayload(),
      controls_evaluated: {
        quorum: { fired: true, attestation_hash: "sha256:" + "e".repeat(64) },
      },
    };
    expect(checkStructure(prefixed)[0]).toBe("FAIL");
    const notFired = {
      ...basePayload(),
      controls_evaluated: { quorum: { fired: false, attestation_hash: "e".repeat(64) } },
    };
    expect(checkStructure(notFired)[0]).toBe("FAIL");
  });

  it("requires policy evaluated=true and a real matched_count >= 1", () => {
    const ok = {
      ...basePayload(),
      controls_evaluated: { policy: { evaluated: true, matched_count: 1 } },
    };
    expect(checkStructure(ok)[0]).toBe("PASS");
    const zero = {
      ...basePayload(),
      controls_evaluated: { policy: { evaluated: true, matched_count: 0 } },
    };
    expect(checkStructure(zero)[0]).toBe("FAIL");
    const boolCount = {
      ...basePayload(),
      controls_evaluated: { policy: { evaluated: true, matched_count: true } },
    };
    expect(checkStructure(boolCount)[0]).toBe("FAIL");
  });
});

describe("checkNonce replay-candidate axis", () => {
  it("passes a receipt declaring no nonce", () => {
    const [res, note] = checkNonce(basePayload());
    expect(res).toBe("PASS");
    expect(note).toContain("no nonce");
  });

  it("documents the cloud passthrough when no index is held", () => {
    const [res, note] = checkNonce({ ...basePayload(), nonce: "ab".repeat(12) });
    expect(res).toBe("PASS");
    expect(note).toContain("duplicate_emission_candidate");
  });

  it("flags a duplicate nonce under the same issuer via a shared index", () => {
    const seen = new Set<string>();
    const payload = { ...basePayload(), nonce: "ab".repeat(12) };
    expect(checkNonce(payload, seen)[0]).toBe("PASS");
    const [res, note] = checkNonce(payload, seen);
    expect(res).toBe("FAIL");
    expect(note).toContain("replay candidate");
  });

  it("same nonce under another issuer is not a duplicate", () => {
    const seen = new Set<string>();
    const a = { ...basePayload(), nonce: "cd".repeat(12) };
    const b = { ...basePayload(), nonce: "cd".repeat(12), issuer_id: "org-2" };
    expect(checkNonce(a, seen)[0]).toBe("PASS");
    expect(checkNonce(b, seen)[0]).toBe("PASS");
  });
});

describe("AsqavNativeAdapter carries the nonce axis", () => {
  const axisResults = (adapter: AsqavNativeAdapter, d: Record<string, unknown>) =>
    Object.fromEntries(
      adapter.extraAxes(d, { keys: [] } as never).map(([name, res]) => [name, res]),
    );

  it("flags a duplicate nonce across receipts through its shared index", () => {
    const adapter = new AsqavNativeAdapter();
    const first = axisResults(adapter, doc("ab".repeat(12)));
    const second = axisResults(adapter, doc("ab".repeat(12)));
    expect(first.nonce).toBe("PASS");
    expect(second.nonce).toBe("FAIL");
  });

  it("rejects an unknown controls_evaluated key in the schema axis", () => {
    const adapter = new AsqavNativeAdapter();
    const d = doc("ef".repeat(12));
    (d.payload as Record<string, unknown>).controls_evaluated = { bogus: {} };
    const [res, note] = adapter.schema(d);
    expect(res).toBe("FAIL");
    expect(note).toContain("bogus");
  });
});
