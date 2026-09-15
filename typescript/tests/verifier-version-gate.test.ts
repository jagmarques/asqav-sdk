// An absent v and an unrecognised v are unverifiable in both wire modes
import { describe, expect, it } from "vitest";

import { AsqavNativeAdapter } from "../src/verifier/adapters/asqavNative.js";
import { checkStructure, isRecognisedWireVersion } from "../src/verifier/vrShim.js";

function payloadDoc(v: unknown, present = true): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    type: "protectmcp:decision",
    issued_at: "2026-05-04T12:00:00+00:00",
    issuer_id: "Asqav Ltd",
    agent_id: "agt_demo_001",
    action_ref: "sha256:" + "d".repeat(64),
    context: { subject: "version-gate" },
    payload_digest: { hash: "d".repeat(64), size: 27 },
    policy_digest: "sha256:" + "9".repeat(64),
    previousReceiptHash: "0".repeat(64),
    decision: "allow",
    mode: "payload",
    tool_name: "demo.action",
  };
  if (present) payload.v = v;
  return payload;
}

function hashDoc(v: unknown): Record<string, unknown> {
  return {
    v,
    mode: "hash",
    payload: null,
    hash: "sha256:" + "6".repeat(64),
    hash_algo: "sha256",
    metadata: {},
    server_timestamp: "2026-06-04T20:31:38.361764",
    action_id: "act_version_gate",
    agent_id: "agt_version_gate",
    org_id: "org_version_gate",
    policy_digest: "7".repeat(64),
    policy_decision: "permit",
    algorithm: "Ed25519",
    key_id: "org_version_gate",
    signature_b64: "AA",
  };
}

describe("wire version gate (draft §5.2.1)", () => {
  it("recognises integers 1 and 2, nothing else", () => {
    expect(isRecognisedWireVersion(1)).toBe(true);
    expect(isRecognisedWireVersion(2)).toBe(true);
    for (const v of [0, 99, -1, "1", 1.5, true, null, undefined, NaN]) {
      expect(isRecognisedWireVersion(v)).toBe(false);
    }
  });

  it("payload mode: absent v SKIPs without inferring version 1", () => {
    const [res, note] = checkStructure(payloadDoc(1, false));
    expect(res).toBe("SKIPPED");
    expect(note).toContain("absence is not version 1");
  });

  it("payload mode: unrecognised v SKIPs without guessing its shape", () => {
    const [res, note] = checkStructure(payloadDoc(99));
    expect(res).toBe("SKIPPED");
    expect(note).toContain("unsupported wire version 99");
  });

  it("payload mode: recognised v still PASSes structure", () => {
    expect(checkStructure(payloadDoc(1))[0]).toBe("PASS");
    expect(checkStructure(payloadDoc(2))[0]).toBe("PASS");
  });

  it("hash mode: unrecognised v SKIPs like the payload gate", () => {
    const adapter = new AsqavNativeAdapter();
    const [res, note] = adapter.schema(hashDoc(99));
    expect(res).toBe("SKIPPED");
    expect(note).toContain("unsupported wire version 99");
  });

  it("hash mode: recognised v still PASSes schema", () => {
    const adapter = new AsqavNativeAdapter();
    expect(adapter.schema(hashDoc(1))[0]).toBe("PASS");
  });
});
