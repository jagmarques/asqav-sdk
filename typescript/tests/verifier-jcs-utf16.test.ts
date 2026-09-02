// The asqav-native verification path orders member names by UTF-16 code unit (RFC 8785 3.2.3) and
// names the pre-cutover dialect instead of verifying it. TS half of test_verify_receipt_jcs_utf16.py.

import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";

import { AsqavNativeAdapter } from "../src/verifier/adapters/asqavNative.js";
import {
  JCS_UTF16_CUTOVER,
  asqavJcs,
  asqavJcsPreCutover,
  hasSupplementaryMemberName,
} from "../src/verifier/canonical.js";
import { verify } from "../src/verifier/core.js";

const ORG = "f94f66c0-c580-432d-a041-29374f7aee07";
const ASTRAL: Record<string, unknown> = { "＠": 1, "😀": 1 };
const RFC8785_BYTES = '{"😀":1,"＠":1}';
const RFC8785_SHA256 = "425159f5c1f0575fbcbf9d05a8f60cde3d040eae5166aa2136657564048651b6";
const CODE_POINT_BYTES = '{"＠":1,"😀":1}';

const sha256Hex = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");
const sign = (msg: Uint8Array, sk: Uint8Array) => Buffer.from(ml_dsa65.sign(msg, sk)).toString("base64");

function key(kid: string, agentId: string, publicKey: Uint8Array): Record<string, unknown> {
  return {
    kid,
    agent_id: agentId,
    issuer_id: ORG,
    alg: "ML-DSA-65",
    public_key: Buffer.from(publicKey).toString("base64"),
    status: "active",
  };
}

function astralPayload(issuedAt: string): Record<string, unknown> {
  const context = { tool_input: ASTRAL };
  const encoded = asqavJcs(context);
  return {
    type: "protectmcp:decision",
    issued_at: issuedAt,
    issuer_id: ORG,
    agent_id: "agt_two",
    action_ref: `sha256:${"8".repeat(64)}`,
    payload_digest: { hash: sha256Hex(encoded), size: encoded.length },
    policy_digest: `sha256:${"3".repeat(64)}`,
    previousReceiptHash: "0".repeat(64),
    decision: "allow",
    context,
  };
}

function receipt(p: Record<string, unknown>, sig: string): Record<string, unknown> {
  return { payload: p, signature: { alg: "ML-DSA-65", kid: ORG, sig }, anchors: [] };
}

function signatureAxis(p: Record<string, unknown>, signedBytes: Uint8Array) {
  const agent = ml_dsa65.keygen();
  const jwks = { keys: [key("agent-two", "agt_two", agent.publicKey)] };
  const res = verify(receipt(p, sign(signedBytes, agent.secretKey)), [new AsqavNativeAdapter()], jwks);
  const axis = res.axes.find((a) => a.axis === "signature");
  if (axis === undefined) throw new Error("no signature axis");
  return { axis, res };
}

describe("asqavJcs member order", () => {
  it("orders supplementary-plane names by UTF-16 code unit", () => {
    const out = asqavJcs(ASTRAL);
    expect(new TextDecoder().decode(out)).toBe(RFC8785_BYTES);
    expect(sha256Hex(out)).toBe(RFC8785_SHA256);
  });

  it("keeps the code-point order only as the named pre-cutover dialect", () => {
    expect(new TextDecoder().decode(asqavJcsPreCutover(ASTRAL))).toBe(CODE_POINT_BYTES);
    expect(new TextDecoder().decode(asqavJcs(ASTRAL))).not.toBe(CODE_POINT_BYTES);
  });

  it("detects a supplementary member name at any depth and ignores values", () => {
    expect(hasSupplementaryMemberName({ a: [{ b: { "😀": 1 } }] })).toBe(true);
    expect(hasSupplementaryMemberName({ a: [{ b: { "￿": "😀" } }] })).toBe(false);
  });
});

describe("asqav-native signature axis across the dialect cutover", () => {
  it("verifies a receipt with supplementary member names signed over RFC 8785 bytes", () => {
    const p = astralPayload("2026-06-19T00:00:00.000000Z");
    const { axis, res } = signatureAxis(p, asqavJcs(p));
    expect(axis.result).toBe("PASS");
    const digest = res.axes.find((a) => a.axis === "payload_digest");
    expect(digest?.result).toBe("PASS");
  });

  it("names the pre-cutover dialect for an older receipt and never verifies it", () => {
    const p = astralPayload("2026-06-19T00:00:00.000000Z");
    const { axis, res } = signatureAxis(p, asqavJcsPreCutover(p));
    expect(axis.result).toBe("FAIL");
    expect(axis.note).toContain("pre-cutover dialect");
    expect(res.verdict).toBe("unverified");
  });

  it("gives a receipt issued after the cutover no retry", () => {
    const later = new Date(Date.parse(JCS_UTF16_CUTOVER) + 365 * 24 * 3600 * 1000).toISOString();
    const p = astralPayload(later);
    const { axis, res } = signatureAxis(p, asqavJcsPreCutover(p));
    expect(axis.result).toBe("FAIL");
    expect(axis.note).not.toContain("pre-cutover dialect");
    expect(res.verdict).toBe("unverified");
  });

  it("leaves a BMP-only receipt untouched, both orders being the same bytes", () => {
    const p = astralPayload("2026-06-19T00:00:00.000000Z");
    p.context = { tool_input: { a: 1, "￿": 2 } };
    const encoded = asqavJcs(p.context);
    p.payload_digest = { hash: sha256Hex(encoded), size: encoded.length };
    const { axis } = signatureAxis(p, asqavJcsPreCutover(p));
    expect(axis.result).toBe("PASS");
  });
});
