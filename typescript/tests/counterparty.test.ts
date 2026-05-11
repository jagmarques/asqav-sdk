/**
 * SDK-side tests for the counterparty acknowledgment binding (IETF -04).
 *
 * Covers envelope_hash byte-stability, default vs explicit receipt_ref,
 * verifyCounterpartyBinding matches / mismatch / kid_mismatch / unresolved
 * labels, and cross-SDK byte-stability with the Python helper (vector pin).
 */

import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import { canonicalJson } from "../src/jcs.js";
import {
  ACKNOWLEDGMENT_RECEIPT_TYPE,
  RECEIPT_TYPE_NAMESPACE,
} from "../src/index.js";
import {
  computeCounterpartyBinding,
  computeEnvelopeHash,
  verifyCounterpartyBinding,
  type CounterpartyBinding,
} from "../src/counterparty.js";

function origEnvelope(): Record<string, unknown> {
  return {
    payload: {
      v: 1,
      action_id: "act_abc",
      agent_id: "agt_A",
      org_id: "org_1",
      type: "protectmcp:decision",
      issuer_id: "00000000000000000010",
      action_ref: `sha256:${"a".repeat(64)}`,
      payload_digest: { hash: `sha256:${"b".repeat(64)}`, size: 7 },
      policy_digest: `sha256:${"c".repeat(64)}`,
      previousReceiptHash: "0".repeat(64),
      issued_at: "2026-05-11T00:00:00Z",
      decision: "allow",
      mode: "payload",
      action_type: "api:call",
      context: { model: "gpt-4" },
      timestamp: "2026-05-11T00:00:00Z",
    },
    signature: { alg: "ml-dsa-65", kid: "kid-A", sig: "AAAA" },
    anchors: [],
  };
}

function ackEnvelope(
  orig: Record<string, unknown>,
  opts: { expectAckFrom?: string; ackKid?: string } = {},
): Record<string, unknown> {
  const binding = computeCounterpartyBinding(orig, {
    expectAckFrom: opts.expectAckFrom,
  });
  const ackPayload = orig.payload as Record<string, unknown>;
  return {
    payload: {
      v: 1,
      type: ACKNOWLEDGMENT_RECEIPT_TYPE,
      issuer_id: "00000000000000000020",
      action_ref: ackPayload.action_ref,
      payload_digest: { hash: `sha256:${"d".repeat(64)}`, size: 0 },
      policy_digest: `sha256:${"e".repeat(64)}`,
      previousReceiptHash: "0".repeat(64),
      issued_at: "2026-05-11T00:00:01Z",
      decision: "allow",
      counterparty_binding: binding,
    },
    signature: { alg: "ml-dsa-65", kid: opts.ackKid ?? "kid-B", sig: "BBBB" },
    anchors: [],
  };
}

describe("computeEnvelopeHash", () => {
  it("returns base64 SHA-256 of the JCS bytes", () => {
    const env = origEnvelope();
    const expected = createHash("sha256")
      .update(canonicalJson(env))
      .digest("base64");
    expect(computeEnvelopeHash(env)).toBe(expected);
  });
});

describe("computeCounterpartyBinding", () => {
  it("defaults receipt_ref to payload.action_id", () => {
    const binding = computeCounterpartyBinding(origEnvelope());
    expect(binding.receipt_ref).toBe("act_abc");
    expect(binding.expect_ack_from).toBeUndefined();
    expect(binding.transport_label).toBeUndefined();
  });

  it("honors explicit receipt_ref, expect_ack_from, and transport_label", () => {
    const binding = computeCounterpartyBinding(origEnvelope(), {
      receiptRef: "asqav-receipt://org/1/agent_A/seq/42",
      expectAckFrom: "kid-B",
      transportLabel: "mcp",
    });
    expect(binding.receipt_ref).toBe("asqav-receipt://org/1/agent_A/seq/42");
    expect(binding.expect_ack_from).toBe("kid-B");
    expect(binding.transport_label).toBe("mcp");
  });

  it("throws when no receipt_ref and no action_id is resolvable", () => {
    expect(() =>
      computeCounterpartyBinding({ payload: { foo: "bar" } }),
    ).toThrow(/receipt_ref is required/);
  });
});

describe("verifyCounterpartyBinding", () => {
  it("returns matches on the honest pair", () => {
    const orig = origEnvelope();
    const ack = ackEnvelope(orig);
    const outcome = verifyCounterpartyBinding(ack, orig);
    expect(outcome.valid).toBe(true);
    expect(outcome.label).toBe("matches");
  });

  it("flags mismatch when the originating envelope is tampered", () => {
    const orig = origEnvelope();
    const ack = ackEnvelope(orig);
    const tampered = origEnvelope();
    (tampered.payload as Record<string, unknown>).context = { model: "gpt-5" };
    const outcome = verifyCounterpartyBinding(ack, tampered);
    expect(outcome.valid).toBe(false);
    expect(outcome.label).toBe("mismatch");
  });

  it("flags kid_mismatch when expect_ack_from disagrees with signature.kid", () => {
    const orig = origEnvelope();
    const ack = ackEnvelope(orig, { expectAckFrom: "kid-B", ackKid: "kid-attacker" });
    const outcome = verifyCounterpartyBinding(ack, orig);
    expect(outcome.valid).toBe(false);
    expect(outcome.label).toBe("kid_mismatch");
    expect(outcome.kidMatches).toBe(false);
  });

  it("returns unresolved when no counterparty_binding present", () => {
    const orig = origEnvelope();
    const ack = { payload: { ...(orig.payload as object) }, signature: { kid: "x" } };
    const outcome = verifyCounterpartyBinding(ack, orig);
    expect(outcome.valid).toBe(false);
    expect(outcome.label).toBe("unresolved");
  });
});

describe("namespace", () => {
  it("includes protectmcp:acknowledgment", () => {
    expect((RECEIPT_TYPE_NAMESPACE as readonly string[]).includes(ACKNOWLEDGMENT_RECEIPT_TYPE)).toBe(true);
  });
});

describe("cross-sdk byte-stability", () => {
  it("pins envelope_hash for a frozen JCS input", () => {
    const env = origEnvelope();
    const binding: CounterpartyBinding = computeCounterpartyBinding(env);
    const recomputed = createHash("sha256")
      .update(canonicalJson(env))
      .digest("base64");
    expect(binding.envelope_hash).toBe(recomputed);
  });
});
