// Agent-to-agent receipts: acknowledgments bound to the sender's bytes
// Mirrors python/tests/test_a2a_acknowledge.py

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ACKNOWLEDGMENT_RECEIPT_TYPE,
  Agent,
  AsqavError,
  DEFAULT_RECEIPT_HEADER,
  _resetForTests,
  computeEnvelopeHash,
  digestResultValue,
  encodeReceiptHeader,
  init,
  receiptOf,
  verifyCounterpartyBinding,
} from "../src/index.js";

const PEER = {
  signature_id: "sig_peer",
  payload: { v: 1, type: "protectmcp:decision", action_id: "act_peer", n: 3 },
  signature: { alg: "ML-DSA-65", kid: "kid_peer", sig: "c2lnX2J5dGVz" },
};

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

function agent(id = "agt_b"): Agent {
  return Agent.attach({
    agent_id: id,
    name: id,
    public_key: "pk",
    key_id: "kid",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-10-13T00:00:00Z",
  });
}

function okBody(): Record<string, unknown> {
  return {
    signature: { alg: "ML-DSA-65", kid: "kid_b", sig: "YWNr" },
    signature_id: "sig_ack",
    action_id: "act_ack",
    timestamp: "2026-10-13T00:00:00Z",
    verification_url: "https://verify",
    compliance_mode: true,
    payload: { v: 1, type: ACKNOWLEDGMENT_RECEIPT_TYPE },
  };
}

function sentBody(spy: ReturnType<typeof vi.spyOn>): Record<string, unknown> {
  return JSON.parse((spy.mock.calls[0][1] as RequestInit).body as string);
}

beforeEach(() => {
  _resetForTests();
  init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("agent-to-agent acknowledgment", () => {
  it("binds the peer envelope bytes on an acknowledgment receipt", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await agent().acknowledge(PEER, { expectAckFrom: "kid_b", transportLabel: "https" });
    const body = sentBody(spy);
    expect(String(spy.mock.calls[0][0])).toContain("/agents/agt_b/sign");
    expect(body.receipt_type).toBe(ACKNOWLEDGMENT_RECEIPT_TYPE);
    expect(body.compliance_mode).toBe(true);
    const binding = body.counterparty_binding as Record<string, unknown>;
    expect(binding.receipt_ref).toBe("sig_peer");
    expect(binding.envelope_hash).toBe(computeEnvelopeHash(PEER));
    expect(body.result_digest).toBe(digestResultValue(PEER));
    expect(binding.expect_ack_from).toBe("kid_b");
    expect(binding.transport_label).toBe("https");
  });

  it("produces a binding that verifies against the peer receipt and not a tampered one", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await agent().acknowledge(PEER);
    const binding = sentBody(spy).counterparty_binding;
    const ack = { payload: { counterparty_binding: binding }, signature: { kid: "kid_b" } };
    expect(verifyCounterpartyBinding(ack, PEER).label).toBe("matches");
    const tampered = { ...PEER, payload: { ...PEER.payload, n: 4 } };
    expect(verifyCounterpartyBinding(ack, tampered).label).toBe("mismatch");
  });

  it("needs a receipt ref, explicit or on the envelope", async () => {
    const { signature_id: _drop, ...bare } = PEER;
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await agent().acknowledge(bare, { receiptRef: "sig_given" });
    expect((sentBody(spy).counterparty_binding as Record<string, unknown>).receipt_ref).toBe("sig_given");
    await expect(agent().acknowledge(bare)).rejects.toThrow(/receipt_ref is required/);
  });

  it("fences the binding to acknowledgment receipts before any request", async () => {
    const spy = vi.spyOn(globalThis, "fetch");
    await expect(
      agent().sign({
        actionType: "a2a:send",
        receiptType: "protectmcp:decision",
        counterpartyBinding: { envelope_hash: "x", receipt_ref: "y" },
      }),
    ).rejects.toThrow(AsqavError);
    await expect(
      agent().sign({
        actionType: "a2a:ack",
        receiptType: ACKNOWLEDGMENT_RECEIPT_TYPE,
        counterpartyBinding: { envelope_hash: "x", receipt_ref: "" },
      }),
    ).rejects.toThrow(/counterparty_binding_malformed/);
    expect(spy).not.toHaveBeenCalled();
  });
});

describe("handing a receipt to a peer", () => {
  it("receiptOf returns the signed envelope and its locator", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    const response = await agent("agt_a").sign({ actionType: "a2a:send", context: { to: "agt_b" } });
    expect(receiptOf(response)).toEqual({
      signature_id: "sig_ack",
      payload: { v: 1, type: ACKNOWLEDGMENT_RECEIPT_TYPE },
      signature: { alg: "ML-DSA-65", kid: "kid_b", sig: "YWNr" },
    });
  });

  it("encodes a header the acceptor reads back", () => {
    const header = encodeReceiptHeader(PEER);
    expect(DEFAULT_RECEIPT_HEADER).toBe("x-asqav-receipt");
    expect(/^[A-Za-z0-9+/=]+$/.test(header)).toBe(true);
    expect(JSON.parse(Buffer.from(header, "base64").toString("utf-8"))).toEqual(PEER);
  });
});
