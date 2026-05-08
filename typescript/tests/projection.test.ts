/**
 * TS SDK projection helpers for the three-key Compliance Receipts envelope.
 *
 * `signatureEnvelope(resp)` returns `{alg, kid, sig}` when the response
 * carries the object form, undefined otherwise. `anchors(resp)` returns
 * the cloud-supplied array, undefined for legacy receipts.
 */

import { describe, expect, it } from "vitest";

import {
  type AnchorEntry,
  type SignatureResponse,
  type SignatureEnvelope,
  anchors,
  anchorsFromResponse,
  encodeAnchorValue,
  signatureEnvelope,
  signatureEnvelopeFromResponse,
} from "../src/index.js";

function bareResponse(extra: Partial<SignatureResponse> = {}): SignatureResponse {
  return {
    signature: "ZmFrZQ==",
    signatureId: "sig_test",
    actionId: "act_test",
    timestamp: "2026-05-08T00:00:00Z",
    verificationUrl: "https://verify",
    ...extra,
  };
}

describe("signatureEnvelope()", () => {
  it("returns undefined for legacy receipts (string signature)", () => {
    expect(signatureEnvelope(bareResponse())).toBeUndefined();
  });

  it("returns {alg, kid, sig} when signature is the object form", () => {
    const envelope: SignatureEnvelope = {
      alg: "ML-DSA-65",
      kid: "key_abc",
      sig: "ZmFrZQ==",
    };
    const resp = bareResponse({
      complianceMode: true,
      signature: envelope,
    });
    const out = signatureEnvelope(resp);
    expect(out).toEqual(envelope);
    out!.sig = "MUTATED";
    expect((resp.signature as SignatureEnvelope).sig).toBe("ZmFrZQ==");
  });

  it("free helper accepts dict input", () => {
    const out = signatureEnvelopeFromResponse({
      compliance_mode: true,
      signature: { alg: "Ed25519", kid: "k1", sig: "s" },
    });
    expect(out).toEqual({ alg: "Ed25519", kid: "k1", sig: "s" });
  });

  it("free helper returns undefined for legacy string signature", () => {
    expect(
      signatureEnvelopeFromResponse({ signature: "ZmFrZQ==" }),
    ).toBeUndefined();
  });
});

describe("anchors()", () => {
  it("returns undefined for legacy receipts", () => {
    expect(anchors(bareResponse())).toBeUndefined();
  });

  it("returns the cloud-supplied list when present", () => {
    const cloudAnchors: AnchorEntry[] = [
      { type: "rfc3161", value: "ZmFrZS1kZXI=", tsa: "freetsa.org" },
      { type: "opentimestamps", value: "ZmFrZS1vdHM=", status: "pending" },
    ];
    const resp = bareResponse({
      complianceMode: true,
      anchors: cloudAnchors,
    });
    const out = anchors(resp);
    expect(out).toEqual(cloudAnchors);
    out![0].tsa = "MUTATED";
    expect(resp.anchors![0].tsa).toBe("freetsa.org");
  });

  it("free helper passes through dict input", () => {
    const out = anchorsFromResponse({
      anchors: [{ type: "rfc3161", value: "v", commit_hash: "h" }],
    });
    expect(out).toEqual([{ type: "rfc3161", value: "v", commit_hash: "h" }]);
  });
});

describe("encodeAnchorValue()", () => {
  it("base64-encodes raw bytes", () => {
    expect(encodeAnchorValue(new Uint8Array([104, 101, 108, 108, 111]))).toBe(
      "aGVsbG8=",
    );
  });
});
