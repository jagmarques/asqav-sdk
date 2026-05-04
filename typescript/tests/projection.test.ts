/**
 * IETF -01 N6 + N7 / T2: TS SDK projection helpers.
 *
 * `signatureObject(resp)` projects to `{alg, kid, sig}` per Section 4.
 * `anchors(resp)` projects to the spec-shape array per Section 4.4.
 * Both gated on `complianceMode=true`; legacy receipts get undefined
 * back so the additive overlay stays byte-stable.
 */

import { describe, expect, it } from "vitest";

import {
  type AnchorEntry,
  type SignatureResponse,
  type SignatureEnvelope,
  anchors,
  anchorsFromResponse,
  encodeAnchorValue,
  signatureObject,
  signatureObjectFromResponse,
} from "../src/index.js";

function bareResponse(extra: Partial<SignatureResponse> = {}): SignatureResponse {
  return {
    signature: "ZmFrZQ==",
    signatureId: "sig_test",
    actionId: "act_test",
    timestamp: "2026-05-04T00:00:00Z",
    verificationUrl: "https://verify",
    ...extra,
  };
}

describe("signatureObject()", () => {
  it("returns undefined for non-compliance receipts", () => {
    expect(signatureObject(bareResponse())).toBeUndefined();
  });

  it("returns the cloud-supplied envelope when present", () => {
    const envelope: SignatureEnvelope = {
      alg: "ML-DSA-65",
      kid: "key_abc",
      sig: "ZmFrZQ==",
    };
    const resp = bareResponse({
      complianceMode: true,
      signatureObject: envelope,
    });
    const out = signatureObject(resp);
    expect(out).toEqual(envelope);
    // Mutation safety: returned object is a fresh copy.
    out!.sig = "MUTATED";
    expect(resp.signatureObject!.sig).toBe("ZmFrZQ==");
  });

  it("projects from flat fields when complianceMode but no envelope", () => {
    const resp = bareResponse({
      algorithm: "ML-DSA-65",
      complianceMode: true,
    });
    const out = signatureObject(resp);
    expect(out).toEqual({
      alg: "ML-DSA-65",
      kid: "",
      sig: "ZmFrZQ==",
    });
  });

  it("free helper accepts dict input (snake_case wire shape)", () => {
    const out = signatureObjectFromResponse({
      compliance_mode: true,
      signatureObject: { alg: "Ed25519", kid: "k1", sig: "s" },
    });
    expect(out).toEqual({ alg: "Ed25519", kid: "k1", sig: "s" });
  });
});

describe("anchors()", () => {
  it("returns undefined for non-compliance receipts", () => {
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
    // Mutation safety: each entry is a copy.
    out![0].tsa = "MUTATED";
    expect(resp.anchors![0].tsa).toBe("freetsa.org");
  });

  it("returns [] under complianceMode when no anchor columns set", () => {
    const out = anchors(bareResponse({ complianceMode: true }));
    expect(out).toEqual([]);
  });

  it("projects rfc3161 from flat snake_case fields via free helper", () => {
    const out = anchorsFromResponse({
      compliance_mode: true,
      rfc3161_serial: "0xabc",
      rfc3161_tsa: "freetsa.org",
    });
    expect(out).toHaveLength(1);
    expect(out![0]).toEqual({
      type: "rfc3161",
      value: "",
      serial: "0xabc",
      tsa: "freetsa.org",
    });
  });

  it("projects opentimestamps from bitcoinAnchor.status via free helper", () => {
    const out = anchorsFromResponse({
      compliance_mode: true,
      bitcoinAnchor: { status: "pending" },
    });
    expect(out!.some((e) => e.type === "opentimestamps")).toBe(true);
  });
});

describe("encodeAnchorValue()", () => {
  it("base64-encodes raw bytes", () => {
    expect(encodeAnchorValue(new Uint8Array([104, 101, 108, 108, 111]))).toBe(
      "aGVsbG8=",
    );
  });
});
