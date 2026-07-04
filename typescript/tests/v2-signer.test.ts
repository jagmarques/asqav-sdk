// v:2 receipt acceptance in the neutral oracle verifier (TS parity with Python).
// Verifies the signature over the body that includes signer + _asqav_tid, surfaces
// signer, and FAILs a tampered in-body signer. Byte-pinned to the backend v:2 body.

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { verify } from "../src/verifier/core.js";
import { ADAPTERS, AsqavNativeAdapter } from "../src/verifier/index.js";

const CORPUS_ROOT = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");
const SIGNER = "https://api.asqav.com";

function load(vec: string, name: string): Record<string, unknown> {
  return JSON.parse(readFileSync(join(CORPUS_ROOT, vec, name), "utf-8")) as Record<string, unknown>;
}

function v2(): { receipt: Record<string, unknown>; jwks: Record<string, unknown> } {
  return {
    receipt: load("asqav-08-v2-signer-canary", "receipt.json"),
    jwks: load("asqav-08-v2-signer-canary", "jwks.json"),
  };
}

describe("v:2 signer + canary acceptance (neutral verifier)", () => {
  it("verifies a v:2 receipt and surfaces signer", () => {
    const { receipt, jwks } = v2();
    const res = verify(receipt, ADAPTERS, jwks);
    expect(res.fmt).toBe("asqav-native");
    expect(res.verdict).toBe("PASS");
    expect(res.axes.find((a) => a.axis === "signature")?.result).toBe("PASS");
    expect(res.signer).toBe(SIGNER);
  });

  it("fails when the in-body signer is tampered (proves in-body coverage)", () => {
    const { receipt, jwks } = v2();
    (receipt.payload as Record<string, unknown>).signer = "https://api.asqav.con";
    const res = verify(receipt, ADAPTERS, jwks);
    expect(res.axes.find((a) => a.axis === "signature")?.result).toBe("FAIL");
    expect(res.verdict).toBe("FAIL");
  });

  it("does not surface or trust a signer moved outside the signed body", () => {
    const { receipt, jwks } = v2();
    delete (receipt.payload as Record<string, unknown>).signer;
    receipt.signer = SIGNER; // loose, outside the canonical body
    const res = verify(receipt, ADAPTERS, jwks);
    expect(res.signer).toBeNull();
    expect(res.verdict).toBe("FAIL");
  });

  it("fails the tampered-canary corpus vector", () => {
    const receipt = load("asqav-09-v2-signer-tampered", "receipt.json");
    const jwks = load("asqav-09-v2-signer-tampered", "jwks.json");
    const res = verify(receipt, ADAPTERS, jwks);
    expect(res.verdict).toBe("FAIL");
  });

  it("does not add a canary axis (neutral verifier cannot recompute _asqav_tid)", () => {
    const { receipt, jwks } = v2();
    const res = verify(receipt, ADAPTERS, jwks);
    expect(res.axes.some((a) => a.axis === "canary")).toBe(false);
  });

  it("still verifies a v:1 receipt and exposes no signer", () => {
    const receipt = load("asqav-01-genesis-permit", "receipt.json");
    const jwks = load("asqav-01-genesis-permit", "jwks.json");
    const res = verify(receipt, ADAPTERS, jwks);
    expect(res.verdict).toBe("PASS");
    expect(res.signer).toBeNull();
  });

  it("attestation() reads signer only from the signed payload", () => {
    const ad = new AsqavNativeAdapter();
    const { receipt } = v2();
    expect(ad.attestation(receipt)).toEqual({ signer: SIGNER });
    const loose = JSON.parse(JSON.stringify(receipt)) as Record<string, unknown>;
    delete (loose.payload as Record<string, unknown>).signer;
    loose.signer = SIGNER;
    expect(ad.attestation(loose)).toEqual({});
  });
});
