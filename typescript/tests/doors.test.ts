// Doors: one receipt, N standard envelopes, byte-identical inner across all.
// Pins per door: valid shape + exact receipt, round-trip inverse, and a parity
// golden the Python test also checks against the same file.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { canonicalJson } from "../src/jcs.js";
import * as doors from "../src/doors.js";

const PARITY = resolve(__dirname, "..", "..", "verifier", "doors-parity");

function load(name: string): doors.Receipt {
  return JSON.parse(readFileSync(resolve(PARITY, `receipt-${name}.json`), "utf-8"));
}

function jcs(obj: unknown): string {
  return new TextDecoder().decode(canonicalJson(obj));
}

const COMPLIANCE = load("compliance");
const HASHMODE = load("hashmode");

describe("doors byte-identical inner", () => {
  it("all five doors carry the same receipt with an identical JCS string", () => {
    for (const receipt of [COMPLIANCE, HASHMODE]) {
      const inners = [
        doors.receiptFromVc(doors.toW3cVc(receipt)),
        doors.receiptFromCloudEvent(doors.toCloudEvent(receipt)),
        doors.receiptFromOtelGenaiAttributes(doors.toOtelGenaiAttributes(receipt)),
        doors.receiptFromC2paAssertion(doors.toC2paAssertion(receipt)),
        doors.receiptFromErc8004ValidationRequest(doors.toErc8004ValidationRequest(receipt)),
      ];
      const canon = new Set(inners.map(jcs));
      expect(canon.size).toBe(1);
      expect(canon.has(jcs(receipt))).toBe(true);
    }
  });
});

describe("W3C VC door", () => {
  it("shape and round-trip", () => {
    const vc = doors.toW3cVc(COMPLIANCE);
    expect((vc["@context"] as string[])[0]).toBe("https://www.w3.org/ns/credentials/v2");
    expect(vc.type).toEqual(["VerifiableCredential", "AsqavReceiptCredential"]);
    expect(vc.credentialSubject).toEqual(COMPLIANCE);
    const proof = vc.proof as doors.Receipt;
    expect(proof.type).toBe("AsqavReceiptSignature2026");
    expect(proof.signatureValue).toBe((COMPLIANCE.signature as doors.Receipt).sig);
    expect(proof.receiptDigest).toBe(doors.receiptDigest(COMPLIANCE));
    expect(vc.validFrom).toBe((COMPLIANCE.payload as doors.Receipt).issued_at);
    expect(doors.receiptFromVc(vc)).toEqual(COMPLIANCE);
  });

  it("hash-mode signature comes from flat fields", () => {
    const vc = doors.toW3cVc(HASHMODE);
    const proof = vc.proof as doors.Receipt;
    expect(proof.signatureValue).toBe(HASHMODE.signature_b64);
    expect(proof.signatureAlgorithm).toBe(HASHMODE.algorithm);
    expect(doors.receiptFromVc(vc)).toEqual(HASHMODE);
  });
});

describe("CloudEvents door", () => {
  it("shape and round-trip", () => {
    const ce = doors.toCloudEvent(COMPLIANCE);
    expect(ce.specversion).toBe("1.0");
    expect(ce.type).toBe("com.asqav.receipt.v1");
    expect(ce.datacontenttype).toBe("application/json");
    expect(ce.asqavreceiptdigest).toBe(doors.receiptDigest(COMPLIANCE));
    expect(ce.time).toBe((COMPLIANCE.payload as doors.Receipt).issued_at);
    expect(ce.data).toEqual(COMPLIANCE);
    expect(doors.receiptFromCloudEvent(ce)).toEqual(COMPLIANCE);
  });
});

describe("OTel GenAI door", () => {
  it("gen_ai keys and round-trip", () => {
    const attrs = doors.toOtelGenaiAttributes(COMPLIANCE);
    const p = COMPLIANCE.payload as doors.Receipt;
    expect(attrs["gen_ai.operation.name"]).toBe(p.type);
    expect(attrs["gen_ai.agent.id"]).toBe(p.agent_id);
    expect(attrs["gen_ai.tool.name"]).toBe(p.tool_name);
    expect(attrs["asqav.receipt.digest"]).toBe(doors.receiptDigest(COMPLIANCE));
    for (const v of Object.values(attrs)) {
      expect(["string", "number", "boolean"]).toContain(typeof v);
    }
    expect(doors.receiptFromOtelGenaiAttributes(attrs)).toEqual(COMPLIANCE);
  });

  it("hash-mode tool call id", () => {
    const attrs = doors.toOtelGenaiAttributes(HASHMODE);
    expect(attrs["gen_ai.tool.call.id"]).toBe(HASHMODE.action_id);
    expect(doors.receiptFromOtelGenaiAttributes(attrs)).toEqual(HASHMODE);
  });
});

describe("C2PA door", () => {
  it("raw-bytes binding and round-trip", () => {
    const a = doors.toC2paAssertion(COMPLIANCE);
    expect(a.label).toBe("com.asqav.receipt.v1");
    const binding = a.hash_binding as doors.Receipt;
    expect(binding.rule).toBe("raw-bytes");
    const raw = Buffer.from(doors.canonicalReceiptBytes(COMPLIANCE));
    expect(binding.hash).toBe(createHash("sha256").update(raw).digest("hex"));
    expect(doors.receiptFromC2paAssertion(a)).toEqual(COMPLIANCE);
  });
});

describe("ERC-8004 door", () => {
  it("validationRequest and round-trip", () => {
    const req = doors.toErc8004ValidationRequest(COMPLIANCE, { validator: "0xabc" });
    expect(req.function).toBe("validationRequest");
    const args = req.args as doors.Receipt;
    expect(args.validator).toBe("0xabc");
    expect(args.agentId).toBe((COMPLIANCE.payload as doors.Receipt).agent_id);
    expect((args.receiptCommitment as string).length).toBe(66);
    expect(req.hashAlgorithm).toBe("asqav-commitment-sha256");
    expect(doors.receiptFromErc8004ValidationRequest(req)).toEqual(COMPLIANCE);
  });
});

describe("doors purity and generic extract", () => {
  it("does not mutate the input", () => {
    const before = jcs(COMPLIANCE);
    doors.wrapAll(COMPLIANCE);
    expect(jcs(COMPLIANCE)).toBe(before);
  });

  it("generic extract recovers the receipt from every door", () => {
    const envs = doors.wrapAll(COMPLIANCE) as Record<string, doors.Receipt>;
    for (const key of ["w3c_vc", "cloudevent", "otel_genai", "c2pa", "erc8004"]) {
      expect(doors.extractReceipt(envs[key])).toEqual(COMPLIANCE);
    }
  });
});

describe("cross-language parity", () => {
  it("TS wrapAll JCS equals the golden the Python test also checks", () => {
    for (const name of ["compliance", "hashmode"]) {
      const golden = readFileSync(resolve(PARITY, `golden-${name}.jcs`), "utf-8");
      const receipt = load(name);
      expect(jcs(doors.wrapAll(receipt))).toBe(golden);
    }
  });
});
