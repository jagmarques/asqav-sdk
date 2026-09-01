/**
 * W3C VC 2.0 eddsa-jcs-2022 adapter, TS half of test_oracle_w3c_vc.py's negatives. Corpus
 * verdicts are gated in verifier-parity.test.ts; this pins the axis-level behaviour.
 */

import { readFileSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { FAIL, PASS, SKIPPED } from "../src/verifier/crypto.js";
import { verify, FAILURE_INVALID, FAILURE_UNVERIFIABLE } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { W3cVcAdapter } from "../src/verifier/adapters/w3cVc.js";
import { AgentReceiptsAdapter } from "../src/verifier/adapters/agentreceipts.js";
import { resolveEd25519Key } from "../src/verifier/did.js";
import { parseJsonPreservingFloats } from "../src/verifier/canonical.js";

const CORPUS_ROOT = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");

function load(vec: string, name: string): Record<string, unknown> {
  return parseJsonPreservingFloats(readFileSync(join(CORPUS_ROOT, vec, name), "utf-8")) as Record<
    string,
    unknown
  >;
}

function provider(vec: string): Record<string, unknown> | null {
  const path = join(CORPUS_ROOT, vec, "did_map.json");
  return existsSync(path) ? (JSON.parse(readFileSync(path, "utf-8")) as Record<string, unknown>) : null;
}

const b58 = (data: Uint8Array): string => {
  const ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  let num = 0n;
  for (const byte of data) num = num * 256n + BigInt(byte);
  let out = "";
  while (num > 0n) {
    out = ALPHABET[Number(num % 58n)] + out;
    num /= 58n;
  }
  let pad = 0;
  for (const byte of data) {
    if (byte === 0) pad++;
    else break;
  }
  return "1".repeat(pad) + out;
};

describe("w3c-vc corpus verdicts (axis-level)", () => {
  it("w3c-vc-01 did:web happy path verifies offline", () => {
    const res = verify(load("w3c-vc-01-didweb-happy-path", "receipt.json"), ADAPTERS, provider("w3c-vc-01-didweb-happy-path"));
    expect(res.fmt).toBe("w3c-vc");
    expect(res.verdict).toBe("verified");
    expect(res.axes.find((a) => a.axis === "signature")?.result).toBe(PASS);
    expect(res.axes.find((a) => a.axis === "expiry")?.result).toBe(PASS);
  });

  it("w3c-vc-05 no injected DID document fails closed (never fetches)", () => {
    const res = verify(load("w3c-vc-05-no-did-document", "receipt.json"), ADAPTERS);
    expect(res.axes.find((a) => a.axis === "signature")?.result).toBe(SKIPPED);
    expect(res.verdict).toBe("unverified");
    expect(res.failureClass).toBe(FAILURE_UNVERIFIABLE);
  });

  it("w3c-vc-06 expired keeps the verified verdict (criterion 426)", () => {
    const res = verify(load("w3c-vc-06-expired", "receipt.json"), ADAPTERS, provider("w3c-vc-06-expired"));
    expect(res.verdict).toBe("verified");
    const expiry = res.axes.find((a) => a.axis === "expiry");
    expect(expiry?.result).toBe(FAIL);
    expect(expiry?.note.startsWith("expired at ")).toBe(true);
  });
});

describe("w3c-vc structure negatives", () => {
  const ad = new W3cVcAdapter();

  it("rejects proof sets", () => {
    const doc = load("w3c-vc-01-didweb-happy-path", "receipt.json");
    doc.proof = [doc.proof];
    expect(ad.detect(doc)).toBe(true);
    const [result, note] = ad.schema(doc);
    expect(result).toBe(FAIL);
    expect(note).toContain("proof sets are not supported");
    const res = verify(doc, ADAPTERS, provider("w3c-vc-01-didweb-happy-path"));
    expect(res.verdict).toBe("unverified");
  });

  it("reports a sibling cryptosuite as an algorithm mismatch (invalid)", () => {
    const doc = load("w3c-vc-01-didweb-happy-path", "receipt.json");
    (doc.proof as Record<string, unknown>).cryptosuite = "eddsa-rdfc-2022";
    expect(ad.detect(doc)).toBe(true);
    const [result, note] = ad.schema(doc);
    expect(result).toBe(FAIL);
    expect(note.startsWith("unsupported signature algorithm")).toBe(true);
    const res = verify(doc, ADAPTERS, provider("w3c-vc-01-didweb-happy-path"));
    expect(res.verdict).toBe("unverified");
    expect(res.failureClass).toBe(FAILURE_INVALID);
  });

  it("requires proofPurpose assertionMethod", () => {
    const doc = load("w3c-vc-01-didweb-happy-path", "receipt.json");
    (doc.proof as Record<string, unknown>).proofPurpose = "authentication";
    const [result, note] = ad.schema(doc);
    expect(result).toBe(FAIL);
    expect(note).toContain("proofPurpose");
  });

  it("rejects a malformed proofValue without crashing", () => {
    const doc = load("w3c-vc-01-didweb-happy-path", "receipt.json");
    for (const bad of ["u-not-base58btc", "", { k: "v" }, null]) {
      const forged = JSON.parse(JSON.stringify(doc)) as Record<string, unknown>;
      (forged.proof as Record<string, unknown>).proofValue = bad;
      const res = verify(forged, ADAPTERS, provider("w3c-vc-01-didweb-happy-path"));
      expect(res.verdict).toBe("unverified");
      expect(ad.extractSignature(forged).sig.length).toBe(0);
    }
  });
});

describe("offline DID-document resolution (the did:web path)", () => {
  const DID = "did:web:example.com";
  const keyA = new Uint8Array(32).fill(1);
  const keyB = new Uint8Array(32).fill(2);

  function docWith(...methods: Array<Record<string, unknown>>): Record<string, unknown> {
    return { id: DID, verificationMethod: methods, assertionMethod: [] };
  }

  it("resolves a fragment-matched publicKeyMultibase Multikey", () => {
    const doc = docWith({
      id: `${DID}#key-1`,
      publicKeyMultibase: "z" + b58(Uint8Array.from([0xed, 0x01, ...keyA])),
    });
    const [key, note] = resolveEd25519Key(`${DID}#key-1`, { [DID]: doc });
    expect(key).toEqual(keyA);
    expect(note).toContain("injected DID document");
  });

  it("fails closed on a missing fragment", () => {
    const doc = docWith({ id: `${DID}#other`, publicKeyBase58: b58(keyA) });
    const [key, note] = resolveEd25519Key(`${DID}#key-1`, { [DID]: doc });
    expect(key).toBeNull();
    expect(note).toContain("no verificationMethod");
  });

  it("prefers the assertionMethod-referenced key", () => {
    const doc = docWith(
      { id: `${DID}#key-a`, publicKeyBase58: b58(keyA) },
      { id: `${DID}#key-b`, publicKeyBase58: b58(keyB) },
    );
    doc.assertionMethod = [`${DID}#key-b`];
    const [key] = resolveEd25519Key(DID, { [DID]: doc });
    expect(key).toEqual(keyB);
  });

  it("resolves a publicKeyJwk OKP/Ed25519 key", () => {
    const x = Buffer.from(keyA).toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
    const doc = docWith({ id: `${DID}#jwk`, publicKeyJwk: { kty: "OKP", crv: "Ed25519", x } });
    const [key] = resolveEd25519Key(`${DID}#jwk`, { [DID]: doc });
    expect(key).toEqual(keyA);
  });

  it("fails closed on a non-Ed25519 multikey (secp256k1 prefix)", () => {
    const doc = docWith({
      id: `${DID}#k`,
      publicKeyMultibase: "z" + b58(Uint8Array.from([0xe7, 0x01, ...keyA])),
    });
    const [key, note] = resolveEd25519Key(`${DID}#k`, { [DID]: doc });
    expect(key).toBeNull();
    expect(note).toContain("no Ed25519 verificationMethod");
  });

  it("keeps the raw-hex injection shape backwards compatible", () => {
    const [key, note] = resolveEd25519Key("did:agent:x#k1", {
      "did:agent:x": Buffer.from(keyA).toString("hex"),
    });
    expect(key).toEqual(keyA);
    expect(note).toContain("injected map");
  });
});

describe("detection exclusion against the sibling VC format", () => {
  it("w3c-vc and agentreceipts never claim each other's receipts", () => {
    const vc = load("w3c-vc-01-didweb-happy-path", "receipt.json");
    const ar = load("agentreceipts-01-didkey-genesis", "receipt.json");
    const w = new W3cVcAdapter();
    const g = new AgentReceiptsAdapter();
    expect([w.detect(vc), g.detect(vc)]).toEqual([true, false]);
    expect([w.detect(ar), g.detect(ar)]).toEqual([false, true]);
  });
});
