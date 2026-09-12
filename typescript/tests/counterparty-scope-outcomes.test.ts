import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createHash, generateKeyPairSync, sign } from "node:crypto";
import { describe, expect, it } from "vitest";
import { computeCounterpartyBinding, verifyCounterpartyBinding } from "../src/counterparty.js";
import { verifyReceiptOffline } from "../src/index.js";
import { ADAPTERS, AerfAdapter, AsqavNativeAdapter, verify, type ExtraAxis, type KeyProvider } from "../src/verifier/index.js";
import { canonicalJson } from "../src/jcs.js";

const ROOT = resolve(__dirname, "../../verifier/conformance-vectors");
const names = ["asqav-31-counterparty-scope-match", "asqav-32-counterparty-anchors-included",
  "asqav-33-counterparty-scope-absent", "asqav-34-counterparty-scope-unknown"];
const load = (name: string, file: string): Record<string, unknown> => JSON.parse(readFileSync(resolve(ROOT, name, file), "utf8"));
const fixture = (name = names[0]!) => ({ receipt: load(name, "receipt.json"), origin: load(name, "originating_envelope.json"), jwks: load(name, "jwks.json") });

describe("counterparty scope outcomes", () => {
  for (const name of names) it(`public offline API: ${name}`, () => {
    const { receipt, origin, jwks } = fixture(name);
    const expected = load(name, "expected.json");
    const result = verifyReceiptOffline(receipt, jwks, null, origin);
    expect(result.verdict).toBe(expected.outcome);
    expect(result.failureClass).toBe(expected.failure_class ?? null);
    expect(result.axes.find(a => a.axis === "signature")!.result).toBe("PASS");
  });

  it("projects only payload and signature and preserves the caller object", () => {
    const { receipt, origin, jwks } = fixture();
    const before = structuredClone(origin);
    const expected = createHash("sha256").update(canonicalJson({ payload: origin.payload, signature: origin.signature })).digest("base64");
    const binding = computeCounterpartyBinding(origin, { receiptRef: "sig_origin" });
    expect(binding.scope).toBe("envelope_minus_anchors");
    expect(binding.envelope_hash).toBe(expected);
    const upgraded = { ...origin, anchors: [{ type: "rfc3161", value: "changed" }], export_metadata: 17 };
    expect(verifyReceiptOffline(receipt, jwks, null, upgraded).verdict).toBe("verified");
    expect(origin).toEqual(before);
  });

  it("binds signature spelling and accepts both digest alphabets", () => {
    const origin = { payload: { action_id: "orig" }, signature: { alg: "Ed25519", kid: "orig", sig: "+/8=" } };
    const binding = computeCounterpartyBinding(origin);
    const ack = { payload: { counterparty_binding: binding } };
    binding.envelope_hash = Buffer.from(binding.envelope_hash, "base64").toString("base64url");
    expect(verifyCounterpartyBinding(ack, origin).valid).toBe(true);
    const unpadded = binding.envelope_hash;
    binding.envelope_hash += "==";
    expect(verifyCounterpartyBinding(ack, origin).label).toBe("malformed");
    binding.envelope_hash = unpadded;
    const changed = { ...origin, signature: { ...origin.signature, sig: "-_8" } };
    expect(verifyCounterpartyBinding(ack, changed).label).toBe("mismatch");
    ack.payload.counterparty_binding = computeCounterpartyBinding(changed);
    expect(verifyCounterpartyBinding(ack, changed).valid).toBe(true);
  });

  for (const scope of [undefined, null, "unknown"]) for (const origin of [null, {}]) {
    it(`dispatches scope ${scope} before malformed fields with origin ${JSON.stringify(origin)}`, () => {
      const binding: Record<string, unknown> = { envelope_hash: [] };
      if (scope !== undefined) binding.scope = scope;
      const result = verifyCounterpartyBinding({ payload: { counterparty_binding: binding } }, origin);
      expect(result.valid).toBeNull();
      expect(result.label).toBe(scope === undefined ? "legacy_scope" : "unrecognised_scope");
    });
  }

  for (const binding of [null, [], false, "bad"]) it(`rejects a present malformed binding ${binding}`, () => {
    expect(verifyCounterpartyBinding({ payload: { counterparty_binding: binding } }).label).toBe("malformed");
  });

  it("does not leak context across concurrent or consecutive calls", async () => {
    const { receipt, origin, jwks } = fixture();
    const results = await Promise.all(Array.from({ length: 12 }, async (_, index) =>
      verifyReceiptOffline(receipt, jwks, null, index % 2 === 0 ? origin : null).verdict));
    expect(results).toEqual(Array.from({ length: 12 }, (_, index) => index % 2 === 0 ? "verified" : "unverified"));
    expect(verifyReceiptOffline(receipt, jwks).failureClass).toBe("unverifiable");
  });

  for (const origin of [null, {}, { payload: [] }, { payload: {}, signature: {} }]) {
    it(`fails cleanly for unavailable origin ${JSON.stringify(origin)}`, () => {
      const { receipt, jwks } = fixture();
      expect(verifyReceiptOffline(receipt, jwks, null, origin).failureClass).toBe("unverifiable");
    });
  }

  it("rejects a deep origin after scope dispatch", () => {
    const { receipt, origin, jwks } = fixture();
    let deep: Record<string, unknown> = {};
    for (let i = 0; i < 205; i++) deep = { nested: deep };
    (origin.payload as Record<string, unknown>).deep = deep;
    expect(verifyReceiptOffline(receipt, jwks, null, origin).failureClass).toBe("unverifiable");
    const absent = fixture(names[2]);
    const explodingOrigin = { get payload(): never { throw new Error("scope must dispatch first"); } };
    const result = verifyReceiptOffline(absent.receipt, absent.jwks, null, explodingOrigin);
    expect(result.axes.find(a => a.axis === "counterparty")!.note).toContain("legacy_scope");
  });

  it("retains an invalid signature beside unknown scope", () => {
    const { receipt, origin, jwks } = fixture(names[2]);
    (receipt.payload as Record<string, unknown>).decision = "deny";
    expect(verifyReceiptOffline(receipt, jwks, null, origin).failureClass).toBe("invalid");
  });

  it("preserves two-argument subclass hooks and foreign-format behavior", () => {
    class Compatible extends AerfAdapter {
      extraAxes(_doc: Record<string, unknown>, _keyProvider: Record<string, unknown> | null): ExtraAxis[] {
        return [["compat", "PASS", "two-argument hook called"]];
      }
    }
    const foreign = load("aerf-01-genesis", "receipt.json");
    const context = Object.freeze({ originatingEnvelope: { get payload(): never { throw new Error("foreign adapter read origin"); } } });
    expect(verify(foreign, [new Compatible()], null, null, context).axes.find(a => a.axis === "compat")!.result).toBe("PASS");
    const manifest = JSON.parse(readFileSync(resolve(ROOT, "manifest.json"), "utf8")) as { dir: string; format: string }[];
    for (const adapter of ADAPTERS.filter(a => a.name !== "asqav-native")) {
      const entry = manifest.find(e => e.format === adapter.name)!;
      const receipt = load(entry.dir, "receipt.json");
      const plain = verify(receipt, [adapter]);
      const contextual = verify(receipt, [adapter], null, null, context);
      expect(contextual.verdict).toBe(plain.verdict);
      expect(contextual.axes.map(a => [a.axis, a.result])).toEqual(plain.axes.map(a => [a.axis, a.result]));
    }
  });

  it.each([["custom_guard", "FAIL"], ["counterparty", "FAIL"], ["counterparty", "SKIPPED"]] as const)("preserves a native subclass guard %s/%s with and without context", (axisName, status) => {
    class Guarded extends AsqavNativeAdapter {
      extraAxes(doc: Record<string, unknown>, provider: KeyProvider): ExtraAxis[] {
        return [...super.extraAxes(doc, provider).filter(axis => axis[0] !== axisName), [axisName, status, "subclass guard"]];
      }
    }
    const { receipt, origin, jwks } = fixture();
    for (const context of [{}, { originatingEnvelope: origin }]) {
      const result = verify(receipt, [new Guarded()], jwks, null, context);
      expect(result.axes.some(a => a.axis === axisName)).toBe(true);
      expect(result.axes.find(a => a.axis === axisName)!.result).toBe(status);
      expect(result.axes.find(a => a.axis === axisName)!.note).toBe("subclass guard");
      expect(result.verdict).toBe("unverified");
    }
  });

  it("uses the actual kid in a hosted signature_envelope", () => {
    const { receipt, origin, jwks } = fixture();
    receipt.signature_envelope = receipt.signature;
    receipt.signature = (receipt.signature_envelope as Record<string, unknown>).sig;
    expect(verifyReceiptOffline(receipt, jwks, null, origin).verdict).toBe("verified");
    delete receipt.signature_envelope;
    const axis = verifyReceiptOffline(receipt, jwks, null, origin).axes.find(a => a.axis === "counterparty")!;
    expect(axis.result).toBe("FAIL");
    expect(axis.note).toContain("kid_mismatch:");
  });

  it("keeps a missing signing key uncertain beside a matching byte binding", () => {
    const { receipt, origin } = fixture();
    const result = verifyReceiptOffline(receipt, { keys: [] }, null, origin);
    expect(result.axes.find(a => a.axis === "counterparty")!.result).toBe("PASS");
    expect(result.axes.find(a => a.axis === "signature")!.result).toBe("SKIPPED");
    expect(result.verdict).toBe("unverified");
    expect(result.failureClass).toBe("unverifiable");
  });

  it.each([["fixture-b", "FAIL"], ["fixture-a", "PASS"], ["fixture-key-a", "PASS"]])("binds acknowledging kid %s to the resolved key", (kid, status) => {
    const { receipt, origin } = fixture();
    const { privateKey, publicKey } = generateKeyPairSync("ed25519");
    const payload = receipt.payload as Record<string, unknown>;
    Object.assign(payload, { issuer_id: "fixture-a", agent_id: "fixture-agent-a" });
    (payload.counterparty_binding as Record<string, unknown>).expect_ack_from = kid;
    receipt.signature = { alg: "Ed25519", kid, sig: sign(null, Buffer.from(canonicalJson(payload)), privateKey).toString("base64") };
    const jwks = { keys: [{ kid: "fixture-key-a", issuer_id: "fixture-a", agent_id: "fixture-agent-a",
      alg: "Ed25519", status: "active", public_key: publicKey.export({ type: "spki", format: "der" }).subarray(-32).toString("base64") }] };
    const result = verifyReceiptOffline(receipt, jwks, null, origin);
    expect(result.axes.find(a => a.axis === "signature")!.result).toBe("PASS");
    expect(result.axes.find(a => a.axis === "counterparty")!.result).toBe(status);
  });
});
