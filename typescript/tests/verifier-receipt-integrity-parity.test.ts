/**
 * Receipt-internal integrity axes, TS half, reading the same axis-parity-cases.json as Python
 * so a rule drifting in one language fails a suite. Covers payload_digest and counterparty_binding.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createHash, generateKeyPairSync, sign } from "node:crypto";

import {
  checkCounterpartyBinding,
  checkPayloadDigest,
} from "../src/verifier/vrShim.js";
import { asqavJcs } from "../src/verifier/canonical.js";
import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";

const CASES_FILE = resolve(__dirname, "..", "..", "verifier", "axis-parity-cases.json");
const TABLE = JSON.parse(readFileSync(CASES_FILE, "utf-8"));

const ALG = "ML-DSA-65";
const KEY = new Uint8Array(1952).fill(0);
const CTX = { amount: 100, currency: "EUR" };
const HONEST = createHash("sha256").update(Buffer.from(asqavJcs(CTX))).digest("hex");

function jwks(): Record<string, unknown> {
  return {
    keys: [
      {
        kid: "iss_1",
        issuer_id: "iss_1",
        agent_id: "ag_1",
        org_id: "org_1",
        alg: ALG,
        public_key: Buffer.from(KEY).toString("base64"),
        status: "active",
      },
    ],
  };
}

function receipt(extra: Record<string, unknown> = {}): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    v: 1,
    type: "protectmcp:decision",
    issuer_id: "iss_1",
    agent_id: "ag_1",
    org_id: "org_1",
    previousReceiptHash: "0".repeat(64),
    issued_at: "2026-08-31T10:00:00Z",
    action_ref: "a".repeat(64),
    context: CTX,
    payload_digest: { hash: HONEST, size: asqavJcs(CTX).length },
    policy_digest: "pd",
    decision: "allow",
    ...extra,
  };
  return {
    payload,
    signature: { alg: ALG, kid: "iss_1", sig: Buffer.alloc(64).toString("base64") },
    anchors: {},
  };
}

describe("receipt-internal integrity parity", () => {
  it("reads a populated shared table", () => {
    expect(TABLE.payload_digest.length).toBeGreaterThanOrEqual(10);
    expect(TABLE.counterparty_binding.length).toBeGreaterThanOrEqual(8);
  });

  for (const c of TABLE.payload_digest) {
    it(`payload_digest: ${c.name}`, () => {
      const [result, note] = checkPayloadDigest(c.payload);
      expect(result, note).toBe(c.expect.result);
      expect(note).toContain(c.expect.note_contains);
    });
  }

  for (const c of TABLE.counterparty_binding) {
    it(`counterparty: ${c.name}`, () => {
      const kid = c.acknowledging_kid ?? c.payload.issuer_id ?? null;
      const [result, note] = checkCounterpartyBinding(c.payload, c.originating_envelope, kid);
      expect(result, note).toBe(c.expect.result);
      expect(note).toContain(c.expect.note_contains);
      expect(result === "PASS" ? true : result === "FAIL" ? false : null).toBe(c.expect.valid);
      const doc = receipt(c.payload);
      (doc.signature as Record<string, unknown>).kid = kid;
      const provider = jwks();
      Object.assign((provider.keys as Record<string, unknown>[])[0]!, { kid, issuer_id: (doc.payload as Record<string, unknown>).issuer_id });
      const axis = verify(doc, ADAPTERS, provider, null, { originatingEnvelope: c.originating_envelope }).axes.find(a => a.axis === "counterparty")!;
      expect(axis.result).toBe(result);
      expect(axis.failureClass).toBe(c.expect.failure_class);
    });
  }

  // Deleting any of the three adapter pushes fails here.
  it("emits payload_digest, counterparty and skew from the oracle", () => {
    const axes = new Set(verify(receipt(), ADAPTERS, jwks()).axes.map((a) => a.axis));
    for (const name of ["payload_digest", "counterparty", "skew"]) {
      expect(axes.has(name), `missing ${name}; got ${[...axes].sort().join(",")}`).toBe(true);
    }
  });

  // Removing payload_digest from INVALID_FAIL_AXES fails here.
  it("treats a lying payload_digest as terminal invalid", () => {
    const r = verify(
      receipt({ payload_digest: { hash: "f".repeat(64), size: 31 } }),
      ADAPTERS,
      jwks(),
    );
    const axis = r.axes.find((a) => a.axis === "payload_digest")!;
    expect(axis.result).toBe("FAIL");
    expect(axis.failureClass).toBe("invalid");
    expect(axis.note).toContain("payload_digest_mismatch");
    expect(r.verdict).toBe("unverified");
  });

  it("never lets a fabricated counterparty binding read as corroborated", () => {
    const forged = { scope: "envelope_minus_anchors", receipt_ref: "sig_NEVER_EXISTED", envelope_hash: Buffer.alloc(32).toString("base64") };
    const axis = verify(receipt({ counterparty_binding: forged }), ADAPTERS, jwks()).axes.find(
      (a) => a.axis === "counterparty",
    )!;
    expect(axis.result).toBe("SKIPPED");
    expect(axis.failureClass).toBe("unverifiable");
    expect(axis.note).toContain("unresolved:");
  });

  it("refuses a postdated receipt", () => {
    const axis = verify(receipt({ issued_at: "2099-01-01T00:00:00Z" }), ADAPTERS, jwks()).axes.find(
      (a) => a.axis === "skew",
    )!;
    expect(axis.result).toBe("FAIL");
    expect(axis.failureClass).toBe("invalid");
  });

  // Absence must PASS: a skip blocks every axis but chain.
  it("does not block when neither claim is present", () => {
    const doc = receipt();
    const payload = doc.payload as Record<string, unknown>;
    delete payload.counterparty_binding;
    delete payload.payload_digest;
    const r = verify(doc, ADAPTERS, jwks());
    for (const name of ["payload_digest", "counterparty"]) {
      const axis = r.axes.find((a) => a.axis === name)!;
      expect(axis.result, `${name}: ${axis.note}`).toBe("PASS");
      expect(axis.failureClass).toBeNull();
    }
  });

  // Genuinely Ed25519-signed over its final payload (criterion 727)
  function genuinelySigned(digest: Record<string, unknown>): {
    doc: Record<string, unknown>;
    provider: Record<string, unknown>;
  } {
    const doc = receipt({ payload_digest: digest });
    const payload = doc.payload as Record<string, unknown>;
    const { privateKey, publicKey } = generateKeyPairSync("ed25519");
    const sig = sign(null, Buffer.from(asqavJcs(payload)), privateKey).toString("base64");
    doc.signature = { alg: "Ed25519", kid: "ed_1", sig };
    const exported = publicKey.export({ format: "der", type: "spki" }) as Buffer;
    const raw = exported.subarray(-32).toString("base64");
    const provider = {
      keys: [
        {
          kid: "ed_1",
          issuer_id: payload.issuer_id,
          agent_id: payload.agent_id,
          alg: "Ed25519",
          status: "active",
          public_key: raw,
        },
      ],
    };
    return { doc, provider };
  }

  it("treats a null size as malformed end to end", () => {
    const { doc, provider } = genuinelySigned({ hash: HONEST, size: null });
    const r = verify(doc, ADAPTERS, provider);
    expect(r.axes.find((a) => a.axis === "signature")!.result).toBe("PASS");
    const axis = r.axes.find((a) => a.axis === "payload_digest")!;
    expect(axis.result).toBe("FAIL");
    expect(axis.failureClass).toBe("invalid");
    expect(axis.note).toContain("non-negative integer");
    expect(r.verdict).toBe("unverified");
  });

  it("keeps an absent-size control passing end to end", () => {
    const { doc, provider } = genuinelySigned({ hash: HONEST });
    const r = verify(doc, ADAPTERS, provider);
    expect(r.axes.find((a) => a.axis === "signature")!.result).toBe("PASS");
    const axis = r.axes.find((a) => a.axis === "payload_digest")!;
    expect(axis.result, axis.note).toBe("PASS");
  });
});
