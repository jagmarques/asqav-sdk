/**
 * key_thumbprint binding axis, TypeScript half (criterion 458).
 *
 * Reads the same verifier/axis-parity-cases.json the Python suite reads, so a
 * rule that drifts in one language fails a suite. The wire tests below are
 * mutation-shaped: each fails if its own call site is deleted.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  akpJwk,
  b64urlNoPad,
  checkKeyBinding,
  jwkThumbprint,
  thumbprintForKey,
} from "../src/verifier/vrShim.js";
import { verify } from "../src/verifier/core.js";
import { ADAPTERS } from "../src/verifier/index.js";

const CASES_FILE = resolve(__dirname, "..", "..", "verifier", "axis-parity-cases.json");
const TABLE = JSON.parse(readFileSync(CASES_FILE, "utf8"));

type KeySpec = { alg: string; fill: number; len: number } | null;

// Expand a table `key` spec into the bytes both languages build it from.
function expand(spec: KeySpec): readonly [string | null, Uint8Array | null] {
  if (spec === null) return [null, null];
  return [spec.alg, new Uint8Array(spec.len).fill(spec.fill)];
}

const ALG = "ML-DSA-65";
const KEY_GOOD = new Uint8Array(1952).fill(0x00);
const KEY_EVIL = new Uint8Array(1952).fill(0x01);
const TP_GOOD = thumbprintForKey(ALG, KEY_GOOD);

function jwks(publicKey: Uint8Array, alg = ALG): Record<string, unknown> {
  return {
    keys: [
      {
        kid: "iss_1",
        issuer_id: "iss_1",
        agent_id: "ag_1",
        org_id: "org_1",
        alg,
        public_key: Buffer.from(publicKey).toString("base64"),
        status: "active",
      },
    ],
  };
}

function receipt(thumbprint?: string): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    v: 1,
    issuer_id: "iss_1",
    agent_id: "ag_1",
    org_id: "org_1",
    previousReceiptHash: "0".repeat(64),
    issued_at: "2026-08-31T10:00:00Z",
    action: { type: "x" },
    hash: "a".repeat(64),
  };
  if (thumbprint !== undefined) payload.key_thumbprint = thumbprint;
  return {
    payload,
    signature: { alg: ALG, kid: "iss_1", sig: Buffer.alloc(64).toString("base64") },
    anchors: {},
  };
}

describe("key_thumbprint binding parity", () => {
  // A table that silently empties would make every case below vacuous.
  it("reads a populated shared table", () => {
    expect(TABLE.key_binding.length).toBeGreaterThanOrEqual(15);
    expect(TABLE.key_thumbprint_vectors.length).toBeGreaterThanOrEqual(4);
    const outcomes = new Set(TABLE.key_binding.map((c: any) => c.expect.result));
    expect([...outcomes].sort()).toEqual(["FAIL", "PASS", "SKIPPED"]);
  });

  for (const testCase of TABLE.key_thumbprint_vectors) {
    it(`RFC 7638 vector: ${testCase.name}`, () => {
      const [alg, pk] = expand(testCase.key);
      expect(thumbprintForKey(alg as string, pk as Uint8Array)).toBe(testCase.thumbprint);
    });
  }

  for (const testCase of TABLE.key_binding) {
    it(`axis case: ${testCase.name}`, () => {
      const [alg, pk] = expand(testCase.key);
      const [result, note] = checkKeyBinding(testCase.payload, alg, pk);
      expect(result, note).toBe(testCase.expect.result);
      expect(note).toContain(testCase.expect.note_contains);
    });
  }

  // pub is unpadded base64url; the directory's own alphabet yields another digest.
  it("builds pub as unpadded base64url", () => {
    const jwk = akpJwk("ML-DSA-87", new Uint8Array(2592).fill(0xff));
    expect(Object.keys(jwk).sort()).toEqual(["alg", "kty", "pub"]);
    expect(jwk.kty).toBe("AKP");
    expect(jwk.pub).not.toMatch(/[/+=]/);
    expect(jwk.pub.startsWith("____")).toBe(true);
  });

  // RFC 7638 requires lexicographic members whatever order the caller passes.
  it("ignores member insertion order", () => {
    const pk = new Uint8Array(1952).fill(0x07);
    const ordered = { alg: ALG, kty: "AKP", pub: b64urlNoPad(pk) };
    const shuffled = { pub: b64urlNoPad(pk), kty: "AKP", alg: ALG };
    expect(jwkThumbprint(ordered)).toBe(jwkThumbprint(shuffled));
    expect(jwkThumbprint(ordered)).toBe(thumbprintForKey(ALG, pk));
  });

  // Deleting the axis from the adapter's extraAxes fails here.
  it("emits the axis from the oracle adapter", () => {
    const result = verify(receipt(TP_GOOD), ADAPTERS, jwks(KEY_GOOD));
    const axis = result.axes.find((a) => a.axis === "key_binding");
    expect(axis, "adapter did not emit a key_binding axis").toBeDefined();
    expect(axis!.result, axis!.note).toBe("PASS");
  });

  // Removing key_binding from INVALID_FAIL_AXES fails here.
  it("treats a substituted key as terminal invalid, never a warning", () => {
    const result = verify(receipt(TP_GOOD), ADAPTERS, jwks(KEY_EVIL));
    const axis = result.axes.find((a) => a.axis === "key_binding")!;
    expect(axis.result).toBe("FAIL");
    expect(axis.failureClass).toBe("invalid");
    expect(axis.note).toContain("key_substituted");
    expect(result.verdict).toBe("unverified");
    expect(result.failureClass).toBe("invalid");
  });

  // A skip blocks every axis but chain, so absence must PASS or legacy receipts break.
  it("does not block a verdict when no thumbprint is bound", () => {
    const axis = verify(receipt(), ADAPTERS, jwks(KEY_GOOD)).axes.find(
      (a) => a.axis === "key_binding",
    )!;
    expect(axis.result).toBe("PASS");
    expect(axis.failureClass).toBeNull();
  });

  // Removing key_thumbprint from UNSIGNED_CLAIM_FIELDS fails here.
  it("refuses a hash-mode receipt displaying an unsigned thumbprint", () => {
    const doc = {
      v: 1,
      mode: "hash",
      hash: "a".repeat(64),
      hash_algo: "sha256",
      metadata: {},
      server_timestamp: "2026-08-31T10:00:00Z",
      action_id: "a1",
      agent_id: "ag_1",
      org_id: "org_1",
      policy_decision: "allow",
      signature_b64: Buffer.alloc(64).toString("base64"),
      key_id: "iss_1",
      key_thumbprint: TP_GOOD,
    };
    const result = verify(doc, ADAPTERS, jwks(KEY_GOOD));
    const structure = result.axes.find((a) => a.axis === "structure")!;
    expect(structure.result, structure.note).toBe("FAIL");
    expect(structure.note).toContain("key_thumbprint");
    expect(result.verdict).toBe("unverified");
  });
});
