/**
 * Gates for acceptor-side admission control (criterion 472, B15), the TypeScript
 * half. Mirrors python/tests/test_acceptor.py rule for rule.
 *
 * The acceptor rules refuse receipts the VERIFIER is content with, so testing
 * them needs receipts that actually verify. Editing a corpus payload cannot do
 * it: the edit breaks the signature, the verifier refuses first, and the rule
 * under test never runs. So these mint real receipts with the corpus's own
 * published seed.
 */

import { createHash, createPrivateKey, createPublicKey, sign as nodeSign } from "node:crypto";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { checkPeerReceipt } from "../src/acceptor.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { verify } from "../src/verifier/core.js";

const CORPUS = resolve(__dirname, "..", "..", "verifier", "conformance-vectors");

const SEED_PHRASE = "asqav conformance corpus v1 seq-continuity signing seed";
const KID = "asqav-seq-vec-key";
const ISSUER = "Asqav Ltd";
const ZERO_DIGEST = createHash("sha256").update("").digest("hex");

// The verifier verifies Ed25519 through node:crypto, so sign through it too
// rather than pulling a second implementation into the fixtures.
const PKCS8_ED25519_PREFIX = Buffer.from("302e020100300506032b657004220420", "hex");
const SPKI_ED25519_PREFIX = Buffer.from("302a300506032b6570032100", "hex");

function seed(): Buffer {
  return createHash("sha256").update(SEED_PHRASE, "utf-8").digest();
}

function privateKey() {
  return createPrivateKey({
    key: Buffer.concat([PKCS8_ED25519_PREFIX, seed()]),
    format: "der",
    type: "pkcs8",
  });
}

function rawPublicKey(): Buffer {
  const spki = createPublicKey(privateKey()).export({ format: "der", type: "spki" });
  return spki.subarray(SPKI_ED25519_PREFIX.length);
}

function jcs(obj: unknown): Buffer {
  // Matches the oracle's asqavJcs for these flat, ASCII payloads.
  return Buffer.from(canonical(obj), "utf-8");
}

function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  const entries = Object.keys(value as Record<string, unknown>)
    .sort()
    .map((k) => `${JSON.stringify(k)}:${canonical((value as Record<string, unknown>)[k])}`);
  return `{${entries.join(",")}}`;
}

function signed(
  actionRef: string,
  previous: string,
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    type: "protectmcp:decision",
    issued_at: "2026-08-30T12:00:00+00:00",
    issuer_id: ISSUER,
    agent_id: "agt_acceptor_001",
    action_ref: actionRef,
    payload_digest: { hash: ZERO_DIGEST, size: 0 },
    policy_digest: `sha256:${ZERO_DIGEST}`,
    previousReceiptHash: previous,
    decision: "allow",
    tool_name: "demo.action",
    ...extra,
  };
  const sig = nodeSign(null, jcs(payload), privateKey());
  return {
    payload,
    signature: { alg: "Ed25519", kid: KID, sig: Buffer.from(sig).toString("base64") },
    anchors: [],
  };
}

function signedJwks(): Record<string, unknown> {
  const pub = rawPublicKey();
  return {
    keys: [
      {
        kid: KID,
        issuer_id: ISSUER,
        alg: "Ed25519",
        status: "active",
        public_key: Buffer.from(pub).toString("base64"),
      },
    ],
  };
}

/** A predecessor and the successor that links to it, both properly signed. */
function chained(firstExtra: Record<string, unknown>, secondExtra: Record<string, unknown>) {
  const first = signed("act_1", "0".repeat(64), firstExtra);
  const link = createHash("sha256").update(jcs(first.payload)).digest("hex");
  const second = signed("act_2", link, secondExtra);
  return { receipt: second, jwks: signedJwks(), predecessor: first };
}

function vec(name: string) {
  const load = (f: string) =>
    JSON.parse(readFileSync(join(CORPUS, name, f), "utf-8")) as Record<string, unknown>;
  return { receipt: load("receipt.json"), jwks: load("jwks.json"), predecessor: load("predecessor.json") };
}

describe("acceptor: admits what verifies", () => {
  it("admits a clean peer receipt", () => {
    const { receipt, jwks, predecessor } = chained({ seq: 7 }, { seq: 8 });
    const got = checkPeerReceipt(receipt, { keyProvider: jwks, predecessor });
    expect(got.accepted, got.reason).toBe(true);
    expect(got.firstFailingEdge).toBeNull();
    expect(got.rule).toBeNull();
  });

  it("the fixture really does verify, so an admission is not a default", () => {
    const { receipt, jwks, predecessor } = chained({ seq: 7 }, { seq: 8 });
    const result = verify(receipt, ADAPTERS, jwks, predecessor);
    expect(["verified", "verified_keyed"]).toContain(result.verdict);
  });
});

describe("acceptor: refuses what does not verify", () => {
  it("refuses a withheld-receipt gap and names the seq edge", () => {
    const { receipt, jwks, predecessor } = vec("asqav-18-seq-gap");
    const got = checkPeerReceipt(receipt, { keyProvider: jwks, predecessor });
    expect(got.accepted).toBe(false);
    expect(got.firstFailingEdge).toBe("seq");
    expect(got.rule).toBe("verifier");
  });

  it("refuses a substituted key though the signature verifies", () => {
    const receipt = JSON.parse(
      readFileSync(join(CORPUS, "asqav-22-key-substituted", "receipt.json"), "utf-8"),
    ) as Record<string, unknown>;
    const jwks = JSON.parse(
      readFileSync(join(CORPUS, "asqav-22-key-substituted", "jwks.json"), "utf-8"),
    ) as Record<string, unknown>;
    const got = checkPeerReceipt(receipt, { keyProvider: jwks });
    expect(got.accepted).toBe(false);
    expect(got.firstFailingEdge).toBe("key_binding");
    expect(got.failureClass).toBe("invalid");
  });
});

describe("acceptor: expiry is an acceptor rule, not a verdict", () => {
  const expiring = (when: Date) =>
    chained({ seq: 7 }, { seq: 8, expires_at: when.toISOString() });

  it("the verifier still calls an expired receipt verified", () => {
    // The premise the rule rests on, asserted rather than assumed.
    const { receipt, jwks, predecessor } = expiring(new Date(Date.now() - 86_400_000));
    const result = verify(receipt, ADAPTERS, jwks, predecessor);
    const expiry = result.axes.find((a) => a.axis === "expiry");
    expect(expiry, "no expiry axis; the premise no longer holds").toBeDefined();
    expect(expiry!.result).toBe("FAIL");
    expect(["verified", "verified_keyed"]).toContain(result.verdict);
  });

  it("refuses an expired receipt", () => {
    const { receipt, jwks, predecessor } = expiring(new Date(Date.now() - 86_400_000));
    const got = checkPeerReceipt(receipt, { keyProvider: jwks, predecessor });
    expect(got.accepted).toBe(false);
    expect(got.rule).toBe("expiry");
  });

  it("admits an unexpired receipt", () => {
    // The control. Without it the rule could refuse everything and pass.
    const { receipt, jwks, predecessor } = expiring(new Date(Date.now() + 86_400_000));
    const got = checkPeerReceipt(receipt, { keyProvider: jwks, predecessor });
    expect(got.accepted, got.reason).toBe(true);
  });

  it("honours the caller-supplied clock", () => {
    const stamp = new Date("2030-01-01T00:00:00Z");
    const { receipt, jwks, predecessor } = expiring(stamp);
    const before = checkPeerReceipt(receipt, {
      keyProvider: jwks,
      predecessor,
      now: new Date(stamp.getTime() - 86_400_000),
    });
    const after = checkPeerReceipt(receipt, {
      keyProvider: jwks,
      predecessor,
      now: new Date(stamp.getTime() + 86_400_000),
    });
    expect(before.accepted, before.reason).toBe(true);
    expect(after.accepted).toBe(false);
    expect(after.rule).toBe("expiry");
  });

  it("refuses an unreadable expiry rather than reading it as no expiry", () => {
    const { receipt, jwks, predecessor } = chained(
      { seq: 7 },
      { seq: 8, expires_at: "not-a-timestamp" },
    );
    const got = checkPeerReceipt(receipt, { keyProvider: jwks, predecessor });
    expect(got.accepted).toBe(false);
    expect(got.rule).toBe("expiry");
  });
});

describe("acceptor: seq downgrade", () => {
  it("refuses a peer that carried a counter and stops", () => {
    const { receipt, jwks, predecessor } = chained({ seq: 7 }, {});
    const got = checkPeerReceipt(receipt, { keyProvider: jwks, predecessor });
    expect(got.accepted).toBe(false);
    expect(got.rule).toBe("seq_downgrade");
  });

  it("still admits a peer that never carried one", () => {
    // Absence stays legal, or every receipt minted before the counter shipped
    // would be refused by an acceptor.
    const { receipt, jwks, predecessor } = chained({}, {});
    const got = checkPeerReceipt(receipt, { keyProvider: jwks, predecessor });
    expect(got.accepted, got.reason).toBe(true);
  });

  it("admits a contiguous pair", () => {
    const { receipt, jwks, predecessor } = chained({ seq: 7 }, { seq: 8 });
    const got = checkPeerReceipt(receipt, { keyProvider: jwks, predecessor });
    expect(got.accepted, got.reason).toBe(true);
  });

  it("does not invent a downgrade with no predecessor", () => {
    const got = checkPeerReceipt(signed("act_1", "0".repeat(64)), {
      keyProvider: signedJwks(),
    });
    expect(got.rule).not.toBe("seq_downgrade");
  });
});

describe("acceptor: challenge", () => {
  it("refuses an unanswered challenge", () => {
    const { receipt, jwks, predecessor } = chained({ seq: 7 }, { seq: 8 });
    const got = checkPeerReceipt(receipt, {
      keyProvider: jwks,
      predecessor,
      challenge: "chal-abc",
    });
    expect(got.accepted).toBe(false);
    expect(got.rule).toBe("challenge");
  });

  it("refuses the wrong challenge", () => {
    const { receipt, jwks, predecessor } = chained(
      { seq: 7 },
      { seq: 8, challenge_nonce: "chal-WRONG" },
    );
    const got = checkPeerReceipt(receipt, {
      keyProvider: jwks,
      predecessor,
      challenge: "chal-abc",
    });
    expect(got.accepted).toBe(false);
    expect(got.rule).toBe("challenge");
    expect(got.failureClass).toBe("invalid");
  });

  it("admits the matching challenge", () => {
    const { receipt, jwks, predecessor } = chained(
      { seq: 7 },
      { seq: 8, challenge_nonce: "chal-abc" },
    );
    const got = checkPeerReceipt(receipt, {
      keyProvider: jwks,
      predecessor,
      challenge: "chal-abc",
    });
    expect(got.accepted, got.reason).toBe(true);
  });

  it("does not invent a mismatch when it issued no challenge", () => {
    const { receipt, jwks, predecessor } = chained(
      { seq: 7 },
      { seq: 8, challenge_nonce: "chal-other" },
    );
    const got = checkPeerReceipt(receipt, { keyProvider: jwks, predecessor });
    expect(got.accepted, got.reason).toBe(true);
  });
});

describe("acceptor: rule order is deterministic", () => {
  it("reports the verifier ahead of the acceptor rules", () => {
    const { receipt, jwks, predecessor } = chained(
      { seq: 7 },
      { seq: 11, expires_at: "2000-01-01T00:00:00Z" },
    );
    const got = checkPeerReceipt(receipt, { keyProvider: jwks, predecessor });
    expect(got.accepted).toBe(false);
    expect(got.rule).toBe("verifier");
    expect(got.firstFailingEdge).toBe("seq");
  });

  it("gives the same decision for the same inputs", () => {
    const { receipt, jwks, predecessor } = vec("asqav-18-seq-gap");
    const a = checkPeerReceipt(receipt, { keyProvider: jwks, predecessor });
    const b = checkPeerReceipt(receipt, { keyProvider: jwks, predecessor });
    expect(a).toEqual(b);
  });
});
