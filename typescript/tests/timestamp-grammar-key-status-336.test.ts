// Key-status timestamp-grammar pin (336); TS half of the Python mirror test
// The same stamps must draw the same verdicts in both engines

import { describe, expect, it } from "vitest";

import { checkPeerReceipt } from "../src/acceptor.js";
import { ADAPTERS } from "../src/verifier/index.js";
import { verify } from "../src/verifier/core.js";
import { checkKeyStatus } from "../src/verifier/vrShim.js";
import {
  createHash,
  createPrivateKey,
  createPublicKey,
  sign as nodeSign,
} from "node:crypto";

const REVOKED_AFTER = "2020-06-01T00:00:00Z";
const BASIC = "20200101T000000Z";

describe("key status pins the stamp grammar (criterion 336)", () => {
  it("FAILs a basic-form issued_at as unparseable", () => {
    const [res, note] = checkKeyStatus("revoked", BASIC, REVOKED_AFTER, false);
    expect(res).toBe("FAIL");
    expect(note.toLowerCase()).toContain("unparseable");
  });

  it("FAILs a basic-form revoked_at as unparseable", () => {
    const [res, note] = checkKeyStatus("revoked", REVOKED_AFTER, BASIC, false);
    expect(res).toBe("FAIL");
    expect(note.toLowerCase()).toContain("unparseable");
  });

  it("FAILs a stamp Date parses but the grammar rejects", () => {
    // A spelled-out English date parses in JS but the grammar refuses it
    const [res, note] = checkKeyStatus("revoked", "June 1, 2020 UTC", REVOKED_AFTER, false);
    expect(res).toBe("FAIL");
    expect(note.toLowerCase()).toContain("unparseable");
  });

  it("keeps the extended-form answers", () => {
    expect(checkKeyStatus("revoked", "2020-05-01T00:00:00Z", REVOKED_AFTER, false)[0]).toBe(
      "SKIPPED",
    );
    expect(checkKeyStatus("revoked", "2020-05-01T00:00:00Z", REVOKED_AFTER, true)[0]).toBe("PASS");
    expect(checkKeyStatus("revoked", "2020-07-01T00:00:00Z", REVOKED_AFTER, false)[0]).toBe("FAIL");
    expect(checkKeyStatus("active", BASIC, REVOKED_AFTER, false)[0]).toBe("PASS");
  });
});

const SEED_PHRASE = "asqav conformance corpus v1 seq-continuity signing seed";
const KID = "asqav-seq-vec-key";
const ISSUER = "Asqav Ltd";
const ZERO_DIGEST = createHash("sha256").update("").digest("hex");
const PKCS8_ED25519_PREFIX = Buffer.from("302e020100300506032b657004220420", "hex");
const SPKI_ED25519_PREFIX = Buffer.from("302a300506032b6570032100", "hex");

function privateKey() {
  return createPrivateKey({
    key: Buffer.concat([PKCS8_ED25519_PREFIX, createHash("sha256").update(SEED_PHRASE, "utf-8").digest()]),
    format: "der",
    type: "pkcs8",
  });
}

function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  const entries = Object.keys(value as Record<string, unknown>)
    .sort()
    .map((k) => `${JSON.stringify(k)}:${canonical((value as Record<string, unknown>)[k])}`);
  return `{${entries.join(",")}}`;
}

function jwks(): Record<string, unknown> {
  const spki = createPublicKey(privateKey()).export({ format: "der", type: "spki" });
  const raw = (spki as Buffer).subarray(SPKI_ED25519_PREFIX.length);
  return {
    keys: [
      {
        kid: KID,
        issuer_id: ISSUER,
        alg: "Ed25519",
        status: "active",
        public_key: raw.toString("base64"),
      },
    ],
  };
}

function receiptWithExpiry(expiresAt: unknown): {
  receipt: Record<string, unknown>;
  jwks: Record<string, unknown>;
} {
  const payload: Record<string, unknown> = {
    type: "protectmcp:decision",
    v: 1,
    mode: "payload",
    issued_at: "2026-08-30T12:00:00+00:00",
    issuer_id: ISSUER,
    agent_id: "agt_acceptor_001",
    action_ref: "act_336",
    payload_digest: { hash: ZERO_DIGEST, size: 0 },
    policy_digest: `sha256:${ZERO_DIGEST}`,
    previousReceiptHash: "0".repeat(64),
    decision: "allow",
    tool_name: "demo.action",
    expires_at: expiresAt,
  };
  const sig = nodeSign(null, Buffer.from(canonical(payload), "utf-8"), privateKey());
  return {
    receipt: {
      payload,
      signature: { alg: "Ed25519", kid: KID, sig: Buffer.from(sig).toString("base64") },
      anchors: [],
    },
    jwks: jwks(),
  };
}

describe("acceptor expiry pins the stamp grammar (criterion 336)", () => {
  it("refuses a future junk-date expiry as unreadable, not as valid", () => {
    // `new Date("January 1, 2100")` parses; the grammar must still refuse it
    const { receipt, jwks } = receiptWithExpiry("January 1, 2100");
    expect(verify(receipt, ADAPTERS, jwks, null).verdict).toMatch(/verified/);
    const got = checkPeerReceipt(receipt, { keyProvider: jwks });
    expect(got.accepted).toBe(false);
    expect(got.rule).toBe("expiry");
    expect(got.reason).toContain("unreadable");
  });

  it("still refuses a basic-form expiry", () => {
    const { receipt, jwks } = receiptWithExpiry("21000101T000000Z");
    const got = checkPeerReceipt(receipt, { keyProvider: jwks });
    expect(got.accepted).toBe(false);
    expect(got.rule).toBe("expiry");
  });
});
