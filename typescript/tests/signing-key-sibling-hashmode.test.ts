// A multi-agent org's hash-mode receipts resolve the actual signer, not the first
// sibling. The TypeScript half of python/tests/test_oracle_multikey_org_resolution.py:
// a hash-mode receipt sets signature.kid to the org id and signs with the agent's own
// key, so the agent bind (agent_id plus the org_id the receipt signs) must pick the
// signer. ANTI-VACUOUS: the signer is the LAST of three siblings, so the pre-change
// kid-first resolution checks the first sibling and FAILs the signature axis.

import { describe, expect, it } from "vitest";
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";

import { AsqavNativeAdapter } from "../src/verifier/adapters/asqavNative.js";
import { asqavJcs } from "../src/verifier/canonical.js";
import { verify } from "../src/verifier/core.js";

const ORG = "org_multikey_shared";

function flat(agentId: string): Record<string, unknown> {
  return {
    v: 1,
    mode: "hash",
    hash: "c".repeat(64),
    hash_algo: "sha256",
    metadata: {},
    server_timestamp: "2026-01-01T00:00:00Z",
    action_id: "act_1",
    agent_id: agentId,
    org_id: ORG,
    policy_digest: "d".repeat(64),
    policy_decision: "allow",
  };
}

function hashReceipt(f: Record<string, unknown>, sig: string): Record<string, unknown> {
  return {
    ...f,
    payload: null,
    algorithm: "ML-DSA-65",
    key_id: ORG,
    signature_b64: sig,
  };
}

function siblingKey(i: number, publicKey: Uint8Array): Record<string, unknown> {
  return {
    kid: `crypto_kid_${i}`,
    agent_id: `agt_${i}`,
    issuer_id: ORG,
    org_id: ORG,
    alg: "ML-DSA-65",
    public_key: Buffer.from(publicKey).toString("base64"),
    status: "active",
  };
}

function axesOf(res: ReturnType<typeof verify>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const a of res.axes) out[a.axis] = a.result;
  return out;
}

describe("hash-mode resolution across three org siblings", () => {
  it("passes a receipt signed by the last sibling", () => {
    const sks = [ml_dsa65.keygen(), ml_dsa65.keygen(), ml_dsa65.keygen()];
    const f = flat("agt_2");
    const sig = Buffer.from(ml_dsa65.sign(asqavJcs(f), sks[2].secretKey)).toString("base64");
    const jwks = { keys: sks.map((kp, i) => siblingKey(i, kp.publicKey)) };
    const res = verify(hashReceipt(f, sig), [new AsqavNativeAdapter()], jwks);
    const axes = axesOf(res);
    expect(axes.signature, JSON.stringify(axes)).toBe("PASS");
    expect(res.verdict, JSON.stringify(axes)).toBe("verified");
    expect(res.failureClass).toBeNull();
  });

  it("reports verified_keyed, never plain verified, for an hmac-sha256 digest (criterion 438)", () => {
    const kp = ml_dsa65.keygen();
    const f = { ...flat("agt_1"), hash_algo: "hmac-sha256" };
    const sig = Buffer.from(ml_dsa65.sign(asqavJcs(f), kp.secretKey)).toString("base64");
    const jwks = { keys: [siblingKey(1, kp.publicKey)] };
    const res = verify(hashReceipt(f, sig), [new AsqavNativeAdapter()], jwks);
    expect(res.verdict).toBe("verified_keyed");
    expect(res.failureClass).toBeNull();
  });

  it("resolves each sibling to its own key", () => {
    const sks = [ml_dsa65.keygen(), ml_dsa65.keygen(), ml_dsa65.keygen()];
    const jwks = { keys: sks.map((kp, i) => siblingKey(i, kp.publicKey)) };
    const ad = new AsqavNativeAdapter();
    sks.forEach((kp, i) => {
      const f = flat(`agt_${i}`);
      const sig = Buffer.from(ml_dsa65.sign(asqavJcs(f), kp.secretKey)).toString("base64");
      const [pk] = ad.resolveKey(hashReceipt(f, sig), jwks);
      expect(pk).not.toBeNull();
      expect(Buffer.from(pk as Uint8Array).equals(Buffer.from(kp.publicKey))).toBe(true);
    });
  });

  it("rejects a signature from an unpublished key", () => {
    const sks = [ml_dsa65.keygen(), ml_dsa65.keygen(), ml_dsa65.keygen()];
    const forger = ml_dsa65.keygen();
    const f = flat("agt_2");
    const sig = Buffer.from(ml_dsa65.sign(asqavJcs(f), forger.secretKey)).toString("base64");
    const jwks = { keys: sks.map((kp, i) => siblingKey(i, kp.publicKey)) };
    const res = verify(hashReceipt(f, sig), [new AsqavNativeAdapter()], jwks);
    expect(res.verdict).not.toBe("PASS");
  });
});
