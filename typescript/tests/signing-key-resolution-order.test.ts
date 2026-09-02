// The asqav-native adapter resolves the signing key by the signed key_thumbprint first. TS half of
// test_signing_key_resolution_order.py: two published keys under one issuer in every directory.

import { describe, expect, it } from "vitest";
import { ml_dsa44, ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";

import { AsqavNativeAdapter } from "../src/verifier/adapters/asqavNative.js";
import { asqavJcs } from "../src/verifier/canonical.js";
import { verify } from "../src/verifier/core.js";
import { matchSigningKey, thumbprintForKey } from "../src/verifier/vrShim.js";

const ORG = "f94f66c0-c580-432d-a041-29374f7aee07";
const OTHER_ORG = "0b6c2b1e-9f7a-4d3c-8a11-5c2e7d904f38";

const sign = (msg: Uint8Array, sk: Uint8Array) => Buffer.from(ml_dsa65.sign(msg, sk)).toString("base64");

function key(
  kid: string,
  agentId: string,
  pk: Uint8Array,
  opts: { issuerId?: string; status?: string; thumbprint?: boolean } = {},
): Record<string, unknown> {
  const issuer = opts.issuerId ?? ORG;
  const entry: Record<string, unknown> = {
    kid,
    agent_id: agentId,
    issuer_id: issuer,
    org_id: issuer,
    alg: "ML-DSA-65",
    kty: "AKP",
    public_key: Buffer.from(pk).toString("base64"),
    status: opts.status ?? "active",
    revoked_at: opts.status === "revoked" ? "2026-07-11T14:57:49Z" : null,
  };
  if (opts.thumbprint !== false) entry.key_thumbprint = thumbprintForKey("ML-DSA-65", pk);
  return entry;
}

function payload(agentId = "agt_two", thumbprint?: string, issuerId = ORG): Record<string, unknown> {
  const p: Record<string, unknown> = {
    type: "protectmcp:decision",
    issued_at: "2026-06-19T00:00:00.000000Z",
    issuer_id: issuerId,
    agent_id: agentId,
    action_ref: `sha256:${"8".repeat(64)}`,
    payload_digest: { hash: "8".repeat(64), size: 512 },
    policy_digest: `sha256:${"3".repeat(64)}`,
    previousReceiptHash: "0".repeat(64),
    decision: "allow",
  };
  if (thumbprint !== undefined) p.key_thumbprint = thumbprint;
  return p;
}

function receipt(p: Record<string, unknown>, sig: string, alg = "ML-DSA-65"): Record<string, unknown> {
  return { payload: p, signature: { alg, kid: ORG, sig }, anchors: [] };
}

function axesOf(doc: Record<string, unknown>, jwks: Record<string, unknown>) {
  const res = verify(doc, [new AsqavNativeAdapter()], jwks);
  const axes: Record<string, { result: string; note: string }> = {};
  for (const a of res.axes) axes[a.axis] = { result: a.result, note: a.note };
  return { res, axes };
}

describe("signing-key resolution order", () => {
  it("resolves the named agent key, not the revoked sibling, when the signature cannot be checked", () => {
    // An algorithm outside the profile, as the live ML-DSA-44 canary of 2026-09-02 was.
    const old = ml_dsa44.keygen();
    const two = ml_dsa44.keygen();
    const p = payload("agt_two");
    const oldKey = key("k-old", "agt_one", old.publicKey, { status: "revoked", thumbprint: false });
    const twoKey = key("k-two", "agt_two", two.publicKey, { thumbprint: false });
    oldKey.alg = "ML-DSA-44";
    twoKey.alg = "ML-DSA-44";
    const jwks = { keys: [oldKey, twoKey] };
    const sig44 = Buffer.from(ml_dsa44.sign(asqavJcs(p), two.secretKey)).toString("base64");
    const { axes } = axesOf(receipt(p, sig44, "ML-DSA-44"), jwks);
    expect(axes.signature.result).toBe("SKIPPED");
    expect(axes.key_status.result).toBe("PASS");
    expect(axes.issuer_bind.result).toBe("PASS");
  });

  it("resolves a rotated agent by the signed thumbprint across two published keys", () => {
    const stale = ml_dsa65.keygen();
    const fresh = ml_dsa65.keygen();
    const p = payload("agt_two", thumbprintForKey("ML-DSA-65", fresh.publicKey));
    const jwks = { keys: [key("k-stale", "agt_two", stale.publicKey), key("k-new", "agt_two", fresh.publicKey)] };
    const { axes } = axesOf(receipt(p, sign(asqavJcs(p), fresh.secretKey)), jwks);
    expect(axes.signature.result).toBe("PASS");
    expect(axes.key_binding.result).toBe("PASS");
    expect(axes.key_status.result).toBe("PASS");
  });

  it("without a signed thumbprint a rotated agent still lands on the first key", () => {
    const stale = ml_dsa65.keygen();
    const fresh = ml_dsa65.keygen();
    const p = payload("agt_two");
    const jwks = { keys: [key("k-stale", "agt_two", stale.publicKey), key("k-new", "agt_two", fresh.publicKey)] };
    const { axes } = axesOf(receipt(p, sign(asqavJcs(p), fresh.secretKey)), jwks);
    expect(axes.signature.result).toBe("FAIL");
  });

  it("keeps a thumbprint naming an unused key from verifying", () => {
    const c = ml_dsa65.keygen();
    const d = ml_dsa65.keygen();
    const p = payload("agt_two", thumbprintForKey("ML-DSA-65", d.publicKey));
    const jwks = { keys: [key("k-c", "agt_two", c.publicKey), key("k-d", "agt_two", d.publicKey)] };
    const { res, axes } = axesOf(receipt(p, sign(asqavJcs(p), c.secretKey)), jwks);
    expect(axes.signature.result).toBe("FAIL");
    expect(res.verdict).toBe("unverified");
    expect(res.failureClass).toBe("invalid");
  });

  it("cannot pass the issuer bind with a foreign org's thumbprint", () => {
    const attacker = ml_dsa65.keygen();
    const victim = ml_dsa65.keygen();
    const p = payload("agt_attacker", thumbprintForKey("ML-DSA-65", attacker.publicKey));
    const jwks = {
      keys: [key("k-victim", "agt_victim", victim.publicKey), key("k-attacker", "agt_attacker", attacker.publicKey, { issuerId: OTHER_ORG })],
    };
    const { res, axes } = axesOf(receipt(p, sign(asqavJcs(p), attacker.secretKey)), jwks);
    expect(axes.issuer_bind.result).toBe("FAIL");
    expect(res.verdict).toBe("unverified");
  });

  it("orders thumbprint, then kid, then agent bind, then issuer", () => {
    const a = key("k-a", "agt_a", ml_dsa65.keygen().publicKey);
    const b = key("k-b", "agt_b", ml_dsa65.keygen().publicKey);
    const jwks = { keys: [a, b] };
    expect(matchSigningKey(jwks, "k-a", "agt_a", ORG, ORG, b.key_thumbprint)).toBe(b);
    expect(matchSigningKey(jwks, "k-a", "agt_b", ORG, ORG)).toBe(a);
    expect(matchSigningKey(jwks, ORG, "agt_b", ORG, ORG)).toBe(b);
    expect(matchSigningKey(jwks, ORG, undefined, ORG, ORG)).toBe(a);
    expect(matchSigningKey(jwks, ORG, "agt_b", ORG, ORG, "sha256:nope")).toBe(b);
  });
});
