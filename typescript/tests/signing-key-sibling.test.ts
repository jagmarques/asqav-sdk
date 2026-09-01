// One org, two agent keys: the entry that answers must be the entry that signed. TS half of
// test_signing_key_sibling.py; every directory here publishes two keys under one issuer.

import { describe, expect, it } from "vitest";
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";

import { AsqavNativeAdapter } from "../src/verifier/adapters/asqavNative.js";
import { asqavJcs } from "../src/verifier/canonical.js";
import { verify } from "../src/verifier/core.js";
import { matchSigningKey } from "../src/verifier/vrShim.js";

const ORG = "f94f66c0-c580-432d-a041-29374f7aee07";
const OTHER_ORG = "0b6c2b1e-9f7a-4d3c-8a11-5c2e7d904f38";

function key(
  kid: string,
  agentId: string,
  opts: { issuerId?: string; publicKey?: Uint8Array | string; status?: string } = {},
): Record<string, unknown> {
  const pub = opts.publicKey ?? "QUFBQQ==";
  return {
    kid,
    agent_id: agentId,
    issuer_id: opts.issuerId ?? ORG,
    alg: "ML-DSA-65",
    public_key: typeof pub === "string" ? pub : Buffer.from(pub).toString("base64"),
    status: opts.status ?? "active",
  };
}

function payload(agentId = "agt_two"): Record<string, unknown> {
  return {
    type: "protectmcp:decision",
    issued_at: "2026-06-19T00:00:00.000000Z",
    issuer_id: ORG,
    agent_id: agentId,
    action_ref: `sha256:${"8".repeat(64)}`,
    payload_digest: { hash: "8".repeat(64), size: 512 },
    policy_digest: `sha256:${"3".repeat(64)}`,
    previousReceiptHash: "0".repeat(64),
    decision: "allow",
  };
}

function receipt(p: Record<string, unknown>, sig: string, kid = ORG): Record<string, unknown> {
  return { payload: p, signature: { alg: "ML-DSA-65", kid, sig }, anchors: [] };
}

function axesOf(doc: Record<string, unknown>, jwks: Record<string, unknown>) {
  const out: Record<string, string> = {};
  for (const [name, res] of new AsqavNativeAdapter().extraAxes(doc, jwks)) out[name] = res;
  return out;
}

describe("signing key resolution across agent siblings", () => {
  it("resolves the agent the receipt names, not the first sibling", () => {
    const jwks = { keys: [key("k-agent-one", "agt_one"), key("k-agent-two", "agt_two")] };
    expect(matchSigningKey(jwks, ORG, "agt_two", ORG)?.kid).toBe("k-agent-two");
  });

  it("resolves the first sibling when it is the named one", () => {
    const jwks = { keys: [key("k-agent-one", "agt_one"), key("k-agent-two", "agt_two")] };
    expect(matchSigningKey(jwks, ORG, "agt_one", ORG)?.kid).toBe("k-agent-one");
  });

  it("lets an exact key id outrank the agent bind", () => {
    const jwks = { keys: [key("k-agent-one", "agt_one"), key("k-agent-two", "agt_two")] };
    expect(matchSigningKey(jwks, "k-agent-one", "agt_two", ORG)?.kid).toBe("k-agent-one");
  });

  it("keeps answering an org kid when the receipt names no agent", () => {
    const jwks = { keys: [key("k-agent-one", "agt_one")] };
    expect(matchSigningKey(jwks, ORG, undefined, ORG)?.kid).toBe("k-agent-one");
  });

  it("rejects an agent key published under another issuer", () => {
    const jwks = { keys: [key("k-foreign", "agt_two", { issuerId: OTHER_ORG })] };
    expect(matchSigningKey(jwks, "kid-absent", "agt_two", ORG)).toBeNull();
  });

  it("treats a junk directory entry as a miss, never a throw", () => {
    const jwks = {
      keys: [null, 7, "text", { kid: "no-bytes", issuer_id: ORG }, key("k-agent-two", "agt_two")],
    };
    expect(matchSigningKey(jwks, ORG, "agt_two", ORG)?.kid).toBe("k-agent-two");
  });
});

describe("the gating axes read the entry that signed", () => {
  it("reports the signing agent's revoked status over an active sibling", () => {
    const jwks = {
      keys: [
        key("k-agent-one", "agt_one", { status: "active" }),
        key("k-agent-two", "agt_two", { status: "revoked" }),
      ],
    };
    expect(axesOf(receipt(payload(), "AAAA"), jwks).key_status).toBe("FAIL");
  });

  it("keeps an active signer active behind a revoked sibling", () => {
    const jwks = {
      keys: [
        key("k-agent-one", "agt_one", { status: "revoked" }),
        key("k-agent-two", "agt_two", { status: "active" }),
      ],
    };
    const axes = axesOf(receipt(payload(), "AAAA"), jwks);
    expect(axes.key_status).toBe("PASS");
    expect(axes.issuer_bind).toBe("PASS");
  });
});

describe("end to end over a real ML-DSA-65 signature", () => {
  const sign = (msg: Uint8Array, sk: Uint8Array) => Buffer.from(ml_dsa65.sign(msg, sk)).toString("base64");

  it("passes a receipt signed by the second agent", () => {
    const one = ml_dsa65.keygen();
    const two = ml_dsa65.keygen();
    const p = payload();
    const jwks = {
      keys: [
        key("k-agent-one", "agt_one", { publicKey: one.publicKey }),
        key("k-agent-two", "agt_two", { publicKey: two.publicKey }),
      ],
    };
    const res = verify(receipt(p, sign(asqavJcs(p), two.secretKey)), [new AsqavNativeAdapter()], jwks);
    const axes: Record<string, string> = {};
    for (const a of res.axes) axes[a.axis] = a.result;
    expect(axes.signature, JSON.stringify(axes)).toBe("PASS");
    expect(axes.key_status).toBe("PASS");
    expect(axes.issuer_bind).toBe("PASS");
    expect(res.verdict, JSON.stringify(axes)).toBe("verified");
    expect(res.failureClass).toBeNull();
  });

  it("rejects a signature from a key the directory never published", () => {
    const one = ml_dsa65.keygen();
    const two = ml_dsa65.keygen();
    const forger = ml_dsa65.keygen();
    const p = payload();
    const jwks = {
      keys: [
        key("k-agent-one", "agt_one", { publicKey: one.publicKey }),
        key("k-agent-two", "agt_two", { publicKey: two.publicKey }),
      ],
    };
    const res = verify(receipt(p, sign(asqavJcs(p), forger.secretKey)), [new AsqavNativeAdapter()], jwks);
    const axes: Record<string, string> = {};
    for (const a of res.axes) axes[a.axis] = a.result;
    expect(axes.signature).toBe("FAIL");
    expect(res.verdict).toBe("unverified");
    expect(res.failureClass).toBe("invalid");
  });

  it("rejects an agent key published under another org", () => {
    const one = ml_dsa65.keygen();
    const foreign = ml_dsa65.keygen();
    const p = payload();
    const jwks = {
      keys: [
        key("k-agent-one", "agt_one", { publicKey: one.publicKey }),
        key("k-foreign", "agt_two", { issuerId: OTHER_ORG, publicKey: foreign.publicKey }),
      ],
    };
    const res = verify(receipt(p, sign(asqavJcs(p), foreign.secretKey)), [new AsqavNativeAdapter()], jwks);
    const axes: Record<string, string> = {};
    for (const a of res.axes) axes[a.axis] = a.result;
    expect(axes.signature).toBe("FAIL");
    expect(res.verdict).toBe("unverified");
    expect(res.failureClass).toBe("invalid");
  });
});
