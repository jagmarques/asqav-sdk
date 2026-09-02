// Sibling rows publishing one key are narrowed by the signed agent_id and then the envelope kid.
// TS half of test_verifier_key_resolution.py: list position decides nothing on its own.

import { describe, expect, it } from "vitest";
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";

import { matchSigningKey, thumbprintForKey } from "../src/verifier/vrShim.js";

const ORG = "f94f66c0-c580-432d-a041-29374f7aee07";
const AGENT = "agt_one";

function row(
  kid: string,
  pk: Uint8Array,
  opts: { agentId?: string; status?: string } = {},
): Record<string, unknown> {
  return {
    kid,
    agent_id: opts.agentId ?? AGENT,
    issuer_id: ORG,
    org_id: ORG,
    alg: "ML-DSA-65",
    kty: "AKP",
    public_key: Buffer.from(pk).toString("base64"),
    status: opts.status ?? "active",
    revoked_at: opts.status === "revoked" ? "2026-01-01T00:00:00Z" : null,
    key_thumbprint: thumbprintForKey("ML-DSA-65", pk),
  };
}

describe("matchSigningKey thumbprint tie-break", () => {
  const { publicKey: pk } = ml_dsa65.keygen();
  const thumb = thumbprintForKey("ML-DSA-65", pk);

  it("lands on the revoked row when the envelope kid names it", () => {
    const jwks = { keys: [row("k-A1", pk), row("k-A1-old", pk, { status: "revoked" })] };
    const entry = matchSigningKey(jwks, "k-A1-old", AGENT, ORG, ORG, thumb);
    expect(entry?.kid).toBe("k-A1-old");
    expect(entry?.status).toBe("revoked");
  });

  it("lands on the active row when the envelope kid names it", () => {
    const jwks = { keys: [row("k-A1-old", pk, { status: "revoked" }), row("k-A1", pk)] };
    const entry = matchSigningKey(jwks, "k-A1", AGENT, ORG, ORG, thumb);
    expect(entry?.kid).toBe("k-A1");
    expect(entry?.status).toBe("active");
  });

  it("narrows by the signed agent id before the kid", () => {
    const jwks = { keys: [row("k-other", pk, { agentId: "agt_two" }), row("k-mine", pk)] };
    const entry = matchSigningKey(jwks, "", AGENT, ORG, ORG, thumb);
    expect(entry?.kid).toBe("k-mine");
  });
});
