// The anchors axis hashes the two-key {payload, signature} object the signer committed.
// TS half of test_verifier_anchor_scope.py: export-side members never reach the anchored bytes.

import { describe, expect, it } from "vitest";

import { asqavJcs } from "../src/verifier/canonical.js";
import { envelopeMinusAnchorsJcs } from "../src/verifier/vrShim.js";

const PAYLOAD = { type: "protectmcp:decision", issuer_id: "org-A", decision: "allow" };
const SIGNATURE = { alg: "ML-DSA-65", kid: "org-A", sig: "AAEC" };

describe("envelopeMinusAnchorsJcs", () => {
  it("hashes the two-key object an export shares with the signed receipt", () => {
    const exported = {
      payload: PAYLOAD,
      signature: SIGNATURE,
      anchors: [{ type: "rfc3161", value: "AAEC" }],
      signature_id: "sig_1",
      exported_at: "2026-09-02T00:00:00Z",
      audit_pack_version: 2,
      payload_b64: "e30=",
    };
    const twoKey = asqavJcs({ payload: PAYLOAD, signature: SIGNATURE });
    expect(Buffer.from(envelopeMinusAnchorsJcs(exported))).toEqual(Buffer.from(twoKey));
  });

  it("is unchanged by an export-side member appearing or vanishing", () => {
    const bare = { payload: PAYLOAD, signature: SIGNATURE, anchors: [] };
    const rich = { ...bare, org_id: "org-A", receipt_url: "https://example.test/r/1" };
    expect(Buffer.from(envelopeMinusAnchorsJcs(rich))).toEqual(
      Buffer.from(envelopeMinusAnchorsJcs(bare)),
    );
  });
});
