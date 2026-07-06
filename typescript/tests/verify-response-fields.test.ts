/**
 * SDK parser surfaces every field the cloud `/verify` returns.
 *
 * The cloud's `VerificationResponse` emits the IETF projection
 * (`signature_envelope`, `anchors`, `algorithm_registry_version`),
 * the `bitcoin_anchor` block, the receipt `type` discriminator, and
 * the `VerificationDetail` sub-axes (`anchor_valid_ots`,
 * `anchor_valid_rfc3161`, `policy_digest_resolved`,
 * `duplicate_emission_candidate`, `regimes_satisfied`). These tests
 * pin that the parser populates the TS interface with all of them and
 * that the `validationLabel` union admits `agent_revoked_before_issuance`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  _resetForTests,
  init,
  verifySignature,
  type VerificationResponse,
} from "../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fullCloudPayload() {
  return {
    type: "signature",
    signature_id: "sig_test_full",
    agent_id: "agent_full",
    agent_name: "full-agent",
    action_id: "act_full",
    action_type: "payment.wire_transfer",
    payload: { amount: 100 },
    signature: "ZmFrZQ==",
    algorithm: "ML-DSA-65",
    signed_at: "2026-05-10T12:00:00+00:00",
    verified: true,
    verification_url: "https://verify.example/sig_test_full",
    bitcoin_anchor: {
      status: "confirmed",
      anchor_tx_ref: "abc123",
      anchor_block_height: 850000,
    },
    verification_detail: {
      signer_key_match: true,
      signature_valid: true,
      algorithm_match: true,
      agent_active: true,
      validation_label: "valid",
      signed_at_skew_seconds: 1.5,
      chain_valid: true,
      anchor_valid_ots: true,
      anchor_valid_rfc3161: true,
      anchor_status_ots: "valid",
      anchor_status_rfc3161: "valid",
      missing_fields: [],
      policy_digest_resolved: true,
      duplicate_emission_candidate: false,
      regimes_satisfied: ["eu_ai_act", "dora"],
    },
    signature_envelope: { alg: "ML-DSA-65", kid: "key_full" },
    anchors: [
      { type: "rfc3161", value: "ZmFrZQ==" },
      { type: "opentimestamps", value: "b3RzZmFrZQ==" },
    ],
    algorithm_registry_version: "2026-05",
    execution_evidence: {
      present: true,
      result_digest: "sha256:" + "b".repeat(64),
      result_bound_receipt: true,
      disposition: "bound",
    },
  };
}

describe("verifySignature response fields", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("exposes IETF projection fields on the parsed response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(fullCloudPayload()),
    );

    const response: VerificationResponse = await verifySignature("sig_test_full");

    expect(response.type).toBe("signature");
    expect(response.bitcoinAnchor).toEqual({
      status: "confirmed",
      anchorTxRef: "abc123",
      anchorBlockHeight: 850000,
    });
    expect(response.signatureEnvelope).toEqual({
      alg: "ML-DSA-65",
      kid: "key_full",
    });
    expect(response.anchors).toHaveLength(2);
    expect(response.anchors?.[0]).toEqual({
      type: "rfc3161",
      value: "ZmFrZQ==",
    });
    expect(response.anchors?.[1]?.type).toBe("opentimestamps");
    expect(response.algorithmRegistryVersion).toBe("2026-05");
  });

  it("exposes the four extra VerificationDetail sub-axes", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(fullCloudPayload()),
    );

    const response = await verifySignature("sig_test_full");
    const detail = response.verificationDetail;

    expect(detail?.anchorValidOts).toBe(true);
    expect(detail?.anchorValidRfc3161).toBe(true);
    expect(detail?.policyDigestResolved).toBe(true);
    expect(detail?.duplicateEmissionCandidate).toBe(false);
  });

  it("exposes execution_evidence with its bound disposition", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(fullCloudPayload()),
    );

    const response = await verifySignature("sig_test_full");

    expect(response.executionEvidence).toEqual({
      present: true,
      resultDigest: "sha256:" + "b".repeat(64),
      resultBoundReceipt: true,
      disposition: "bound",
    });
  });

  it("leaves new fields undefined when the server omits them", async () => {
    const minimalPayload = {
      signature_id: "sig_minimal",
      agent_id: "agent_minimal",
      agent_name: "minimal",
      action_id: "act_minimal",
      action_type: "x.y",
      payload: null,
      signature: "ZmFrZQ==",
      algorithm: "Ed25519",
      signed_at: "2026-05-10T12:00:00+00:00",
      verified: true,
      verification_url: "https://verify.example/sig_minimal",
    };

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(minimalPayload));

    const response = await verifySignature("sig_minimal");

    expect(response.type).toBeUndefined();
    expect(response.bitcoinAnchor).toBeUndefined();
    expect(response.signatureEnvelope).toBeUndefined();
    expect(response.anchors).toBeUndefined();
    expect(response.algorithmRegistryVersion).toBeUndefined();
    expect(response.verificationDetail).toBeUndefined();
    expect(response.executionEvidence).toBeUndefined();
  });

  it("test_regimes_satisfied_present", async () => {
    // Cloud PR #337 added regimes_satisfied after the SDK projection
    // PR #160 landed; pin that the SDK parser now reads it.
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(fullCloudPayload()),
    );

    const response = await verifySignature("sig_test_full");
    expect(response.verificationDetail?.regimesSatisfied).toEqual([
      "eu_ai_act",
      "dora",
    ]);
  });

  it("test_anchor_type_accepts_opentimestamps_and_rfc3161", async () => {
    // _project_anchors emits "opentimestamps" (canonical) and
    // "rfc3161". The SDK parser does not silently rewrite.
    const payload = {
      ...fullCloudPayload(),
      anchors: [
        { type: "rfc3161", value: "ZmFrZQ==" },
        { type: "opentimestamps", value: "Y2Fub25pY2Fs" },
      ],
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(payload));

    const response = await verifySignature("sig_test_full");
    const types = response.anchors?.map((a) => a.type);
    expect(types).toEqual(["rfc3161", "opentimestamps"]);
  });

  it("admits agent_revoked_before_issuance on the validation-label union", async () => {
    const revokedPayload = {
      ...fullCloudPayload(),
      verified: false,
      verification_detail: {
        ...fullCloudPayload().verification_detail,
        validation_label: "agent_revoked_before_issuance",
      },
    };

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(revokedPayload));

    const response = await verifySignature("sig_test_full");

    expect(response.verificationDetail?.validationLabel).toBe(
      "agent_revoked_before_issuance",
    );
  });
});
