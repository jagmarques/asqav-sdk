/**
 * Counterparty acknowledgment binding helpers for the IETF Compliance Receipts profile. The binding
 * travels on the signed payload of B's `protectmcp:acknowledgment` receipt and gates verification.
 */

import { createHash } from "node:crypto";

import { canonicalJson } from "./jcs.js";

/**
 * Cross-agent byte-equality binding to an originating receipt: `envelope_hash` over A's full canonical
 * signed envelope, `receipt_ref` to fetch it. `transport_label` is operational, never a basis for trust.
 */
export interface CounterpartyBinding {
  envelope_hash: string;
  receipt_ref: string;
  expect_ack_from?: string;
  transport_label?: string;
}

/**
 * Per-axis outcome of the SDK-side counterparty-binding check; `label` is matches | mismatch |
 * unresolved | kid_mismatch, mirroring the cloud's `counterparty_binding_verified` axis.
 */
export interface CounterpartyBindingVerification {
  valid: boolean;
  envelopeHashMatches: boolean;
  kidMatches: boolean | null;
  label: "matches" | "mismatch" | "unresolved" | "kid_mismatch";
}

/**
 * Base64 SHA-256 of the originating envelope's full JCS bytes, mirroring the cloud's
 * `compute_envelope_hash` so acknowledger, verifier and cloud produce byte-identical digests.
 */
export function computeEnvelopeHash(
  envelope: Record<string, unknown>,
): string {
  return createHash("sha256").update(canonicalJson(envelope)).digest("base64");
}

/** Options for {@link computeCounterpartyBinding}. */
export interface ComputeCounterpartyBindingOptions {
  /**
   * Resolvable locator for A's receipt, defaulting to `payload.action_id` then `signature_id`. Pass an
   * explicit value when the Audit Pack uses a different identifier scheme.
   */
  receiptRef?: string;
  /** Optional declared acknowledger identifier; the verifier cross-checks
   * the acknowledging receipt's `signature.kid` against this value. */
  expectAckFrom?: string;
  /** Optional operational transport hint (`mcp` | `bus` | `http`). */
  transportLabel?: string;
}

/**
 * Build a {@link CounterpartyBinding} over the canonical bytes of the dict as-passed; callers MUST pass
 * the bytes B actually received, not a re-canonicalization, so intermediary tampering is detectable.
 */
export function computeCounterpartyBinding(
  originatingEnvelope: Record<string, unknown>,
  options: ComputeCounterpartyBindingOptions = {},
): CounterpartyBinding {
  let receiptRef = options.receiptRef;
  if (receiptRef === undefined) {
    const payload = originatingEnvelope.payload;
    if (payload && typeof payload === "object") {
      const p = payload as Record<string, unknown>;
      const candidate = p.action_id ?? p.signature_id;
      if (typeof candidate === "string") {
        receiptRef = candidate;
      }
    }
  }
  if (receiptRef === undefined) {
    throw new Error(
      "receipt_ref is required: originating envelope payload has no action_id",
    );
  }
  const binding: CounterpartyBinding = {
    envelope_hash: computeEnvelopeHash(originatingEnvelope),
    receipt_ref: receiptRef,
  };
  if (options.expectAckFrom !== undefined) {
    binding.expect_ack_from = options.expectAckFrom;
  }
  if (options.transportLabel !== undefined) {
    binding.transport_label = options.transportLabel;
  }
  return binding;
}

/**
 * Re-derive and compare the binding's `envelope_hash`, returning a label that separates a tampered
 * handoff from a missing originator. `kidMatches` is null when `expect_ack_from` was not declared.
 */
export function verifyCounterpartyBinding(
  acknowledgmentEnvelope: Record<string, unknown>,
  originatingEnvelope: Record<string, unknown>,
): CounterpartyBindingVerification {
  const payload = acknowledgmentEnvelope.payload;
  if (!payload || typeof payload !== "object") {
    return { valid: false, envelopeHashMatches: false, kidMatches: null, label: "unresolved" };
  }
  const binding = (payload as Record<string, unknown>).counterparty_binding;
  if (!binding || typeof binding !== "object") {
    return { valid: false, envelopeHashMatches: false, kidMatches: null, label: "unresolved" };
  }
  const b = binding as Record<string, unknown>;
  const expectedHash = typeof b.envelope_hash === "string" ? b.envelope_hash : undefined;
  const actualHash = computeEnvelopeHash(originatingEnvelope);
  const envelopeHashMatches = expectedHash === actualHash;

  const expectAckFrom = typeof b.expect_ack_from === "string" ? b.expect_ack_from : undefined;
  let kidMatches: boolean | null = null;
  if (expectAckFrom !== undefined) {
    const sig = acknowledgmentEnvelope.signature;
    const ackKid =
      sig && typeof sig === "object"
        ? ((sig as Record<string, unknown>).kid as unknown)
        : undefined;
    kidMatches = ackKid === expectAckFrom;
  }

  if (!envelopeHashMatches) {
    return { valid: false, envelopeHashMatches: false, kidMatches, label: "mismatch" };
  }
  if (kidMatches === false) {
    return { valid: false, envelopeHashMatches: true, kidMatches: false, label: "kid_mismatch" };
  }
  return { valid: true, envelopeHashMatches: true, kidMatches, label: "matches" };
}
