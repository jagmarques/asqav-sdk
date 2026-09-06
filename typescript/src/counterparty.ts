/**
 * Counterparty acknowledgment binding helpers for the IETF Compliance Receipts profile. The binding
 * travels on the signed payload of B's `protectmcp:acknowledgment` receipt and gates verification.
 */

import { createHash } from "node:crypto";

import { canonicalJson } from "./jcs.js";

/**
 * Byte binding over payload and the exact signature object; transport_label is an operational hint.
 */
export interface CounterpartyBinding {
  envelope_hash: string;
  receipt_ref: string;
  expect_ack_from?: string;
  transport_label?: string;
  scope?: string;
}

/**
 * Per-axis outcome of the SDK-side counterparty-binding check; `label` is matches | mismatch |
 * unresolved | kid_mismatch, mirroring the cloud's `counterparty_binding_verified` axis.
 */
export interface CounterpartyBindingVerification {
  valid: boolean | null;
  envelopeHashMatches: boolean | null;
  kidMatches: boolean | null;
  label: "matches" | "mismatch" | "unresolved" | "kid_mismatch" | "malformed" | "legacy_scope" | "unrecognised_scope" | null;
}

export const BINDING_SCOPE = "envelope_minus_anchors";

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Hash exactly payload and signature, preserving the encoded signature spelling. */
export function computeEnvelopeHash(
  envelope: unknown,
): string {
  if (!isObject(envelope) || !isObject(envelope.payload) || !isObject(envelope.signature)) {
    throw new Error("counterparty_origin_unavailable: signing envelope required");
  }
  const signature = envelope.signature;
  if (["alg", "kid", "sig"].some(key => typeof signature[key] !== "string" || !signature[key])) {
    throw new Error("counterparty_origin_unavailable: signature object is incomplete");
  }
  const projection = { payload: envelope.payload, signature };
  const stack: [unknown, number][] = [[projection, 0]];
  while (stack.length) {
    const [node, depth] = stack.pop()!;
    if (depth > 200) throw new Error("counterparty_origin_unavailable: nesting exceeds 200 levels");
    const children = isObject(node) ? Object.values(node) : Array.isArray(node) ? node : [];
    for (const child of children) stack.push([child, depth + 1]);
  }
  return createHash("sha256").update(canonicalJson(projection)).digest("base64");
}

/** Options for {@link computeCounterpartyBinding}. */
export interface ComputeCounterpartyBindingOptions {
  /**
   * Pass the originating signature_id for hosted admission; action_id is an offline fallback.
   */
  receiptRef?: string;
  /** Optional declared acknowledger identifier; the verifier cross-checks
   * the acknowledging receipt's `signature.kid` against this value. */
  expectAckFrom?: string;
  /** Optional operational transport hint (`mcp` | `bus` | `http`). */
  transportLabel?: string;
}

/**
 * Build from the peer's original signing envelope, retaining the signature object as received.
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
    scope: BINDING_SCOPE,
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
  originatingEnvelope: unknown = null,
): CounterpartyBindingVerification {
  const result = (valid: boolean | null, label: CounterpartyBindingVerification["label"]): CounterpartyBindingVerification =>
    ({ valid, envelopeHashMatches: valid, kidMatches: null, label });
  const payload = acknowledgmentEnvelope?.payload;
  if (!isObject(payload)) {
    return result(false, "malformed");
  }
  if (!Object.hasOwn(payload, "counterparty_binding")) {
    return result(null, null);
  }
  const b = payload.counterparty_binding;
  if (!isObject(b)) return result(false, "malformed");
  if (!Object.hasOwn(b, "scope")) return result(null, "legacy_scope");
  if (b.scope !== BINDING_SCOPE) return result(null, "unrecognised_scope");
  const expectedHash = bindingDigest(b);
  if (expectedHash === null) return result(false, "malformed");
  let actualHash: Buffer;
  try {
    actualHash = Buffer.from(computeEnvelopeHash(originatingEnvelope), "base64");
  } catch {
    return result(null, "unresolved");
  }
  const envelopeHashMatches = expectedHash.equals(actualHash);

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

function bindingDigest(binding: Record<string, unknown>): Buffer | null {
  if (typeof binding.receipt_ref !== "string" || !binding.receipt_ref) return null;
  if (["expect_ack_from", "transport_label"].some(key => binding[key] != null && typeof binding[key] !== "string")) return null;
  const value = binding.envelope_hash;
  if (typeof value !== "string" || !/^[A-Za-z0-9+/_-]+={0,2}$/.test(value) || (value.includes("=") && value.length % 4 !== 0)) return null;
  const decoded = Buffer.from(value, "base64");
  return decoded.length === 32 ? decoded : null;
}
