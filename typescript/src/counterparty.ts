/**
 * Counterparty acknowledgment binding helpers for the IETF Compliance Receipts profile. The binding
 * travels on the signed payload of B's `protectmcp:acknowledgment` receipt and gates verification.
 */

import { createHash } from "node:crypto";

import { canonicalJson } from "./jcs.js";
import {
  exceedsDepth,
  isPreservedContainer,
  isRawFloatLeaf,
  parseJsonPreservingFloats,
} from "./verifier/canonical.js";

/**
 * Byte binding over payload and the exact signature object.
 * transport_label is an operational hint.
 */
export interface CounterpartyBinding {
  envelope_hash: string;
  receipt_ref: string;
  expect_ack_from?: string;
  transport_label?: string;
  scope?: string;
  /** Interaction profile version; absent means the legacy unprofiled path. */
  profile_version?: string;
  /** Exact endorsed terms; `terms_digest` must recompute over it. */
  terms?: Record<string, unknown>;
  /** base64 SHA-256 of the terms object's JCS bytes. */
  terms_digest?: string;
}

/**
 * Per-axis outcome of the SDK-side counterparty-binding check; `label` is matches | mismatch |
 * unresolved | kid_mismatch, mirroring the cloud's `counterparty_binding_verified` axis.
 */
export interface CounterpartyBindingVerification {
  valid: boolean | null;
  envelopeHashMatches: boolean | null;
  kidMatches: boolean | null;
  label: "matches" | "mismatch" | "unresolved" | "kid_mismatch" | "malformed" | "legacy_scope" | "unrecognised_scope" | "profile_unsupported" | null;
}

export const BINDING_SCOPE = "envelope_minus_anchors";

/** Interaction profile version this SDK can speak (must-understand). */
export const PROFILE_VERSION = "0.1";

const BINDING_MEMBERS = new Set([
  "envelope_hash",
  "receipt_ref",
  "expect_ack_from",
  "transport_label",
  "scope",
  "profile_version",
  "terms",
  "terms_digest",
]);

/** SHA-256 over JCS bytes of the exact terms object (matches the cloud). */
export function termsDigestOf(terms: unknown): string {
  if (!isObject(terms)) {
    throw new Error("counterparty_binding_malformed: terms must be an object");
  }
  return createHash("sha256").update(canonicalJson(terms)).digest("base64");
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Hash payload and signature only, keeping the encoded signature spelling. Raw
 *  JSON text is read with the float-preserving parser, as Python reads it. */
export function computeEnvelopeHash(
  origin: unknown,
): string {
  const envelope = typeof origin === "string" ? parseOrigin(origin) : origin;
  if (!isObject(envelope) || !isObject(envelope.payload) || !isObject(envelope.signature)) {
    throw new Error("counterparty_origin_unavailable: signing envelope required");
  }
  const signature = envelope.signature;
  if (["alg", "kid", "sig"].some(key => typeof signature[key] !== "string" || !signature[key])) {
    throw new Error("counterparty_origin_unavailable: signature object is incomplete");
  }
  const projection = { payload: envelope.payload, signature };
  // A parsed RawFloat is one leaf here, as a float is in Python
  if (exceedsDepth(projection, 200)) {
    throw new Error("counterparty_origin_unavailable: nesting exceeds 200 levels");
  }
  if (hasAmbiguousNegativeZero(projection)) {
    throw new Error("counterparty_origin_unavailable: a parsed -0 was -0 or -0.0; pass the raw text");
  }
  return createHash("sha256").update(canonicalBytes(projection)).digest("base64");
}

function parseOrigin(text: string): unknown {
  try {
    return parseJsonPreservingFloats(text);
  } catch (cause) {
    throw new Error("counterparty_origin_unavailable: origin is not strict JSON", { cause });
  }
}

/** A -0 outside a float-preserving parse was -0 (Python's 0) or -0.0 */
function hasAmbiguousNegativeZero(projection: Record<string, unknown>): boolean {
  const stack: unknown[] = [projection];
  while (stack.length > 0) {
    const node = stack.pop();
    if (node === null || typeof node !== "object" || isRawFloatLeaf(node)) continue;
    const children = Object.values(node);
    if (!isPreservedContainer(node) && children.some((v) => Object.is(v, -0))) return true;
    for (const child of children) stack.push(child);
  }
  return false;
}

/** Refuse what the canonicaliser refuses, with the Python error text */
function canonicalBytes(projection: unknown): Uint8Array {
  try {
    return canonicalJson(projection);
  } catch (cause) {
    throw new Error("counterparty_origin_unavailable: canonical bytes unavailable", { cause });
  }
}

/** Options for {@link computeCounterpartyBinding}. */
export interface ComputeCounterpartyBindingOptions {
  /**
   * Pass the originating signature_id for hosted admission;
   * action_id is an offline fallback.
   */
  receiptRef?: string;
  /** Optional declared acknowledger identifier; the verifier cross-checks
   * the acknowledging receipt's `signature.kid` against this value. */
  expectAckFrom?: string;
  /** Optional operational transport hint (`mcp` | `bus` | `http`). */
  transportLabel?: string;
  /** Declared profile version; unknown is refused, never downgraded. */
  profileVersion?: string;
  /** Exact endorsed terms; the emitted digest recomputes over them. */
  terms?: Record<string, unknown>;
}

/**
 * Build from the peer's original signing envelope, retaining the
 * signature object as received.
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
  if (options.profileVersion !== undefined) {
    if (options.profileVersion !== PROFILE_VERSION) {
      throw new Error(
        `counterparty_profile_unsupported: ${JSON.stringify(options.profileVersion)} is not ${JSON.stringify(PROFILE_VERSION)}`,
      );
    }
    binding.profile_version = options.profileVersion;
  }
  if (options.terms !== undefined) {
    binding.terms = options.terms;
    binding.terms_digest = termsDigestOf(options.terms);
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
  const profileVersion = b.profile_version;
  if (profileVersion !== undefined && profileVersion !== null) {
    if (profileVersion !== PROFILE_VERSION) return result(null, "profile_unsupported");
    if (Object.keys(b).some(key => !BINDING_MEMBERS.has(key))) return result(false, "malformed");
  }
  const terms = b.terms;
  const declaredTermsDigest = b.terms_digest;
  if (terms !== undefined || declaredTermsDigest !== undefined) {
    if (typeof declaredTermsDigest !== "string" || !isObject(terms)) {
      return result(false, "malformed");
    }
    let recomputed: string;
    try {
      recomputed = termsDigestOf(terms);
    } catch {
      return result(false, "malformed");
    }
    if (recomputed !== declaredTermsDigest) return result(false, "malformed");
  }
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
