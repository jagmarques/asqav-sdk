/**
 * Offline replay / chain verification for the IETF Compliance Receipts profile.
 *
 * See https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/
 *
 * The verifier walks an ordered list of signed envelopes for one agent,
 * re-derives each predecessor's `previousReceiptHash`, and reports any
 * mismatch as a chain break. Equivalent to the Python SDK's
 * `replay.ReplayTimeline.verify_chain` rewritten for the v2 chain shape.
 *
 * v2 (default) chain link:
 *
 *   previousReceiptHash[i+1] = sha256(canonicalJson(envelope[i]))   // 64 hex
 *   previousReceiptHash[0]   = "0".repeat(64)
 *
 * Legacy chain link (kept under `legacy: true` for one release):
 *
 *   prev_hash[i+1] = sha256(json.dumps(
 *     {signature_id, action_type, timestamp, prev_hash}, sort_keys=True
 *   ))
 *
 * This module is import-free of any HTTP / network code so it can run
 * over a downloaded ComplianceBundle.
 */

import { createHash } from "node:crypto";

import { canonicalJson } from "./jcs.js";

/** Seed value for the first record on every chain. */
export const FIRST_RECEIPT_SEED = "0".repeat(64);

/** A single record in the chain to verify. */
export interface ChainRecord {
  /** The full signed envelope (the bytes the cloud signed, as a JSON
   * object). Must include `previousReceiptHash`. Other fields are
   * passed through as-is into the JCS canonical bytes. */
  signedEnvelope: Record<string, unknown>;
  /** Optional pre-computed chain hash for this record. When omitted,
   * the verifier derives it from the envelope. */
  chainHash?: string;
}

/** Per-record outcome. */
export interface ChainStepResult {
  index: number;
  /** True iff `signedEnvelope.previousReceiptHash` matches the predecessor's
   * derived hash (or the seed for index 0). */
  chainValid: boolean;
  /** What the predecessor's chain hash should have been. */
  expectedPreviousReceiptHash: string;
  /** What this record claims its predecessor was. */
  actualPreviousReceiptHash: string;
  /** This record's derived chain hash. */
  derivedChainHash: string;
  /** When `chainHash` was supplied on the input, true if it matches the
   * derived value (catches storage-side mutation). Undefined when not
   * supplied. */
  storedChainHashMatches?: boolean;
}

/** Aggregate outcome. */
export interface ChainVerificationResult {
  chainIntegrity: boolean;
  steps: ChainStepResult[];
}

export interface VerifyChainOptions {
  /** Use the legacy v1 chain shape instead of the v2 / IETF shape.
   * Kept for one release per the backward-compat window. */
  legacy?: boolean;
}

/**
 * Re-derive the chain over an ordered list of signed envelopes for one
 * agent and return per-step + aggregate validity.
 *
 * The list MUST be in chain order (oldest first). Mixing agents yields
 * `chainIntegrity=false` because the predecessor links won't match.
 */
export function verifyChain(
  records: ChainRecord[],
  options: VerifyChainOptions = {},
): ChainVerificationResult {
  if (options.legacy === true) {
    return verifyChainLegacy(records);
  }

  const steps: ChainStepResult[] = [];
  let allValid = true;
  let expectedPrev = FIRST_RECEIPT_SEED;

  for (let i = 0; i < records.length; i++) {
    const record = records[i];
    const env = record.signedEnvelope;
    const actualPrev = String(env.previousReceiptHash ?? env.previous_receipt_hash ?? "");
    const derived = sha256Hex(canonicalJson(env));

    const chainValid = actualPrev === expectedPrev;
    if (!chainValid) allValid = false;

    let storedChainHashMatches: boolean | undefined;
    if (record.chainHash !== undefined) {
      storedChainHashMatches = record.chainHash === derived;
      if (!storedChainHashMatches) allValid = false;
    }

    steps.push({
      index: i,
      chainValid,
      expectedPreviousReceiptHash: expectedPrev,
      actualPreviousReceiptHash: actualPrev,
      derivedChainHash: derived,
      storedChainHashMatches,
    });

    expectedPrev = derived;
  }

  return { chainIntegrity: allValid, steps };
}

/**
 * Convenience: derive the v2 chain hash for one signed envelope.
 *
 *   sha256(canonicalJson(envelope))
 */
export function deriveChainHash(envelope: Record<string, unknown>): string {
  return sha256Hex(canonicalJson(envelope));
}

function sha256Hex(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

// === Pre-IETF v1 chain shape (retained for one release) ===

interface LegacyMinimalRecord {
  signature_id?: string;
  action_type?: string;
  timestamp?: number | string;
  signed_at?: number | string;
}

function verifyChainLegacy(records: ChainRecord[]): ChainVerificationResult {
  const steps: ChainStepResult[] = [];
  let allValid = true;
  let prevHash = "";

  for (let i = 0; i < records.length; i++) {
    const env = records[i].signedEnvelope as LegacyMinimalRecord & Record<string, unknown>;
    const actualPrev = String(env.previousReceiptHash ?? env.previous_hash ?? "");

    const chainValid = i === 0 ? true : actualPrev === prevHash;
    if (!chainValid) allValid = false;

    const entry = JSON.stringify({
      signature_id: env.signature_id ?? "",
      action_type: env.action_type ?? "",
      timestamp: env.signed_at ?? env.timestamp ?? 0,
      prev_hash: prevHash,
    });
    // JSON.stringify in JS does not sort keys; the Python legacy form
    // uses sort_keys=True. We match Python's output by sorting.
    const sortedEntry = JSON.stringify(JSON.parse(entry), Object.keys(JSON.parse(entry)).sort());
    const derived = sha256Hex(new TextEncoder().encode(sortedEntry));

    steps.push({
      index: i,
      chainValid,
      expectedPreviousReceiptHash: i === 0 ? "" : prevHash,
      actualPreviousReceiptHash: actualPrev,
      derivedChainHash: derived,
    });

    prevHash = derived;
  }

  return { chainIntegrity: allValid, steps };
}
