/**
 * Offline replay / chain verification for the IETF Compliance Receipts profile.
 *
 * See https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/
 *
 * The verifier walks an ordered list of signed envelopes for one agent,
 * re-derives each predecessor's `previousReceiptHash`, and reports any
 * mismatch as a chain break. Equivalent to the Python SDK's
 * `replay.ReplayTimeline.verify_chain`.
 *
 * Chain link:
 *
 *   previousReceiptHash[i+1] = sha256(canonicalJson(envelope[i]))   // 64 hex
 *   previousReceiptHash[0]   = "0".repeat(64)
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

export type VerifyChainOptions = Record<string, never>;

/**
 * Re-derive the chain over an ordered list of signed envelopes for one
 * agent and return per-step + aggregate validity.
 *
 * The list MUST be in chain order (oldest first). Mixing agents yields
 * `chainIntegrity=false` because the predecessor links won't match.
 */
export function verifyChain(
  records: ChainRecord[],
  _options: VerifyChainOptions = {},
): ChainVerificationResult {
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
 * Convenience: derive the chain hash for one signed envelope.
 *
 *   sha256(canonicalJson(envelope))
 */
export function deriveChainHash(envelope: Record<string, unknown>): string {
  return sha256Hex(canonicalJson(envelope));
}

function sha256Hex(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}
