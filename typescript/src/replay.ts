/**
 * Offline replay / chain verification for the IETF Compliance Receipts profile: walk ordered envelopes
 * for one agent, re-derive each `previousReceiptHash`, report mismatches. Network-free by design.
 */

import { createHash } from "node:crypto";

import { canonicalJson } from "./jcs.js";

/** Seed value for the first record on every chain. */
export const FIRST_RECEIPT_SEED = "0".repeat(64);

/** A single record in the chain to verify. */
export interface ChainRecord {
  /**
   * The full signed envelope as a JSON object; must include `previousReceiptHash`. Other fields pass
   * through as-is into the JCS canonical bytes.
   */
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
  /**
   * When `chainHash` was supplied on the input, true if it matches the derived value, catching
   * storage-side mutation. Undefined when not supplied.
   */
  storedChainHashMatches?: boolean;
}

/** Aggregate outcome. */
export interface ChainVerificationResult {
  chainIntegrity: boolean;
  steps: ChainStepResult[];
}

/**
 * Reserved for future verifier knobs. No keys are accepted today; any unrecognized key throws so
 * silent-fallback bugs surface immediately.
 */
export type VerifyChainOptions = Record<string, never>;

/**
 * Re-derive the chain over an ordered list of signed envelopes for one agent (oldest first) and return
 * per-step plus aggregate validity. Throws TypeError on any unrecognized `options` key.
 */
export function verifyChain(
  records: ChainRecord[],
  options: VerifyChainOptions = {},
): ChainVerificationResult {
  const unknownKeys = Object.keys(options ?? {});
  if (unknownKeys.length > 0) {
    throw new TypeError(
      `verifyChain: unsupported option(s) ${unknownKeys.map((k) => JSON.stringify(k)).join(", ")}. ` +
        "The verifier accepts no options today. " +
        "is not honored and would have silently fallen back to v2 verification. " +
        "Migrate bundles to carry `signedEnvelope` on every step.",
    );
  }

  const steps: ChainStepResult[] = [];
  let allValid = true;
  let expectedPrev = FIRST_RECEIPT_SEED;

  for (let i = 0; i < records.length; i++) {
    const record = records[i];
    const env = record.signedEnvelope;
    const chainInput = payloadForChain(env);
    const actualPrev = String(
      chainInput.previousReceiptHash ?? chainInput.previous_receipt_hash ?? "",
    );
    const derived = sha256Hex(canonicalJson(chainInput));

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
 *   sha256(canonicalJson(payload))   // bundle-shaped envelopes are unwrapped
 */
export function deriveChainHash(envelope: Record<string, unknown>): string {
  return sha256Hex(canonicalJson(payloadForChain(envelope)));
}

/**
 * Return the inner compliance payload the cloud hashes for the chain link, unwrapping the bundle-shaped
 * `{payload, signature, anchors}` wrapper when present.
 */
function payloadForChain(
  envelope: Record<string, unknown>,
): Record<string, unknown> {
  const inner = envelope.payload;
  if (
    inner !== null &&
    typeof inner === "object" &&
    !Array.isArray(inner) &&
    ("signature" in envelope || "anchors" in envelope)
  ) {
    return inner as Record<string, unknown>;
  }
  return envelope;
}

function sha256Hex(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}
