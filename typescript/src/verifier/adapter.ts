/** Format-adapter seam, port of the oracle's `adapter.py`: format code sits in
 * it; canonicaliser, verify, chain walk and vector runner are shared */

import type { VerifyState } from "./crypto.js";

/** The bytes and metadata needed to check one issuer signature. */
export interface SignatureMaterial {
  /** Raw signature bytes (already decoded from hex / base64url / multibase). */
  sig: Uint8Array;
  /** Algorithm token routed through `verifySignature`. */
  alg: string;
  /** Key identifier used by `resolveKey` to find the public key. */
  kid: string;
}

/** The hash-chain linkage for one receipt. */
export interface ChainStep {
  /** Previous-hash field value, or null when omitted on genesis; unknown as the
   * value is producer-set: a non-string must reach the chain axis as-is */
  prevField: unknown;
  /** True when this receipt declares itself first on its chain. */
  isGenesis: boolean;
  /** Recompute the predecessor's chain digest (the bytes the format hashes) */
  recompute: (predecessor: Record<string, unknown>) => string;
}

/** `(result, note)` pair returned by structural and extra-axis checks. */
export type AxisCheck = readonly [VerifyState, string];

/** A named extra axis: `[axisName, result, note]`. */
export type ExtraAxis = readonly [string, VerifyState, string];

/** Format-shaped key material the runner injects (JWKS, key map, or null) */
export type KeyProvider = Record<string, unknown> | null;

/** Comparison input local to one invocation; never a key provider. */
export interface VerificationContext {
  readonly originatingEnvelope?: unknown;
  /** Wall-clock override the time axes evaluate against; undefined = now. */
  readonly evaluationTime?: unknown;
  /** Trust roots the caller evaluated against (attribution only). */
  readonly trustRoots?: unknown;
  /** Revocation snapshot the caller evaluated against (attribution only). */
  readonly revocationSnapshot?: unknown;
  /** Policy revision the caller evaluated against (attribution only). */
  readonly policyRevision?: unknown;
  /** Evidence cutoff the caller evaluated against (attribution only). */
  readonly evidenceKnowledgeCutoff?: unknown;
  /** Event time the caller evaluated against (attribution only). */
  readonly eventTime?: unknown;
  /** Effective time the caller evaluated against (attribution only). */
  readonly effectiveTime?: unknown;
  /** Who ran the evaluation (attribution only). */
  readonly evaluator?: unknown;
}

/** One receipt format's binding to the shared core. `name` labels it in results
 * and the manifest; `detect` is a cheap structural fingerprint, no crypto */
export abstract class FormatAdapter {
  /** Stable format label surfaced in results and the vector manifest. */
  abstract readonly name: string;

  /** Cheap structural test: does this adapter own `doc`? No crypto. */
  abstract detect(doc: Record<string, unknown>): boolean;

  /** Pull the issuer signature bytes plus alg and kid out of `doc`. */
  abstract extractSignature(doc: Record<string, unknown>): SignatureMaterial;

  /** Return `[publicKeyBytesOrNull, note]` for the receipt's signer. */
  abstract resolveKey(
    doc: Record<string, unknown>,
    keyProvider: KeyProvider,
  ): readonly [Uint8Array | null, string];

  /** Return the exact bytes the signature covers. */
  abstract signingInput(doc: Record<string, unknown>): Uint8Array;

  /** Describe how this receipt links to its predecessor. */
  abstract chainStep(doc: Record<string, unknown>): ChainStep;

  /** Structural check; returns `[result, note]`. */
  abstract schema(doc: Record<string, unknown>): AxisCheck;

  /** Axes beyond structure/signature/chain as [axisName, result, note]; default
   * none. One tuple per REQUIRED layer, so a missing layer FAILs the verdict */
  extraAxes(_doc: Record<string, unknown>, _keyProvider: KeyProvider): ExtraAxis[] {
    return [];
  }

  extraAxesWithContext(
    doc: Record<string, unknown>, keyProvider: KeyProvider, _context: VerificationContext,
  ): ExtraAxis[] {
    return this.extraAxes(doc, keyProvider);
  }

  // Raw so the axis can report a malformed counter; extraAxes never sees the
  // predecessor continuity needs, so the counter is read here
  seqOf(_doc: Record<string, unknown>): unknown {
    return null;
  }

  // In-signed-body attestation (e.g. signer) in the verdict, not a check
  // (default none)
  attestation(_doc: Record<string, unknown>): Record<string, unknown> {
    return {};
  }

  /** True when the digest is keyed: consistent, not third-party re-derivable,
   * so a fully-checked receipt reports `verified_keyed`. Default false */
  keyedDigest(_doc: Record<string, unknown>): boolean {
    return false;
  }

  /** Bytes under the pre-cutover canonical dialect, else null; a signature
   * passing only under these is reported as that dialect, never verified */
  preCutoverSigningInput(_doc: Record<string, unknown>): Uint8Array | null {
    return null;
  }

  /** Parts whose numbers are signed (`chained`: what a successor's link
   *  hashes); default the whole document */
  signedParts(doc: Record<string, unknown>, _chained = false): unknown[] {
    return [doc];
  }

  /** Refusal note or null; default no-op, the core reports a note early. */
  profilePrecheck(
    _doc: Record<string, unknown>,
    _predecessor: Record<string, unknown> | null = null,
    _predecessorFmt: string | null = null,
    _originatingEnvelope: unknown = null,
  ): string | null {
    return null;
  }
}
