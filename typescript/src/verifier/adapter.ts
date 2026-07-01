/**
 * The format-adapter seam - a port of the Python oracle's `verifier/oracle/adapter.py`.
 *
 * Everything format-specific lives behind this seam; everything shared (the JCS
 * canonicaliser, the Ed25519 / ES256 verify, the SHA-256 chain walk, the vector
 * runner) is infrastructure the core drives.
 */

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
  /**
   * The value of the format's previous-hash field, or null when the format omits
   * it on genesis (AERF) - distinct from an explicit-null genesis.
   */
  prevField: string | null;
  /** True when this receipt declares itself first on its chain. */
  isGenesis: boolean;
  /** Recompute the predecessor's chain digest (encodes which bytes the format hashes). */
  recompute: (predecessor: Record<string, unknown>) => string;
}

/** `(result, note)` pair returned by structural and extra-axis checks. */
export type AxisCheck = readonly [VerifyState, string];

/** A named extra axis: `[axisName, result, note]`. */
export type ExtraAxis = readonly [string, VerifyState, string];

/** Format-shaped key material the runner injects (a JWKS dict, a key map, or null). */
export type KeyProvider = Record<string, unknown> | null;

/**
 * One receipt format's binding to the shared verification core. Subclasses
 * implement the six methods; `name` labels the format in results and the manifest;
 * `detect` is a cheap structural fingerprint (no crypto).
 */
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

  /**
   * Format-specific axes beyond structure / issuer-signature / chain. Returns a
   * list of `[axisName, result, note]`. The default is none; a format whose
   * verification procedure layers additional REQUIRED checks returns one tuple
   * per check so a missing-but-required or invalid layer FAILs the verdict.
   */
  extraAxes(_doc: Record<string, unknown>, _keyProvider: KeyProvider): ExtraAxis[] {
    return [];
  }

  // In-body attestation surfaced in the verdict (e.g. signer). Not a check.
  // Default none; a format with an in-signed-body attestation returns it.
  attestation(_doc: Record<string, unknown>): Record<string, unknown> {
    return {};
  }
}
