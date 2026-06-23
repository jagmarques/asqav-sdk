/**
 * Algorithm agility for receipt signing.
 *
 * The cloud accepts `ml-dsa-44`, `ml-dsa-65` (default), `ml-dsa-87`,
 * `ed25519`, and `es256`. All three ML-DSA variants are server-side only
 * (node:crypto has no ML-DSA); the local verifier returns SKIPPED for
 * them. Ed25519 and ES256 work locally: the SDK can generate keypairs and
 * sign off-cloud (handy for `user_intent` envelopes and test fixtures).
 *
 * Public surface:
 *   SUPPORTED_ALGORITHMS   - the set the SDK accepts on `Agent.create`.
 *   isSupportedAlgorithm   - tiny type-guard.
 *   generateKeypair        - Ed25519 / ES256 keypair (raw bytes + b64).
 *   signMessage            - sign canonical bytes with the local key.
 *   verifyMessage          - verify a local signature.
 */

import {
  createPrivateKey,
  createPublicKey,
  generateKeyPairSync,
  sign,
  verify,
} from "node:crypto";

export const SUPPORTED_ALGORITHMS = [
  "ml-dsa-65",
  "ml-dsa-44",
  "ml-dsa-87",
  "ed25519",
  "es256",
] as const;

export type SupportedAlgorithm = (typeof SUPPORTED_ALGORITHMS)[number];

/** Algorithms the SDK can keypair-generate and sign with locally. */
export const LOCAL_SIGNING_ALGORITHMS = ["ed25519", "es256"] as const;

export type LocalSigningAlgorithm = (typeof LOCAL_SIGNING_ALGORITHMS)[number];

export function isSupportedAlgorithm(value: string): value is SupportedAlgorithm {
  return (SUPPORTED_ALGORITHMS as readonly string[]).includes(value);
}

export function isLocalSigningAlgorithm(
  value: string,
): value is LocalSigningAlgorithm {
  return (LOCAL_SIGNING_ALGORITHMS as readonly string[]).includes(value);
}

/** Public + private bytes for a locally-generated keypair. */
export interface LocalKeypair {
  algorithm: LocalSigningAlgorithm;
  /** PKCS8 DER private key (base64). */
  privateKeyPkcs8B64: string;
  /** SPKI DER public key (base64). */
  publicKeySpkiB64: string;
}

/**
 * Generate a fresh keypair for a local-signing algorithm. ML-DSA is
 * intentionally rejected here because it is server-side only.
 */
export function generateKeypair(algorithm: LocalSigningAlgorithm): LocalKeypair {
  if (algorithm === "ed25519") {
    const { publicKey, privateKey } = generateKeyPairSync("ed25519");
    return {
      algorithm,
      privateKeyPkcs8B64: privateKey
        .export({ type: "pkcs8", format: "der" })
        .toString("base64"),
      publicKeySpkiB64: publicKey
        .export({ type: "spki", format: "der" })
        .toString("base64"),
    };
  }
  if (algorithm === "es256") {
    const { publicKey, privateKey } = generateKeyPairSync("ec", {
      namedCurve: "P-256",
    });
    return {
      algorithm,
      privateKeyPkcs8B64: privateKey
        .export({ type: "pkcs8", format: "der" })
        .toString("base64"),
      publicKeySpkiB64: publicKey
        .export({ type: "spki", format: "der" })
        .toString("base64"),
    };
  }
  // Exhaustiveness guard.
  const exhaustive: never = algorithm;
  throw new Error(`Unsupported local algorithm: ${exhaustive as string}`);
}

/**
 * Sign canonical bytes with a locally-generated keypair.
 *
 * Returns the raw signature as base64. For `es256`, the output is the
 * IEEE P1363 (raw r||s) form the cloud's `_verify_signature` expects;
 * Node's default for ECDSA is DER, so we convert.
 */
export function signMessage(
  algorithm: LocalSigningAlgorithm,
  privateKeyPkcs8B64: string,
  message: Uint8Array,
): string {
  const key = createPrivateKey({
    key: Buffer.from(privateKeyPkcs8B64, "base64"),
    format: "der",
    type: "pkcs8",
  });
  if (algorithm === "ed25519") {
    return sign(null, Buffer.from(message), key).toString("base64");
  }
  if (algorithm === "es256") {
    const der = sign("sha256", Buffer.from(message), {
      key,
      dsaEncoding: "ieee-p1363",
    });
    return der.toString("base64");
  }
  const exhaustive: never = algorithm;
  throw new Error(`Unsupported local algorithm: ${exhaustive as string}`);
}

/** Verify a base64 signature against a base64 SPKI public key. */
export function verifyMessage(
  algorithm: LocalSigningAlgorithm,
  publicKeySpkiB64: string,
  message: Uint8Array,
  signatureB64: string,
): boolean {
  const key = createPublicKey({
    key: Buffer.from(publicKeySpkiB64, "base64"),
    format: "der",
    type: "spki",
  });
  const sig = Buffer.from(signatureB64, "base64");
  if (algorithm === "ed25519") {
    return verify(null, Buffer.from(message), key, sig);
  }
  if (algorithm === "es256") {
    return verify(
      "sha256",
      Buffer.from(message),
      { key, dsaEncoding: "ieee-p1363" },
      sig,
    );
  }
  const exhaustive: never = algorithm;
  throw new Error(`Unsupported local algorithm: ${exhaustive as string}`);
}
