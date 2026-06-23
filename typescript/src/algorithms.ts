// Algorithm agility for receipt signing.

// SUPPORTED_ALGORITHMS is the client-side validation set (all five:
// ml-dsa-65 default, ml-dsa-44, ml-dsa-87, ed25519, es256), so Agent.create
// passes any of the five past local validation.

// The cloud signs server-side only with ml-dsa-{44,65,87}. ed25519/es256
// clear local validation but the cloud returns HTTP 400 on Agent.create
// (see index.ts create comment + both READMEs).

// ed25519/es256 are for LOCAL keypair generation and local sign/verify via
// the node:crypto helpers below (LOCAL_SIGNING_ALGORITHMS), not cloud signing.

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

// Generate a fresh keypair for a local-signing algorithm.
// ML-DSA is rejected here because it is server-side only.
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

// Sign canonical bytes with a locally-generated keypair, raw signature as
// base64. es256 emits IEEE P1363 (raw r||s) to match the cloud verifier.
// Node defaults ECDSA to DER, so we convert.
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
