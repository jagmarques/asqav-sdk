/**
 * Signature-verify dispatch shared across format adapters - a port of the Python
 * oracle's `verifier/oracle/crypto.py`.
 *
 * One entry point, `verifySignature(alg, pk, msg, sig)`, returns a 3-state
 * `{result, note}` where result is `PASS` / `FAIL` / `SKIPPED`, so the oracle
 * never prints a PASS for a signature it could not actually check.
 *
 * Algorithm wiring:
 *   - `Ed25519` / `EdDSA` : `node:crypto.verify` over a raw 32-byte public key.
 *   - `ES256`             : ECDSA P-256 / SHA-256 over a 65-byte uncompressed
 *                           point; signature is the 64-byte raw r||s form
 *                           (ieee-p1363), which `node:crypto` verifies directly.
 *   - `ML-DSA-65`         : `@noble/post-quantum` (MIT, pure JS). API:
 *                           `verify(sig, msg, publicKey)` -> boolean.
 *                           Public key is 1952 raw bytes; signature is 3309 bytes.
 *                           Byte-compatible with FIPS 204 / dilithium-py.
 *
 * Malformed key or signature bytes -> FAIL, never throw.
 */

import { createPublicKey, verify } from "node:crypto";
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";

import { INVALID, PASS, SKIPPED, UNVERIFIABLE } from "./taxonomy.js";

export { INVALID, PASS, SKIPPED, UNVERIFIABLE };

/**
 * Axis-result tokens under the criterion 418 taxonomy. FAIL is gone: every
 * failure names its class, INVALID (a binding the check refuted) or
 * UNVERIFIABLE (a recomputation that could not complete), and SKIPPED survives
 * only for axes that do not apply to the receipt at all.
 */
export type VerifyState = "PASS" | "INVALID" | "UNVERIFIABLE" | "SKIPPED";

export interface VerifyOutcome {
  result: VerifyState;
  note: string;
  /** Closed failure-class token; "none" when nothing failed. */
  reasonCode: string;
}

/** ML-DSA-65 (FIPS 204) wire lengths; a short input is malformed, never checked. */
const MLDSA65_PK_LEN = 1952;
const MLDSA65_SIG_LEN = 3309;

/** Raw Ed25519 wire lengths (RFC 8032). */
const ED25519_PK_LEN = 32;
const ED25519_SIG_LEN = 64;

/** Raw ES256 wire lengths: the 65-byte uncompressed point and 64-byte r||s. */
const ES256_PK_LEN = 65;
const ES256_SIG_LEN = 64;

const SPKI_ED25519_PREFIX = Buffer.from("302a300506032b6570032100", "hex");

function base64url(buf: Buffer): string {
  return buf.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/**
 * ML-DSA-65 verify via @noble/post-quantum (MIT, pure JS/WASM-free).
 *
 * Noble's API: verify(sig, msg, publicKey) -> boolean.
 * Key is 1952 raw bytes; signature is 3309 bytes (FIPS 204 ML-DSA-65).
 * Byte-compatible with Python dilithium-py.
 */
function verifyMlDsa65(pk: Uint8Array, msg: Uint8Array, sig: Uint8Array): VerifyOutcome {
  if (pk.length !== MLDSA65_PK_LEN) {
    return {
      result: UNVERIFIABLE,
      note: `bad ML-DSA-65 public key: expected ${MLDSA65_PK_LEN} bytes, got ${pk.length}`,
      reasonCode: "key_malformed",
    };
  }
  if (sig.length !== MLDSA65_SIG_LEN) {
    return {
      result: UNVERIFIABLE,
      note: `bad ML-DSA-65 signature: expected ${MLDSA65_SIG_LEN} bytes, got ${sig.length}`,
      reasonCode: "signature_malformed",
    };
  }
  try {
    const ok = ml_dsa65.verify(sig, msg, pk);
    return ok
      ? { result: PASS, note: "signature valid", reasonCode: "none" }
      : { result: INVALID, note: "signature mismatch", reasonCode: "signature_mismatch" };
  } catch (exc) {
    return {
      result: UNVERIFIABLE,
      note: `verify error: ${(exc as Error).message}`,
      reasonCode: "signature_malformed",
    };
  }
}

/** Ed25519 verify over a raw 32-byte public key (RFC 8032). */
function verifyEd25519(pk: Uint8Array, msg: Uint8Array, sig: Uint8Array): VerifyOutcome {
  if (pk.length !== ED25519_PK_LEN) {
    return {
      result: UNVERIFIABLE,
      note: `bad Ed25519 public key: expected ${ED25519_PK_LEN} bytes, got ${pk.length}`,
      reasonCode: "key_malformed",
    };
  }
  if (sig.length !== ED25519_SIG_LEN) {
    return {
      result: UNVERIFIABLE,
      note: `bad Ed25519 signature: expected ${ED25519_SIG_LEN} bytes, got ${sig.length}`,
      reasonCode: "signature_malformed",
    };
  }
  let key;
  try {
    const spki = Buffer.concat([SPKI_ED25519_PREFIX, Buffer.from(pk)]);
    key = createPublicKey({ key: spki, format: "der", type: "spki" });
  } catch (exc) {
    return {
      result: UNVERIFIABLE,
      note: `bad Ed25519 public key: ${(exc as Error).message}`,
      reasonCode: "key_malformed",
    };
  }
  try {
    const ok = verify(null, Buffer.from(msg), key, Buffer.from(sig));
    return ok
      ? { result: PASS, note: "signature valid", reasonCode: "none" }
      : { result: INVALID, note: "signature mismatch", reasonCode: "signature_mismatch" };
  } catch (exc) {
    return {
      result: UNVERIFIABLE,
      note: `verify error: ${(exc as Error).message}`,
      reasonCode: "signature_malformed",
    };
  }
}

/**
 * ES256 (ECDSA P-256 over SHA-256) verify.
 *
 * The public key is the 65-byte uncompressed point (0x04 || X || Y). The
 * signature is the 64-byte raw r||s (ieee-p1363) form WebCrypto / JOSE emit,
 * which `node:crypto.verify` accepts with `dsaEncoding: "ieee-p1363"`.
 */
function verifyEs256(pk: Uint8Array, msg: Uint8Array, sig: Uint8Array): VerifyOutcome {
  if (pk.length !== ES256_PK_LEN || pk[0] !== 0x04) {
    return {
      result: UNVERIFIABLE,
      note: `bad P-256 public key: expected ${ES256_PK_LEN}-byte uncompressed point, got ${pk.length}`,
      reasonCode: "key_malformed",
    };
  }
  if (sig.length !== ES256_SIG_LEN) {
    return {
      result: UNVERIFIABLE,
      note: `ES256 signature must be ${ES256_SIG_LEN}-byte raw r||s, got ${sig.length}`,
      reasonCode: "signature_malformed",
    };
  }
  let key;
  try {
    const x = base64url(Buffer.from(pk.slice(1, 33)));
    const y = base64url(Buffer.from(pk.slice(33, 65)));
    key = createPublicKey({ key: { kty: "EC", crv: "P-256", x, y }, format: "jwk" });
  } catch (exc) {
    return {
      result: UNVERIFIABLE,
      note: `bad P-256 public key: ${(exc as Error).message}`,
      reasonCode: "key_malformed",
    };
  }
  try {
    const ok = verify("sha256", Buffer.from(msg), { key, dsaEncoding: "ieee-p1363" }, Buffer.from(sig));
    return ok
      ? { result: PASS, note: "signature valid", reasonCode: "none" }
      : { result: INVALID, note: "signature mismatch", reasonCode: "signature_mismatch" };
  } catch (exc) {
    return {
      result: UNVERIFIABLE,
      note: `verify error: ${(exc as Error).message}`,
      reasonCode: "signature_malformed",
    };
  }
}

type Verifier = (pk: Uint8Array, msg: Uint8Array, sig: Uint8Array) => VerifyOutcome;

const DISPATCH: Record<string, Verifier> = {
  "ML-DSA-65": verifyMlDsa65,
  ED25519: verifyEd25519,
  EDDSA: verifyEd25519,
  ES256: verifyEs256,
};

/** Dispatch to the algorithm's verifier; an unsupported alg cannot be recomputed. */
export function verifySignature(
  alg: string,
  pk: Uint8Array,
  msg: Uint8Array,
  sig: Uint8Array,
): VerifyOutcome {
  const fn = DISPATCH[(typeof alg === "string" ? alg : "").toUpperCase()];
  if (fn === undefined) {
    const known = Object.keys(DISPATCH).sort().join(", ");
    return {
      result: UNVERIFIABLE,
      note: `unsupported alg '${alg}' (oracle checks ${known})`,
      reasonCode: "algorithm_unsupported",
    };
  }
  return fn(pk, msg, sig);
}

import { createHash } from "node:crypto";

/** Lowercase hex SHA-256 - the chain primitive every format shares. */
export function sha256Hex(data: Uint8Array): string {
  return createHash("sha256").update(Buffer.from(data)).digest("hex");
}
