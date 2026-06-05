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
 *   - `ML-DSA-65`         : SKIPPED. Node has no ML-DSA, so the verifier reports
 *                           it cannot check the post-quantum path - never a
 *                           false PASS, never a FAIL (mirrors the Python oracle
 *                           without dilithium-py).
 *
 * Malformed key or signature bytes -> FAIL, never throw.
 */

import { createPublicKey, verify } from "node:crypto";

export const PASS = "PASS";
export const FAIL = "FAIL";
export const SKIPPED = "SKIPPED";

export type VerifyState = "PASS" | "FAIL" | "SKIPPED";

export interface VerifyOutcome {
  result: VerifyState;
  note: string;
}

const SPKI_ED25519_PREFIX = Buffer.from("302a300506032b6570032100", "hex");

function base64url(buf: Buffer): string {
  return buf.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** ML-DSA-65: Node has no implementation, so SKIP - never a false PASS. */
function verifyMlDsa65(): VerifyOutcome {
  return {
    result: SKIPPED,
    note: "ML-DSA-65 not available in node:crypto; signature axis skipped",
  };
}

/** Ed25519 verify over a raw 32-byte public key (RFC 8032). */
function verifyEd25519(pk: Uint8Array, msg: Uint8Array, sig: Uint8Array): VerifyOutcome {
  let key;
  try {
    if (pk.length !== 32) {
      return { result: FAIL, note: `bad Ed25519 public key: expected 32 bytes, got ${pk.length}` };
    }
    const spki = Buffer.concat([SPKI_ED25519_PREFIX, Buffer.from(pk)]);
    key = createPublicKey({ key: spki, format: "der", type: "spki" });
  } catch (exc) {
    return { result: FAIL, note: `bad Ed25519 public key: ${(exc as Error).message}` };
  }
  try {
    const ok = verify(null, Buffer.from(msg), key, Buffer.from(sig));
    return ok
      ? { result: PASS, note: "signature valid" }
      : { result: FAIL, note: "signature mismatch" };
  } catch (exc) {
    return { result: FAIL, note: `verify error: ${(exc as Error).message}` };
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
  let key;
  try {
    if (pk.length !== 65 || pk[0] !== 0x04) {
      return { result: FAIL, note: `bad P-256 public key: expected 65-byte uncompressed point, got ${pk.length}` };
    }
    const x = base64url(Buffer.from(pk.slice(1, 33)));
    const y = base64url(Buffer.from(pk.slice(33, 65)));
    key = createPublicKey({ key: { kty: "EC", crv: "P-256", x, y }, format: "jwk" });
  } catch (exc) {
    return { result: FAIL, note: `bad P-256 public key: ${(exc as Error).message}` };
  }
  if (sig.length !== 64) {
    return { result: FAIL, note: `ES256 signature must be 64-byte raw r||s, got ${sig.length}` };
  }
  try {
    const ok = verify("sha256", Buffer.from(msg), { key, dsaEncoding: "ieee-p1363" }, Buffer.from(sig));
    return ok
      ? { result: PASS, note: "signature valid" }
      : { result: FAIL, note: "signature mismatch" };
  } catch (exc) {
    return { result: FAIL, note: `verify error: ${(exc as Error).message}` };
  }
}

type Verifier = (pk: Uint8Array, msg: Uint8Array, sig: Uint8Array) => VerifyOutcome;

const DISPATCH: Record<string, Verifier> = {
  "ML-DSA-65": verifyMlDsa65,
  ED25519: verifyEd25519,
  EDDSA: verifyEd25519,
  ES256: verifyEs256,
};

/** Dispatch to the algorithm's verifier; SKIP an unsupported alg. */
export function verifySignature(
  alg: string,
  pk: Uint8Array,
  msg: Uint8Array,
  sig: Uint8Array,
): VerifyOutcome {
  const fn = DISPATCH[(alg || "").toUpperCase()];
  if (fn === undefined) {
    const known = Object.keys(DISPATCH).sort().join(", ");
    return { result: SKIPPED, note: `unsupported alg '${alg}' (oracle checks ${known})` };
  }
  return fn(pk, msg, sig);
}

import { createHash } from "node:crypto";

/** Lowercase hex SHA-256 - the chain primitive every format shares. */
export function sha256Hex(data: Uint8Array): string {
  return createHash("sha256").update(Buffer.from(data)).digest("hex");
}
