/**
 * Authproof adapter - ES256 over insertion-order JSON, key embedded as a JWK.
 * A port of the Python oracle's `verifier/oracle/adapters/authproof.py`.
 *
 * Authproof delegation receipts are signed by the reference JS SDK
 * (`Commonguy25/authproof-sdk`, `src/authproof.js`), authoritative for the wire
 * shape. A receipt is a flat object whose signature is ECDSA P-256 / SHA-256 over
 * `JSON.stringify(receipt-without-signature)` in INSERTION order (not JCS, no
 * pre-hash). The signature is hex-encoded raw r||s (64 bytes); the signer's public
 * key is embedded as a P-256 JWK at `signerPublicKey`.
 *
 * In TypeScript the signing input is produced with native `JSON.stringify`, which
 * is exactly what the JS SDK signs - byte-for-byte, no re-derivation.
 */

import {
  FormatAdapter,
  type AxisCheck,
  type ChainStep,
  type KeyProvider,
  type SignatureMaterial,
} from "../adapter.js";

/** A shipping-SDK delegationId is `auth-<epoch_ms>-<5 base36 chars>`. */
const DELEGATION_ID = /^auth-\d+-[a-z0-9]{5}$/;

/** Fields the SDK always writes, used for the structural check. */
const REQUIRED = [
  "delegationId",
  "issuedAt",
  "scope",
  "boundaries",
  "timeWindow",
  "operatorInstructions",
  "instructionsHash",
  "signerPublicKey",
  "signature",
] as const;

function safeHex(value: unknown): Uint8Array {
  if (typeof value !== "string") return new Uint8Array(0);
  if (value.length % 2 !== 0 || !/^[0-9a-fA-F]*$/.test(value)) return new Uint8Array(0);
  const out = new Uint8Array(value.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(value.slice(i * 2, i * 2 + 2), 16);
  return out;
}

function b64url(value: unknown): Uint8Array {
  if (typeof value !== "string") return new Uint8Array(0);
  try {
    const s = value.replace(/-/g, "+").replace(/_/g, "/");
    const padded = s + "=".repeat((-s.length % 4 + 4) % 4);
    return new Uint8Array(Buffer.from(padded, "base64"));
  } catch {
    return new Uint8Array(0);
  }
}

function isP256Jwk(jwk: unknown): jwk is Record<string, unknown> {
  return (
    jwk !== null &&
    typeof jwk === "object" &&
    !Array.isArray(jwk) &&
    (jwk as Record<string, unknown>).kty === "EC" &&
    (jwk as Record<string, unknown>).crv === "P-256"
  );
}

export class AuthproofAdapter extends FormatAdapter {
  readonly name = "authproof";

  detect(doc: Record<string, unknown>): boolean {
    const did = doc.delegationId;
    return (
      typeof did === "string" &&
      DELEGATION_ID.test(did) &&
      "operatorInstructions" in doc &&
      "instructionsHash" in doc &&
      isP256Jwk(doc.signerPublicKey) &&
      typeof doc.signature === "string"
    );
  }

  extractSignature(doc: Record<string, unknown>): SignatureMaterial {
    return { sig: safeHex(doc.signature), alg: "ES256", kid: "" };
  }

  resolveKey(doc: Record<string, unknown>): readonly [Uint8Array | null, string] {
    const jwk = doc.signerPublicKey;
    if (!isP256Jwk(jwk)) {
      return [null, "signerPublicKey is not a P-256 JWK"];
    }
    const x = b64url(jwk.x);
    const y = b64url(jwk.y);
    if (x.length !== 32 || y.length !== 32) {
      return [null, "P-256 JWK coordinates are not 32 bytes each"];
    }
    const point = new Uint8Array(65);
    point[0] = 0x04;
    point.set(x, 1);
    point.set(y, 33);
    return [point, "resolved embedded P-256 JWK"];
  }

  signingInput(doc: Record<string, unknown>): Uint8Array {
    // The SDK signs JSON.stringify of the receipt minus `signature`, key order intact.
    const body: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(doc)) {
      if (k !== "signature") body[k] = v;
    }
    return new TextEncoder().encode(JSON.stringify(body));
  }

  chainStep(): ChainStep {
    // The base Authproof receipt carries no in-band chain link of its own.
    return { prevField: null, isGenesis: true, recompute: () => "" };
  }

  schema(doc: Record<string, unknown>): AxisCheck {
    const missing = REQUIRED.filter((f) => doc[f] === undefined || doc[f] === null);
    if (missing.length > 0) {
      return ["FAIL", `authproof receipt missing fields: ${missing.join(",")}`];
    }
    return ["PASS", "authproof delegation receipt; required fields present"];
  }
}
