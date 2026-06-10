/**
 * ACTA adapter - Ed25519 over the JCS of the payload, signed directly.
 * A port of the Python oracle's `verifier/oracle/adapters/acta.py`.
 *
 * Targets draft-farley-acta-signed-receipts-01. The envelope is
 * `{payload, signature:{alg, kid, sig}}`; the signature is Ed25519 over the JCS
 * canonical bytes of the `payload` object, signed DIRECTLY (no pre-hash), with the
 * sig as a lowercase hex string. The OPTIONAL Commitment Mode is NOT implemented:
 * a commitment-mode receipt fails the baseline signature check (the honest outcome).
 *
 * Asqav-native is a PROFILE of ACTA, so the payloads look identical; the formats
 * are kept disjoint by the signature encoding (ACTA hex, Asqav-native base64) plus
 * the Asqav-only `anchors` envelope key.
 *
 * The chain field `previousReceiptHash` is SHA-256 of the JCS of the FULL
 * predecessor receipt INCLUDING its signature.
 */

import {
  FormatAdapter,
  type AxisCheck,
  type ChainStep,
  type KeyProvider,
  type SignatureMaterial,
} from "../adapter.js";
import { jcs } from "../canonical.js";
import { sha256Hex } from "../crypto.js";
import { b64decode } from "../vrShim.js";

/** True for a non-empty even-length lowercase-hex string (an ACTA signature). */
export function isLowerHex(s: unknown): boolean {
  return (
    typeof s === "string" &&
    s.length > 0 &&
    s.length % 2 === 0 &&
    /^[0-9a-f]+$/.test(s)
  );
}

function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

function asRecord(v: unknown): Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

export class ActaAdapter extends FormatAdapter {
  readonly name = "acta";

  detect(doc: Record<string, unknown>): boolean {
    const payload = doc.payload;
    const sig = doc.signature;
    const payloadIsObj = payload !== null && typeof payload === "object" && !Array.isArray(payload);
    const sigIsObj = sig !== null && typeof sig === "object" && !Array.isArray(sig);
    if (!payloadIsObj || !sigIsObj) return false;
    if ("anchors" in doc) return false; // Asqav-native envelope marker; never ACTA.
    return isLowerHex((sig as Record<string, unknown>).sig);
  }

  extractSignature(doc: Record<string, unknown>): SignatureMaterial {
    const sig = asRecord(doc.signature);
    return {
      sig: hexToBytes((sig.sig as string) ?? ""),
      alg: (sig.alg as string) ?? "EdDSA",
      kid: (sig.kid as string) ?? "",
    };
  }

  resolveKey(
    doc: Record<string, unknown>,
    keyProvider: KeyProvider,
  ): readonly [Uint8Array | null, string] {
    const kid = (asRecord(doc.signature).kid as string) ?? "";
    const jwkKeys = ((keyProvider?.keys as Array<Record<string, unknown>>) ?? []);
    for (const jwk of jwkKeys) {
      if (jwk.kid !== kid) continue;
      if (jwk.kty !== "OKP" || jwk.crv !== "Ed25519") {
        return [null, `kid '${kid}' is not an OKP/Ed25519 JWK`];
      }
      const raw = b64decode((jwk.x as string) ?? "");
      if (raw.length !== 32) {
        return [null, `kid '${kid}' Ed25519 key is ${raw.length} bytes, expected 32`];
      }
      return [raw, `resolved kid ${kid} from ACTA JWK Set`];
    }
    return [null, `kid '${kid}' not in ACTA JWK Set`];
  }

  signingInput(doc: Record<string, unknown>): Uint8Array {
    // Baseline: Ed25519 signs the JCS bytes of the payload directly, no pre-hash.
    return jcs(asRecord(doc.payload));
  }

  chainStep(doc: Record<string, unknown>): ChainStep {
    const payload = asRecord(doc.payload);
    const prev = payload.previousReceiptHash;
    const isGenesis = !("previousReceiptHash" in payload);
    // Chain hash covers the full predecessor receipt, signature included.
    return {
      prevField: (prev as string | null) ?? null,
      isGenesis,
      recompute: (pred) => sha256Hex(jcs(pred)),
    };
  }

  schema(doc: Record<string, unknown>): AxisCheck {
    const payload = doc.payload;
    const sig = doc.signature;
    const payloadIsObj = payload !== null && typeof payload === "object" && !Array.isArray(payload);
    const sigIsObj = sig !== null && typeof sig === "object" && !Array.isArray(sig);
    if (!payloadIsObj || !sigIsObj) {
      return ["FAIL", "ACTA receipt needs object payload and signature"];
    }
    const p = payload as Record<string, unknown>;
    const s = sig as Record<string, unknown>;
    const missing: string[] = [];
    // Require only the temporal anchor + signature triple; `type`/`issuer_id` are
    // OPTIONAL Asqav-profile fields.
    for (const f of ["issued_at"]) {
      if (!(f in p)) missing.push(f);
    }
    for (const f of ["alg", "kid", "sig"]) {
      if (!(f in s)) missing.push(`signature.${f}`);
    }
    if (missing.length > 0) {
      return ["FAIL", `missing required fields: ${missing.join(",")}`];
    }
    if ("previousReceiptHash" in p && p.previousReceiptHash === null) {
      return ["FAIL", "genesis must omit previousReceiptHash, not set it null"];
    }
    // False-attestation guard: when the Asqav-profile `issuer_id` is present it
    // MUST match the signing kid. Absent it, there is nothing to contradict.
    if ("issuer_id" in p && p.issuer_id !== s.kid) {
      return ["FAIL", "issuer_id must match signature.kid"];
    }
    if (s.alg !== "EdDSA" && s.alg !== "Ed25519") {
      return ["FAIL", `unsupported ACTA alg ${JSON.stringify(s.alg)}; only EdDSA baseline is implemented`];
    }
    return ["PASS", "issued_at + signature triple present; issuer_id (when present) matches kid; EdDSA baseline"];
  }
}
