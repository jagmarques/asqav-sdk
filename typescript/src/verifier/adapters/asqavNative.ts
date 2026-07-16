/**
 * Asqav-native adapter - the existing verify_receipt logic behind the seam.
 * A port of the Python oracle's `verifier/oracle/adapters/asqav_native.py`.
 *
 * Two real Asqav wire shapes route here, kept mutually exclusive at detection:
 *
 *   - COMPLIANCE mode: the `{payload, signature, anchors}` envelope whose
 *     `payload` carries `previousReceiptHash` / `issuer_id`. The signature signs
 *     the canonical bytes of `payload` DIRECTLY; the chain hash is SHA-256 of the
 *     predecessor payload's canonical bytes.
 *   - HASH mode: the default `/sign` output - a FLAT receipt with `mode:"hash"`,
 *     `payload:null`, a `signature_b64` (or `signature`), and a `hash`. The
 *     signing input is the flat 11-field object the cloud rebuilds.
 *
 * Canonicalisation goes through `asqavJcs`, byte-identical to the cloud's
 * `verify_receipt.canonical_json`.
 */

import {
  FormatAdapter,
  type AxisCheck,
  type ChainStep,
  type ExtraAxis,
  type KeyProvider,
  type SignatureMaterial,
} from "../adapter.js";
import { asqavJcs } from "../canonical.js";
import { sha256Hex } from "../crypto.js";
import { isLowerHex } from "./acta.js";
import {
  FIRST_RECEIPT_SEED,
  b64decode,
  checkKeyStatus,
  checkStructure,
  normaliseEnvelope,
  resolveKey,
  resolveRevokedAt,
} from "../vrShim.js";

/** Field set the cloud's hash-mode signer canonicalises (for the structure check). */
const HASH_MODE_FIELDS = [
  "v",
  "mode",
  "hash",
  "hash_algo",
  "metadata",
  "server_timestamp",
  "action_id",
  "agent_id",
  "org_id",
  "policy_digest",
  "policy_decision",
] as const;

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

/** True for a flat hash-mode signature receipt (mode=hash, null payload, a sig). */
function isHashMode(doc: Record<string, unknown>): boolean {
  // Python: mode must be "hash" and payload must be None (null or absent).
  if (doc.mode !== "hash" || (doc.payload !== null && doc.payload !== undefined)) {
    return false;
  }
  return Boolean(doc.signature_b64 || doc.signature);
}

/** Normalise to the signed payload regardless of envelope nesting. */
function payloadOf(doc: Record<string, unknown>): Record<string, unknown> {
  const env = normaliseEnvelope(doc);
  const p = env.payload;
  return isRecord(p) ? p : env;
}

/** Decode signature material; empty on malformed input so verify FAILs, never crashes. */
function safeB64(value: unknown): Uint8Array {
  if (typeof value !== "string") return new Uint8Array(0);
  try {
    return b64decode(value);
  } catch {
    return new Uint8Array(0);
  }
}

export class AsqavNativeAdapter extends FormatAdapter {
  readonly name = "asqav-native";

  detect(doc: Record<string, unknown>): boolean {
    if (isHashMode(doc)) return true;
    const sig = doc.signature;
    // An ACTA receipt carries a lowercase-hex sig; decline it so the formats stay disjoint.
    if (isRecord(sig) && isLowerHex(sig.sig)) return false;
    const payload = doc.payload;
    if (isRecord(payload) && "previousReceiptHash" in payload) return true;
    // A bare payload (the conformance-vector shape) is also Asqav-native.
    return "previousReceiptHash" in doc && "issuer_id" in doc;
  }

  extractSignature(doc: Record<string, unknown>): SignatureMaterial {
    if (isHashMode(doc)) {
      return {
        sig: safeB64(doc.signature_b64 || doc.signature || ""),
        alg: (doc.algorithm as string) ?? "ML-DSA-65",
        kid: (doc.key_id as string) ?? "",
      };
    }
    const env = normaliseEnvelope(doc);
    let sigObj = env.signature;
    if (typeof sigObj === "string") {
      sigObj = { alg: "ML-DSA-65", kid: payloadOf(doc).issuer_id ?? "", sig: sigObj };
    }
    const so = isRecord(sigObj) ? sigObj : {};
    return {
      sig: safeB64(so.sig ?? ""),
      alg: (so.alg as string) ?? "ML-DSA-65",
      kid: (so.kid as string) ?? "",
    };
  }

  resolveKey(
    doc: Record<string, unknown>,
    keyProvider: KeyProvider,
  ): readonly [Uint8Array | null, string] {
    const jwks = keyProvider ?? { keys: [] };
    const kid = this.extractSignature(doc).kid;
    const [pk, status] = resolveKey(jwks, kid);
    if (pk === null) {
      return [null, `kid '${kid}' not in jwks directory`];
    }
    return [pk, `resolved kid ${kid} (status=${status})`];
  }

  signingInput(doc: Record<string, unknown>): Uint8Array {
    if (isHashMode(doc)) {
      return this.hashModeSigningInput(doc);
    }
    // Asqav signs the canonical bytes of the payload directly, no pre-hash.
    return asqavJcs(payloadOf(doc));
  }

  private hashModeSigningInput(doc: Record<string, unknown>): Uint8Array {
    const flat = {
      v: 1,
      mode: "hash",
      hash: doc.hash ?? null,
      hash_algo: doc.hash_algo ?? "sha256",
      metadata: doc.metadata ?? {},
      server_timestamp: doc.server_timestamp ?? null,
      action_id: doc.action_id ?? null,
      agent_id: doc.agent_id ?? null,
      org_id: doc.org_id ?? null,
      policy_digest: doc.policy_digest ?? null,
      policy_decision: doc.policy_decision ?? null,
    };
    return asqavJcs(flat);
  }

  chainStep(doc: Record<string, unknown>): ChainStep {
    if (isHashMode(doc)) {
      // A hash-mode signature receipt carries no in-band chain link of its own.
      return { prevField: null, isGenesis: true, recompute: () => "" };
    }
    const prev = payloadOf(doc).previousReceiptHash;
    const isGenesis = prev === FIRST_RECEIPT_SEED;
    return {
      prevField: (prev as string | null) ?? null,
      isGenesis,
      recompute: (pred) => sha256Hex(asqavJcs(payloadOf(pred))),
    };
  }

  schema(doc: Record<string, unknown>): AxisCheck {
    if (isHashMode(doc)) {
      const missing = HASH_MODE_FIELDS.filter(
        (f) => (doc[f] === undefined || doc[f] === null) && f !== "policy_digest",
      );
      if (missing.length > 0) {
        return ["FAIL", `hash-mode receipt missing fields: ${missing.join(",")}`];
      }
      return ["PASS", "hash-mode signature receipt; required flat fields present"];
    }
    return checkStructure(payloadOf(doc));
  }

  // Gate on signing key revocation status (mirrors Python extra_axes).
  // No-op when key is absent; the signature axis already handles that.
  extraAxes(doc: Record<string, unknown>, keyProvider: KeyProvider): ExtraAxis[] {
    const jwks = (keyProvider ?? { keys: [] }) as Record<string, unknown>;
    const kid = this.extractSignature(doc).kid;
    const [pk, status] = resolveKey(jwks, kid);
    if (pk === null) return [];
    const issuedAt = isHashMode(doc)
      ? String(doc.server_timestamp ?? "")
      : String(payloadOf(doc).issued_at ?? "");
    const revokedAt = resolveRevokedAt(jwks, kid);
    // Offline anchor presence is unverifiable (anchors unsigned); pass false
    // so a forged anchor never rides a revoked key to PASS.
    const [res, note] = checkKeyStatus(status, issuedAt, revokedAt, false);
    return [["key_status", res, note]];
  }

  // Surface the v:2 in-body signer. null for v:1 and hash-mode. Read only from
  // the signed payload, so a signer appended as loose metadata is never surfaced.
  attestation(doc: Record<string, unknown>): Record<string, unknown> {
    if (isHashMode(doc)) return {};
    const signer = payloadOf(doc).signer;
    return signer !== undefined && signer !== null ? { signer } : {};
  }
}
