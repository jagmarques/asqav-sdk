/**
 * Asqav-native adapter, a port of the Python oracle's `asqav_native.py`. Two wire shapes route here,
 * mutually exclusive at detection: the compliance envelope and the flat hash-mode receipt.
 */

import {
  FormatAdapter,
  type AxisCheck,
  type ChainStep,
  type ExtraAxis,
  type KeyProvider,
  type SignatureMaterial,
} from "../adapter.js";
import {
  JCS_UTF16_CUTOVER,
  asqavJcs,
  asqavJcsPreCutover,
  hasSupplementaryMemberName,
} from "../canonical.js";
import { sha256Hex } from "../crypto.js";
import { isLowerHex } from "./acta.js";
import {
  FIRST_RECEIPT_SEED,
  b64decode,
  checkExpiry,
  checkIssuerBinding,
  checkCounterpartyBinding,
  checkKeyBinding,
  checkKeyStatus,
  checkPayloadDigest,
  checkSkew,
  checkNonce,
  checkOrgBinding,
  checkStructure,
  keyIssuerOf,
  keyOrgOf,
  matchSigningKey,
  normaliseEnvelope,
  revokedAtOf,
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
// Claims belonging to the signed payload; hash mode signs the flat fields only,
// so a thumbprint or seq pasted onto one binds nothing.
const UNSIGNED_CLAIM_FIELDS = ["issuer_id", "previousReceiptHash", "key_thumbprint", "seq"];

/**
 * alg plus raw public-key bytes of the resolved directory entry. The directory publishes standard
 * base64 while an AKP thumbprint is over unpadded base64url, so the decode precedes the digest.
 */
function resolvedKeyMaterial(
  entry: Record<string, unknown> | null,
): readonly [unknown, Uint8Array | null] {
  if (entry === null) return [null, null];
  try {
    return [entry.alg, b64decode(entry.public_key as string)];
  } catch {
    return [entry.alg, null];
  }
}

function isHashMode(doc: Record<string, unknown>): boolean {
  // Routing reads the shape of the signed unit only: a record payload is the
  // compliance signed unit, any other shape routes here (mirrors _is_hash_mode).
  if (doc.mode !== "hash" || isRecord(doc.payload)) return false;
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

  /** Shared per instance, so a duplicate (issuer_id, nonce) pair is flagged (draft 5.7). */
  private readonly seenNonces = new Set<string>();

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

  /**
   * The one JWKS entry this receipt's signature is checked against. Every axis resolves through here: kid
   * sits outside the signed bytes, so a second independent lookup would be attacker-steerable.
   */
  private signingKeyEntry(
    doc: Record<string, unknown>,
    jwks: Record<string, unknown>,
  ): Record<string, unknown> | null {
    const payload = payloadOf(doc);
    return matchSigningKey(
      jwks,
      this.extractSignature(doc).kid,
      payload.agent_id ?? doc.agent_id,
      payload.issuer_id,
      payload.org_id ?? doc.org_id,
    );
  }

  resolveKey(
    doc: Record<string, unknown>,
    keyProvider: KeyProvider,
  ): readonly [Uint8Array | null, string] {
    const jwks = (keyProvider ?? { keys: [] }) as Record<string, unknown>;
    const kid = this.extractSignature(doc).kid;
    const entry = this.signingKeyEntry(doc, jwks);
    if (entry === null) {
      return [null, `kid '${kid}' not in jwks directory`];
    }
    const pk = b64decode(entry.public_key as string);
    const status = (entry.status as string) ?? null;
    if (kid && (kid === entry.issuer_id || kid === entry.kid)) {
      return [pk, `resolved kid ${kid} (status=${status})`];
    }
    return [pk, `resolved agent key ${entry.kid} (status=${status})`];
  }

  signingInput(doc: Record<string, unknown>): Uint8Array {
    if (isHashMode(doc)) {
      return this.hashModeSigningInput(doc);
    }
    // Asqav signs the canonical bytes of the payload directly, no pre-hash.
    return asqavJcs(payloadOf(doc));
  }

  // Only a payload-mode receipt issued before the cutover with a member name above U+FFFF has
  // a dated dialect; hash-mode members are ASCII, so both orders coincide there.
  preCutoverSigningInput(doc: Record<string, unknown>): Uint8Array | null {
    if (isHashMode(doc)) return null;
    const payload = payloadOf(doc);
    if (!hasSupplementaryMemberName(payload)) return null;
    const issued = Date.parse(String(payload.issued_at ?? ""));
    if (Number.isNaN(issued) || issued >= Date.parse(JCS_UTF16_CUTOVER)) return null;
    return asqavJcsPreCutover(payload);
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
      // A claim outside the signed field set is unauthenticated whatever it says.
      const unsigned = UNSIGNED_CLAIM_FIELDS.filter((f) => f in doc);
      if (unsigned.length > 0) {
        return [
          "FAIL",
          `hash-mode receipt carries claim fields its signature does not cover: ${unsigned.join(",")}`,
        ];
      }
      return ["PASS", "hash-mode signature receipt; required flat fields present"];
    }
    return checkStructure(payloadOf(doc));
  }

  // Gate on expiry, signing key revocation status, and its issuer (mirrors Python extra_axes).
  // The key axes are a no-op when the key is absent; the signature axis handles that.
  extraAxes(doc: Record<string, unknown>, keyProvider: KeyProvider): ExtraAxis[] {
    const hashMode = isHashMode(doc);
    // Expiry reads only the signed bytes, so no key is needed. Hash mode signs no
    // expires_at, and reading the flat doc would gate on an uncovered field.
    const axes: ExtraAxis[] = [["expiry", ...checkExpiry(hashMode ? {} : payloadOf(doc))]];
    axes.push(["nonce", ...checkNonce(hashMode ? {} : payloadOf(doc), this.seenNonces)]);
    const jwks = (keyProvider ?? { keys: [] }) as Record<string, unknown>;
    const entry = this.signingKeyEntry(doc, jwks);
    // Reported before the no-entry return, so a receipt binding no thumbprint
    // still says so rather than dropping the axis when nothing resolved.
    const [boundAlg, boundPk] = resolvedKeyMaterial(entry);
    const signedUnit = hashMode ? {} : payloadOf(doc);
    axes.push(["key_binding", ...checkKeyBinding(signedUnit, boundAlg, boundPk)]);
    // No database offline, so a claimed binding reports unresolved rather than
    // riding along as corroboration nobody checked
    axes.push(["counterparty", ...checkCounterpartyBinding(signedUnit)]);
    axes.push(["payload_digest", ...checkPayloadDigest(signedUnit)]);
    // Hash mode signs no issued_at, so skew reads the flat server_timestamp there
    const stamp = hashMode ? doc.server_timestamp : signedUnit.issued_at;
    axes.push(["skew", ...checkSkew(stamp)]);
    if (entry === null) return axes;
    // Both wire shapes name their issuer inside the signed bytes: issuer_id in
    // compliance mode, org_id in hash mode.
    const issuedAt = hashMode
      ? String(doc.server_timestamp ?? "")
      : String(payloadOf(doc).issued_at ?? "");
    // Offline anchor presence is unverifiable (anchors unsigned); pass false
    // so a forged anchor never rides a revoked key to PASS.
    const status = (entry.status as string) ?? null;
    const [res, note] = checkKeyStatus(status, issuedAt, revokedAtOf(entry), false);
    // The key that verified is bound back to the issuer the signed bytes name.
    const keyIssuer = keyIssuerOf(entry);
    const [bindRes, bindNote] = hashMode
      ? checkOrgBinding(keyIssuer, keyOrgOf(entry), doc.org_id)
      : checkIssuerBinding(keyIssuer, payloadOf(doc).issuer_id);
    return [
      ...axes,
      ["key_status", res, note],
      ["issuer_bind", bindRes, bindNote],
    ];
  }

  // Hash mode signs the flat field set only, so a seq sitting there binds nothing.
  seqOf(doc: Record<string, unknown>): unknown {
    if (isHashMode(doc)) return null;
    return payloadOf(doc).seq ?? null;
  }

  // Surface the v:2 in-body signer. null for v:1 and hash-mode. Read only from
  // the signed payload, so a signer appended as loose metadata is never surfaced.
  attestation(doc: Record<string, unknown>): Record<string, unknown> {
    if (isHashMode(doc)) return {};
    const signer = payloadOf(doc).signer;
    return signer !== undefined && signer !== null ? { signer } : {};
  }

  /**
   * A hash-mode digest sealed with the org salt is keyed (criterion 438): internally consistent but not
   * third-party re-derivable, so a fully-checked receipt reports verified_keyed, never plain verified.
   */
  keyedDigest(doc: Record<string, unknown>): boolean {
    return isHashMode(doc) && doc.hash_algo === "hmac-sha256";
  }
}
