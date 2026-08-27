/**
 * W3C VC 2.0 adapter - DataIntegrityProof with the eddsa-jcs-2022 cryptosuite.
 * A port of the Python oracle's `verifier/oracle/adapters/w3c_vc.py`.
 *
 * A Verifiable Credential 2.0 envelope secured by a `DataIntegrityProof` whose
 * `cryptosuite` is `eddsa-jcs-2022` (W3C TR vc-di-eddsa): Ed25519 (RFC 8032)
 * signs `SHA-256(JCS(proofOptions)) || SHA-256(JCS(unsecuredDocument))` where
 * JCS is strict RFC 8785, proofOptions is `proof` with `proofValue` removed,
 * and unsecuredDocument is the credential with `proof` removed. `proofValue` is
 * the raw 64-byte signature multibase base58btc ('z') encoded.
 *
 * When the proof options carry their own `@context` the spec's transform
 * requires the document `@context` to start with it in order and canonicalises
 * the document with the proof's `@context` substituted; both rules are enforced.
 *
 * Structure is intentionally stricter than the suite spec: proofPurpose must be
 * `assertionMethod` and the verificationMethod DID must equal the issuer DID,
 * fail-closed, mirroring the agentreceipts adapter. A W3C VC carries no in-band
 * chain link, so the chain axis reports genesis.
 */

import { createHash } from "node:crypto";

import {
  FormatAdapter,
  type AxisCheck,
  type ChainStep,
  type ExtraAxis,
  type KeyProvider,
  type SignatureMaterial,
} from "../adapter.js";
import { jcsRfc8785 } from "../canonical.js";
import { b58btcDecode, resolveEd25519Key } from "../did.js";

/** First @context value a VC 2.0 credential must carry. */
const VC_V2_CONTEXT = "https://www.w3.org/ns/credentials/v2";
const PROOF_TYPE = "DataIntegrityProof";
const CRYPTOSUITE = "eddsa-jcs-2022";

function asRecord(v: unknown): Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

/** The doc's proof member when it is a single object; {} for absent/list proofs. */
function proofOf(doc: Record<string, unknown>): Record<string, unknown> {
  return asRecord(doc.proof);
}

/** The issuer DID whether issuer is a bare string or an object with an id. */
function issuerDid(doc: Record<string, unknown>): string | null {
  const issuer = doc.issuer;
  if (typeof issuer === "string") return issuer;
  if (issuer !== null && typeof issuer === "object" && !Array.isArray(issuer)) {
    const id = (issuer as Record<string, unknown>).id;
    if (typeof id === "string") return id;
  }
  return null;
}

/** xsd:datetime (RFC 3339 profile) -> epoch ms, or null when unparseable. */
function parseDatetime(value: unknown): number | null {
  if (typeof value !== "string") return null;
  const text = value.trim();
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|z|[+-]\d{2}:\d{2})$/.test(text)) return null;
  const ms = Date.parse(text);
  return Number.isNaN(ms) ? null : ms;
}

/** proof minus proofValue - the exact object the spec canonicalises for hash one. */
function proofOptions(doc: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(proofOf(doc))) {
    if (k !== "proofValue") out[k] = v;
  }
  return out;
}

/** Document minus proof; the proof's @context substitutes the document's when present. */
function unsecured(doc: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(doc)) {
    if (k !== "proof") out[k] = v;
  }
  const ctx = proofOf(doc)["@context"];
  if (ctx !== undefined) out["@context"] = ctx;
  return out;
}

/** True when the proof's @context is absent or prefixes the document @context in order. */
function contextPrefixOk(doc: Record<string, unknown>): boolean {
  const ctx = proofOf(doc)["@context"];
  if (ctx === undefined) return true;
  if (!Array.isArray(ctx)) return false;
  const docCtx = doc["@context"];
  if (!Array.isArray(docCtx)) return false;
  return ctx.every((v, i) => docCtx[i] !== undefined && JSON.stringify(docCtx[i]) === JSON.stringify(v));
}

/** Decode a multibase 'z' (base58btc) proofValue to raw signature bytes. */
function multibaseZDecode(value: string): Uint8Array {
  if (!value || value[0] !== "z") {
    throw new Error("proofValue is not multibase 'z' (base58btc) encoded");
  }
  return b58btcDecode(value.slice(1));
}

function sha256(data: Uint8Array): Uint8Array {
  return new Uint8Array(createHash("sha256").update(Buffer.from(data)).digest());
}

/** W3C VC 2.0 - DataIntegrityProof eddsa-jcs-2022, Ed25519 over RFC 8785 JCS. */
export class W3cVcAdapter extends FormatAdapter {
  readonly name = "w3c-vc";

  detect(doc: Record<string, unknown>): boolean {
    const types = doc.type;
    if (!Array.isArray(types) || !types.includes("VerifiableCredential")) return false;
    const proof = doc.proof;
    const candidates = Array.isArray(proof)
      ? proof
      : proof !== null && typeof proof === "object"
        ? [proof]
        : [];
    // Any DataIntegrityProof on a VC routes here; the cryptosuite check belongs
    // to the schema axis so a sibling suite reports as an algorithm mismatch
    return candidates.some(
      (p) => p !== null && typeof p === "object" && (p as Record<string, unknown>).type === PROOF_TYPE,
    );
  }

  extractSignature(doc: Record<string, unknown>): SignatureMaterial {
    const proof = proofOf(doc);
    const value = typeof proof.proofValue === "string" ? proof.proofValue : "";
    let sig: Uint8Array;
    try {
      sig = multibaseZDecode(value);
    } catch {
      sig = new Uint8Array(0);
    }
    const vm = typeof proof.verificationMethod === "string" ? proof.verificationMethod : "";
    return { sig, alg: "EdDSA", kid: vm };
  }

  resolveKey(
    doc: Record<string, unknown>,
    keyProvider: KeyProvider,
  ): readonly [Uint8Array | null, string] {
    const vm = proofOf(doc).verificationMethod;
    return resolveEd25519Key(typeof vm === "string" ? vm : "", keyProvider);
  }

  signingInput(doc: Record<string, unknown>): Uint8Array {
    // hashData = SHA-256(JCS(proofOptions)) || SHA-256(JCS(unsecured)) (TR vc-di-eddsa)
    const optionsHash = sha256(jcsRfc8785(proofOptions(doc)));
    const documentHash = sha256(jcsRfc8785(unsecured(doc)));
    return Uint8Array.from([...optionsHash, ...documentHash]);
  }

  chainStep(_doc: Record<string, unknown>): ChainStep {
    // A W3C VC carries no in-band chain link of its own
    return { prevField: null, isGenesis: true, recompute: () => "" };
  }

  schema(doc: Record<string, unknown>): AxisCheck {
    const ctx = doc["@context"];
    if (!Array.isArray(ctx) || ctx.length === 0 || ctx[0] !== VC_V2_CONTEXT) {
      return ["FAIL", "first @context must be the W3C VC 2.0 credentials context"];
    }
    const types = doc.type;
    if (!Array.isArray(types) || !types.includes("VerifiableCredential")) {
      return ["FAIL", "type must include VerifiableCredential"];
    }
    const proof = doc.proof;
    if (Array.isArray(proof)) {
      return ["FAIL", "proof sets are not supported; exactly one proof object is required"];
    }
    if (proof === null || typeof proof !== "object") {
      return ["FAIL", "missing required VC fields: proof"];
    }
    const p = proof as Record<string, unknown>;
    if (p.type !== PROOF_TYPE) {
      return ["FAIL", "proof.type must be DataIntegrityProof"];
    }
    if (p.cryptosuite !== CRYPTOSUITE) {
      return [
        "FAIL",
        `unsupported signature algorithm: cryptosuite '${p.cryptosuite}' ` +
          `(this verifier checks '${CRYPTOSUITE}')`,
      ];
    }
    const vm = p.verificationMethod;
    if (typeof vm !== "string" || !vm.startsWith("did:")) {
      return ["FAIL", "proof.verificationMethod must be a DID URL"];
    }
    if (p.proofPurpose !== "assertionMethod") {
      return ["FAIL", "proof.proofPurpose must be assertionMethod"];
    }
    const issuer = issuerDid(doc);
    if (issuer === null || !issuer.startsWith("did:")) {
      return ["FAIL", "issuer must be a DID"];
    }
    if (vm.split("#")[0] !== issuer) {
      // bind the signing key to the issuer or anyone self-signs as a victim
      return [
        "FAIL",
        "proof.verificationMethod is not controlled by issuer (signing-key DID != issuer DID)",
      ];
    }
    const subject = doc.credentialSubject;
    if (!(subject !== null && typeof subject === "object" && (!Array.isArray(subject) || subject.length > 0))) {
      return ["FAIL", "missing required VC fields: credentialSubject"];
    }
    if (p.created !== undefined && parseDatetime(p.created) === null) {
      return ["FAIL", `unreadable proof.created: '${p.created}'`];
    }
    if (!contextPrefixOk(doc)) {
      return ["FAIL", "proof @context is not a prefix of the document @context"];
    }
    return ["PASS", "required VC 2.0 fields present; DataIntegrityProof eddsa-jcs-2022"];
  }

  /** validFrom/validUntil bound validity; the expiry axis never folds the verdict (426). */
  extraAxes(doc: Record<string, unknown>, _keyProvider: KeyProvider): ExtraAxis[] {
    const now = Date.now();
    const validFrom = doc.validFrom;
    if (validFrom !== undefined) {
      const parsed = parseDatetime(validFrom);
      if (parsed === null) {
        return [["expiry", "FAIL", `unreadable expires_at (validFrom '${validFrom}')`]];
      }
      if (now < parsed) {
        return [["expiry", "FAIL", `not yet valid: validFrom ${validFrom}`]];
      }
    }
    const validUntil = doc.validUntil;
    if (validUntil !== undefined) {
      const parsed = parseDatetime(validUntil);
      if (parsed === null) {
        return [["expiry", "FAIL", `unreadable expires_at (validUntil '${validUntil}')`]];
      }
      if (now >= parsed) {
        return [["expiry", "FAIL", `expired at ${validUntil}`]];
      }
    }
    return [["expiry", "PASS", "no validFrom/validUntil constraint breached"]];
  }
}
