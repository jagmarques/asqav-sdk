/**
 * agent-receipts adapter - W3C-VC AgentReceipt, Ed25519 over strict RFC 8785 JCS.
 * A port of the Python oracle's `verifier/oracle/adapters/agentreceipts.py`.
 *
 * An AgentReceipt is a W3C Verifiable Credential 2.0 envelope. The signature is
 * Ed25519 over the strict JCS bytes of the receipt with the `proof` member
 * removed, signed DIRECTLY. `proof.type` is `Ed25519Signature2020` and
 * `proof.proofValue` is multibase `u` (base64url, no padding).
 *
 * Uses `jcsRfc8785` (UTF-16 key sort, ECMAScript numbers), NOT the AERF/ACTA
 * `jcs` dialect. The signing key resolves from `proof.verificationMethod` through
 * the shared DID resolver.
 *
 * Chain rule: `credentialSubject.chain.previous_receipt_hash` is `"sha256:"` +
 * hex(SHA-256(JCS(predecessor without proof))). Genesis carries the field present
 * and explicitly `null`.
 */

import {
  FormatAdapter,
  type AxisCheck,
  type ChainStep,
  type KeyProvider,
  type SignatureMaterial,
} from "../adapter.js";
import { jcsRfc8785 } from "../canonical.js";
import { sha256Hex } from "../crypto.js";
import { resolveEd25519Key } from "../did.js";

/** SHA-256 hash prefix the chain field carries before the hex digest. */
const SHA256_PREFIX = "sha256:";

function signable(doc: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(doc)) {
    if (k !== "proof") out[k] = v;
  }
  return out;
}

/** Decode a multibase `u` (base64url, no padding) proofValue to raw bytes. */
function multibaseUDecode(value: string): Uint8Array {
  if (!value || value[0] !== "u") {
    throw new Error("proofValue is not multibase 'u' (base64url) encoded");
  }
  const body = value.slice(1).replace(/-/g, "+").replace(/_/g, "/");
  const padded = body + "=".repeat((-body.length % 4 + 4) % 4);
  return new Uint8Array(Buffer.from(padded, "base64"));
}

function asRecord(v: unknown): Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

function chainOf(doc: Record<string, unknown>): Record<string, unknown> {
  const sub = doc.credentialSubject;
  if (sub !== null && typeof sub === "object" && !Array.isArray(sub)) {
    const chain = (sub as Record<string, unknown>).chain;
    return asRecord(chain);
  }
  return {};
}

export class AgentReceiptsAdapter extends FormatAdapter {
  readonly name = "agentreceipts";

  detect(doc: Record<string, unknown>): boolean {
    const types = doc.type;
    const proof = doc.proof;
    if (!Array.isArray(types) || !types.includes("AgentReceipt")) return false;
    return (
      proof !== null &&
      typeof proof === "object" &&
      !Array.isArray(proof) &&
      typeof (proof as Record<string, unknown>).proofValue === "string"
    );
  }

  extractSignature(doc: Record<string, unknown>): SignatureMaterial {
    const proof = asRecord(doc.proof);
    let sig: Uint8Array;
    try {
      sig = multibaseUDecode((proof.proofValue as string) ?? "");
    } catch {
      sig = new Uint8Array(0);
    }
    return { sig, alg: "Ed25519", kid: (proof.verificationMethod as string) ?? "" };
  }

  resolveKey(
    doc: Record<string, unknown>,
    keyProvider: KeyProvider,
  ): readonly [Uint8Array | null, string] {
    const kid = (asRecord(doc.proof).verificationMethod as string) ?? "";
    return resolveEd25519Key(kid, keyProvider);
  }

  signingInput(doc: Record<string, unknown>): Uint8Array {
    // Strict-JCS bytes of the receipt minus proof, signed directly (no pre-hash).
    return jcsRfc8785(signable(doc));
  }

  chainStep(doc: Record<string, unknown>): ChainStep {
    const chain = chainOf(doc);
    const present = "previous_receipt_hash" in chain;
    const prev = chain.previous_receipt_hash;
    const isGenesis = present && (prev === null || prev === undefined);
    return {
      prevField: (prev as string | null) ?? null,
      isGenesis,
      recompute: (pred) => SHA256_PREFIX + sha256Hex(jcsRfc8785(signable(pred))),
    };
  }

  schema(doc: Record<string, unknown>): AxisCheck {
    const required = [
      "@context",
      "id",
      "type",
      "version",
      "issuer",
      "issuanceDate",
      "credentialSubject",
      "proof",
    ];
    const missing = required.filter((f) => !(f in doc));
    if (missing.length > 0) {
      return ["FAIL", `missing required VC fields: ${missing.join(",")}`];
    }
    const types = doc.type;
    if (!Array.isArray(types) || !types.includes("VerifiableCredential") || !types.includes("AgentReceipt")) {
      return ["FAIL", "type must include VerifiableCredential and AgentReceipt"];
    }
    if (asRecord(doc.proof).type !== "Ed25519Signature2020") {
      return ["FAIL", "proof.type must be Ed25519Signature2020"];
    }
    const issuer = doc.issuer;
    const issuerDid =
      issuer !== null && typeof issuer === "object" && !Array.isArray(issuer)
        ? (issuer as Record<string, unknown>).id
        : issuer;
    const vmDid = ((asRecord(doc.proof).verificationMethod as string) ?? "").split("#")[0];
    if (!vmDid || vmDid !== issuerDid) {
      return ["FAIL", "proof.verificationMethod is not controlled by issuer (signing-key DID != issuer DID)"];
    }
    const chain = chainOf(doc);
    if (!("previous_receipt_hash" in chain)) {
      return ["FAIL", "chain.previous_receipt_hash must be present (null on genesis), never omitted"];
    }
    return ["PASS", "required VC fields present; type AgentReceipt; chain link well-formed"];
  }
}
