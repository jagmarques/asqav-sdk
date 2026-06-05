/**
 * AERF adapter - Ed25519 over RFC 8785 JCS, signed directly (no pre-hash).
 * A port of the Python oracle's `verifier/oracle/adapters/aerf.py`.
 *
 * AERF receipts are a flat object. The signature is Ed25519 over the JCS bytes of
 * the receipt with the non-payload fields stripped - `signature`, `timestamp`,
 * `parent_signature`, `parent_key_id`, `log_inclusion_proof` - signed DIRECTLY.
 *
 * Chain rule (AERF SPEC): `previous_receipt_hash` is the SHA-256 of the
 * predecessor's signable payload with the SAME field set stripped (signature
 * EXCLUDED). Genesis OMITS the field entirely; a present `previous_receipt_hash:
 * null` is a malformed genesis.
 *
 * AERF layers two conditionally-REQUIRED signature checks in `extraAxes`:
 *   - parent counter-signature: REQUIRED when `impact_tags` is non-empty.
 *   - PDP binding: REQUIRED when `impact_tags` is non-empty; the PDP key signs
 *     the canonical tuple `{context_hash_sha256, in_policy, policy_hash}`.
 */

import {
  FormatAdapter,
  type AxisCheck,
  type ChainStep,
  type ExtraAxis,
  type KeyProvider,
  type SignatureMaterial,
} from "../adapter.js";
import { jcs } from "../canonical.js";
import { FAIL, PASS, SKIPPED, sha256Hex, verifySignature } from "../crypto.js";

/** Fields removed before canonicalising for both the signature and the chain hash. */
const STRIP = new Set([
  "signature",
  "timestamp",
  "parent_signature",
  "parent_key_id",
  "log_inclusion_proof",
]);

/** Ed25519 SPKI header (12-byte prefix + 32 raw key bytes). */
const SPKI_PREFIX = Buffer.from("302a300506032b6570032100", "hex");

function safeHex(value: unknown): Uint8Array {
  if (typeof value !== "string") return new Uint8Array(0);
  if (value.length % 2 !== 0 || !/^[0-9a-fA-F]*$/.test(value)) return new Uint8Array(0);
  const out = new Uint8Array(value.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(value.slice(i * 2, i * 2 + 2), 16);
  return out;
}

function signable(doc: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(doc)) {
    if (!STRIP.has(k)) out[k] = v;
  }
  return out;
}

/** Return the raw 32-byte key from raw input or an RFC 8410 SPKI value. */
function rawEd25519(key: Uint8Array): Uint8Array {
  if (key.length === 44 && Buffer.from(key.slice(0, 12)).equals(SPKI_PREFIX)) {
    return key.slice(12);
  }
  return key;
}

function resolveRaw(keyProvider: KeyProvider, kid: string): Uint8Array | null {
  const material = keyProvider?.[kid];
  if (material === undefined || material === null) return null;
  return rawEd25519(material instanceof Uint8Array ? material : safeHex(material));
}

export class AerfAdapter extends FormatAdapter {
  readonly name = "aerf";

  detect(doc: Record<string, unknown>): boolean {
    return doc.type === "notarised_evidence" && "evidence_hash_sha512" in doc;
  }

  extractSignature(doc: Record<string, unknown>): SignatureMaterial {
    return {
      sig: safeHex(doc.signature),
      alg: "Ed25519",
      kid: (doc.key_id as string) ?? "",
    };
  }

  resolveKey(
    doc: Record<string, unknown>,
    keyProvider: KeyProvider,
  ): readonly [Uint8Array | null, string] {
    const keys = keyProvider ?? {};
    const kid = (doc.key_id as string) ?? "";
    const material = keys[kid];
    if (material === undefined || material === null) {
      return [null, `key_id '${kid}' not supplied to the key provider`];
    }
    const raw = rawEd25519(material instanceof Uint8Array ? material : safeHex(material));
    return [raw, `resolved key_id ${kid}`];
  }

  signingInput(doc: Record<string, unknown>): Uint8Array {
    // AERF signs the JCS canonical bytes of the stripped payload directly.
    return jcs(signable(doc));
  }

  chainStep(doc: Record<string, unknown>): ChainStep {
    const prev = doc.previous_receipt_hash;
    const isGenesis = !("previous_receipt_hash" in doc);
    return {
      prevField: (prev as string | null) ?? null,
      isGenesis,
      recompute: (pred) => sha256Hex(jcs(signable(pred))),
    };
  }

  schema(doc: Record<string, unknown>): AxisCheck {
    const required = [
      "id",
      "type",
      "plan_id",
      "agent",
      "action",
      "in_policy",
      "evidence_hash_sha512",
      "observed_at",
      "key_id",
      "signature",
    ];
    const missing = required.filter((f) => !(f in doc));
    if (missing.length > 0) {
      return ["FAIL", `missing required fields: ${missing.join(",")}`];
    }
    if ("previous_receipt_hash" in doc && doc.previous_receipt_hash === null) {
      return ["FAIL", "genesis must omit previous_receipt_hash, not set it null"];
    }
    return ["PASS", "required fields present; type notarised_evidence"];
  }

  extraAxes(doc: Record<string, unknown>, keyProvider: KeyProvider): ExtraAxis[] {
    const axes: ExtraAxis[] = [];
    const impactTags = doc.impact_tags;
    const hasImpact = Array.isArray(impactTags) ? impactTags.length > 0 : Boolean(impactTags);
    if (hasImpact || doc.parent_signature) {
      axes.push(this.parentAxis(doc, keyProvider, hasImpact));
    }
    if (hasImpact || doc.pdp_signature) {
      axes.push(this.pdpAxis(doc, keyProvider, hasImpact));
    }
    return axes;
  }

  private parentAxis(doc: Record<string, unknown>, keyProvider: KeyProvider, required: boolean): ExtraAxis {
    const sigHex = doc.parent_signature;
    if (!sigHex) {
      if (required) return ["parent_signature", FAIL, "parent_signature absent"];
      return ["parent_signature", SKIPPED, "no parent_signature on this receipt"];
    }
    const pk = resolveRaw(keyProvider, (doc.parent_key_id as string) ?? "");
    if (pk === null) {
      return ["parent_signature", SKIPPED, "parent_key_id not supplied to the key provider"];
    }
    const { result, note: why } = verifySignature("Ed25519", pk, this.signingInput(doc), safeHex(sigHex));
    const note = result === PASS ? "parent counter-signature valid" : `parent signature verification FAILED: ${why}`;
    return ["parent_signature", result, note];
  }

  private pdpAxis(doc: Record<string, unknown>, keyProvider: KeyProvider, required: boolean): ExtraAxis {
    const sigHex = doc.pdp_signature;
    if (!sigHex) {
      if (required) return ["pdp_signature", FAIL, "pdp_signature absent"];
      return ["pdp_signature", SKIPPED, "no pdp_signature on this receipt"];
    }
    const pk = resolveRaw(keyProvider, (doc.pdp_key_id as string) ?? "");
    if (pk === null) {
      return ["pdp_signature", SKIPPED, "pdp_key_id not supplied to the key provider"];
    }
    const tupleBytes = jcs({
      context_hash_sha256: doc.context_hash_sha256 ?? null,
      in_policy: doc.in_policy ?? null,
      policy_hash: doc.policy_hash ?? null,
    });
    const { result, note: why } = verifySignature("Ed25519", pk, tupleBytes, safeHex(sigHex));
    const note = result === PASS ? "pdp binding valid" : `pdp signature verification FAILED: ${why}`;
    return ["pdp_signature", result, note];
  }
}
