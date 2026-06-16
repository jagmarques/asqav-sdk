/**
 * Pipelock EvidenceReceipt v2 adapter - Ed25519 over JCS with zeroed signature.
 * A port of the Python oracle's `verifier/oracle/adapters/pipelock.py`.
 *
 * Signing rule: clone the receipt dict, zero out the `signature` object (all
 * four fields set to empty strings), then JCS-canonicalize (RFC 8785 dialect)
 * the result. The signature is Ed25519 PureEdDSA over those bytes; the 64-byte
 * signature is hex-encoded with an `ed25519:` prefix (`ed25519:<128 hex chars>`).
 *
 * Key resolution: EvidenceReceipt v2 does NOT embed its signer public key in
 * the envelope. The caller must supply the raw 32-byte Ed25519 key hex via the
 * key provider, keyed by `signature.signer_key_id`. When the key is absent the
 * signature axis SKIPs and the verdict is INCOMPLETE, never a hiding PASS.
 *
 * Chain rule: `chain_prev_hash` is `"sha256:" + hex(SHA-256(JCS(full predecessor
 * receipt)))` for non-genesis receipts. Genesis receipts set `chain_prev_hash`
 * to `"genesis"` or `"sha256:0"` (both are accepted). Unlike AERF, the chain
 * hash covers the full predecessor receipt INCLUDING its signature.
 *
 * Canonicalization: Pipelock v2 uses strict RFC 8785 JCS (NFC strings,
 * lexicographic UTF-16 code-unit key sort). We reuse `jcsRfc8785` which
 * byte-matches Pipelock's Go canonicaliser for all-ASCII field sets and NFC input.
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

const RECORD_TYPE = "evidence_receipt_v2";
const RECEIPT_VERSION = 2;
const SIG_PREFIX = "ed25519:";

const GENESIS_SENTINELS = new Set(["genesis", "sha256:0"]);

const REQUIRED_ENVELOPE = [
  "record_type",
  "receipt_version",
  "payload_kind",
  "event_id",
  "timestamp",
  "signature",
] as const;

const PAYLOAD_KINDS = new Set([
  "proxy_decision",
  "contract_ratified",
  "contract_promote_intent",
  "contract_promote_committed",
  "contract_rollback_authorized",
  "contract_rollback_committed",
  "contract_demoted",
  "contract_expired",
  "contract_drift",
  "shadow_delta",
  "opportunity_missing",
  "key_rotation",
  "contract_redaction_request",
]);

const PAYLOAD_AUTHORITY: Record<string, string> = {
  proxy_decision: "receipt-signing",
  contract_ratified: "receipt-signing",
  contract_promote_intent: "contract-activation-signing",
  contract_promote_committed: "receipt-signing",
  contract_rollback_authorized: "contract-activation-signing",
  contract_rollback_committed: "receipt-signing",
  contract_demoted: "receipt-signing",
  contract_expired: "receipt-signing",
  contract_drift: "receipt-signing",
  shadow_delta: "receipt-signing",
  opportunity_missing: "receipt-signing",
  key_rotation: "contract-activation-signing",
  contract_redaction_request: "contract-activation-signing",
};

function safeHex(value: unknown): Uint8Array {
  if (typeof value !== "string") return new Uint8Array(0);
  if (value.length % 2 !== 0 || !/^[0-9a-fA-F]*$/.test(value)) return new Uint8Array(0);
  const out = new Uint8Array(value.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(value.slice(i * 2, i * 2 + 2), 16);
  return out;
}

/**
 * Build the signable preimage: full receipt with the `signature` sub-object
 * zeroed out (all four fields set to empty strings), then JCS RFC 8785.
 * Mirrors `pipelock_verify._evidence._signable_preimage`.
 */
function signablePreimage(doc: Record<string, unknown>): Uint8Array {
  const clone: Record<string, unknown> = { ...doc };
  clone["signature"] = {
    signer_key_id: "",
    key_purpose: "",
    algorithm: "",
    signature: "",
  };
  return jcsRfc8785(clone);
}

/** SHA-256 of the full receipt (signature included) - the chain hash primitive. */
function fullReceiptHash(doc: Record<string, unknown>): string {
  return sha256Hex(jcsRfc8785(doc));
}

export class PipelockEvidenceAdapter extends FormatAdapter {
  readonly name = "pipelock-evidence-v2";

  detect(doc: Record<string, unknown>): boolean {
    return (
      doc["record_type"] === RECORD_TYPE &&
      doc["receipt_version"] === RECEIPT_VERSION &&
      doc["signature"] !== null &&
      typeof doc["signature"] === "object" &&
      !Array.isArray(doc["signature"])
    );
  }

  extractSignature(doc: Record<string, unknown>): SignatureMaterial {
    const sigProof = (doc["signature"] ?? {}) as Record<string, unknown>;
    const sigValue = sigProof["signature"] ?? "";
    let sigBytes: Uint8Array;
    if (typeof sigValue === "string" && sigValue.startsWith(SIG_PREFIX)) {
      sigBytes = safeHex(sigValue.slice(SIG_PREFIX.length));
    } else {
      sigBytes = new Uint8Array(0);
    }
    return {
      sig: sigBytes,
      alg: "Ed25519",
      kid: (sigProof["signer_key_id"] as string) ?? "",
    };
  }

  resolveKey(
    doc: Record<string, unknown>,
    keyProvider: KeyProvider,
  ): readonly [Uint8Array | null, string] {
    const sigProof = (doc["signature"] ?? {}) as Record<string, unknown>;
    const kid = (sigProof["signer_key_id"] as string) ?? "";
    const provider = keyProvider ?? {};
    const material = provider[kid];
    if (material === undefined || material === null) {
      return [null, `signer_key_id '${kid}' not in key provider`];
    }
    const raw = material instanceof Uint8Array ? material : safeHex(String(material));
    if (raw.length !== 32) {
      return [null, `signer_key_id '${kid}' key is ${raw.length} bytes, expected 32`];
    }
    return [raw, `resolved signer_key_id '${kid}'`];
  }

  signingInput(doc: Record<string, unknown>): Uint8Array {
    return signablePreimage(doc);
  }

  chainStep(doc: Record<string, unknown>): ChainStep {
    const prev = doc["chain_prev_hash"];
    const prevStr = typeof prev === "string" ? prev : null;
    const isGenesis = prevStr === null || GENESIS_SENTINELS.has(prevStr);
    return {
      prevField: prevStr,
      isGenesis,
      recompute: (pred) => "sha256:" + fullReceiptHash(pred),
    };
  }

  schema(doc: Record<string, unknown>): AxisCheck {
    const missing = REQUIRED_ENVELOPE.filter((f) => doc[f] === undefined || doc[f] === null);
    if (missing.length > 0) {
      return ["FAIL", `pipelock-evidence-v2 missing required fields: ${missing.join(",")}`];
    }

    if (doc["record_type"] !== RECORD_TYPE) {
      return ["FAIL", `record_type must be '${RECORD_TYPE}'`];
    }
    if (doc["receipt_version"] !== RECEIPT_VERSION) {
      return ["FAIL", `receipt_version must be ${RECEIPT_VERSION}`];
    }

    const payloadKind = (doc["payload_kind"] as string) ?? "";
    if (!PAYLOAD_KINDS.has(payloadKind)) {
      return ["FAIL", `unknown payload_kind: '${payloadKind}'`];
    }

    const sigProof = doc["signature"] as Record<string, unknown>;
    if (typeof sigProof !== "object" || Array.isArray(sigProof)) {
      return ["FAIL", "signature must be an object"];
    }

    const algorithm = (sigProof["algorithm"] as string) ?? "";
    if (algorithm !== "ed25519") {
      return ["FAIL", `unsupported signature algorithm '${algorithm}'; only ed25519 is implemented`];
    }

    const keyPurpose = (sigProof["key_purpose"] as string) ?? "";
    const requiredPurpose = PAYLOAD_AUTHORITY[payloadKind];
    if (requiredPurpose && keyPurpose !== requiredPurpose) {
      return [
        "FAIL",
        `key_purpose mismatch: '${payloadKind}' requires '${requiredPurpose}', got '${keyPurpose}'`,
      ];
    }

    const sigValue = (sigProof["signature"] as string) ?? "";
    if (typeof sigValue !== "string" || !sigValue.startsWith(SIG_PREFIX)) {
      return ["FAIL", `signature.signature must start with '${SIG_PREFIX}'`];
    }

    return [
      "PASS",
      `pipelock-evidence-v2 envelope; payload_kind='${payloadKind}'; key_purpose authority-matrix check passed`,
    ];
  }
}
