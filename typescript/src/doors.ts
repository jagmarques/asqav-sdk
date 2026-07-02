// Standards-interop doors: one Asqav receipt, N native envelopes (W3C VC, CloudEvents,
// OTel GenAI, C2PA, ERC-8004). Inner receipt is byte-identical across every door.
// Pure and offline: no signing, no network, no mutation, no clock reads. See doors.py.
import { createHash } from "node:crypto";

import { canonicalJson } from "./jcs.js";

export type Receipt = Record<string, unknown>;

export const VC_V2_CONTEXT = "https://www.w3.org/ns/credentials/v2";
export const ASQAV_DOORS_CONTEXT = "https://asqav.com/context/doors/v1";
export const VC_CREDENTIAL_TYPE = ["VerifiableCredential", "AsqavReceiptCredential"];
export const VC_PROOF_TYPE = "AsqavReceiptSignature2026";

export const CLOUDEVENT_TYPE = "com.asqav.receipt.v1";
export const CLOUDEVENT_DIGEST_ATTR = "asqavreceiptdigest";

export const C2PA_ASSERTION_LABEL = "com.asqav.receipt.v1";

export const OTEL_RECEIPT_ATTR = "asqav.receipt";
export const OTEL_DIGEST_ATTR = "asqav.receipt.digest";
export const OTEL_SIGNATURE_ATTR = "asqav.signature";

export const ERC8004_ZERO_ADDRESS = "0x" + "00".repeat(20);

/** JCS canonical string of the receipt; the same bytes both SDKs hash. */
function canonicalString(receipt: Receipt): string {
  return new TextDecoder().decode(canonicalJson(receipt));
}

/** JCS canonical bytes of the receipt. */
export function canonicalReceiptBytes(receipt: Receipt): Uint8Array {
  return canonicalJson(receipt);
}

/** `sha256:<hex>` over the raw canonical receipt bytes (the C2PA raw-bytes rule). */
export function receiptDigest(receipt: Receipt): string {
  return "sha256:" + sha256Hex(receipt);
}

function sha256Hex(receipt: Receipt): string {
  return createHash("sha256").update(Buffer.from(canonicalReceiptBytes(receipt))).digest("hex");
}

function requireObject(receipt: unknown): Receipt {
  if (receipt === null || typeof receipt !== "object" || Array.isArray(receipt)) {
    throw new TypeError("receipt must be a parsed JSON object");
  }
  return receipt as Receipt;
}

function payload(receipt: Receipt): Receipt {
  const p = receipt.payload;
  return p !== null && typeof p === "object" && !Array.isArray(p) ? (p as Receipt) : receipt;
}

function signatureBits(receipt: Receipt): { sig: unknown; alg: unknown; kid: unknown } {
  const sig = receipt.signature;
  if (sig !== null && typeof sig === "object" && !Array.isArray(sig)) {
    const s = sig as Receipt;
    return { sig: s.sig, alg: s.alg, kid: s.kid };
  }
  if (sig !== undefined && sig !== null) {
    return { sig, alg: receipt.algorithm, kid: receipt.key_id };
  }
  return { sig: receipt.signature_b64, alg: receipt.algorithm, kid: receipt.key_id };
}

function field(receipt: Receipt, ...keys: string[]): unknown {
  const p = payload(receipt);
  for (const key of keys) {
    if (p[key] !== undefined && p[key] !== null) return p[key];
    if (receipt[key] !== undefined && receipt[key] !== null) return receipt[key];
  }
  return undefined;
}

const agentId = (r: Receipt) => field(r, "agent_id");
const actionId = (r: Receipt) => field(r, "action_id", "action_ref");
const issuerId = (r: Receipt) => field(r, "issuer_id", "org_id");
const timestamp = (r: Receipt) => field(r, "issued_at", "server_timestamp");

/** Deterministic offline pointer to the receipt by digest. Not a live endpoint. */
function verifyUrn(receipt: Receipt): string {
  return "urn:asqav:receipt:" + sha256Hex(receipt);
}

/** Wrap the receipt as a W3C VC 2.0 with the whole receipt as `credentialSubject`. */
export function toW3cVc(receipt: Receipt, opts: { issuer?: string } = {}): Receipt {
  requireObject(receipt);
  const { sig, alg, kid } = signatureBits(receipt);
  const issuer = opts.issuer !== undefined ? opts.issuer : issuerId(receipt);
  const vc: Receipt = {
    "@context": [VC_V2_CONTEXT, ASQAV_DOORS_CONTEXT],
    type: [...VC_CREDENTIAL_TYPE],
    issuer: issuer !== undefined && issuer !== null ? issuer : "https://asqav.com",
    credentialSubject: receipt,
    proof: {
      type: VC_PROOF_TYPE,
      proofPurpose: "assertionMethod",
      signatureAlgorithm: alg ?? null,
      verificationKeyId: kid ?? null,
      signatureValue: sig ?? null,
      receiptDigest: receiptDigest(receipt),
    },
  };
  const ts = timestamp(receipt);
  if (ts !== undefined && ts !== null) vc.validFrom = ts;
  return vc;
}

/** Wrap the receipt as a CloudEvents 1.0 event with the receipt as `data`. */
export function toCloudEvent(receipt: Receipt): Receipt {
  requireObject(receipt);
  const agent = agentId(receipt);
  const action = actionId(receipt);
  const event: Receipt = {
    specversion: "1.0",
    id: action !== undefined && action !== null ? String(action) : receiptDigest(receipt),
    source:
      agent !== undefined && agent !== null ? "urn:asqav:agent:" + String(agent) : "urn:asqav",
    type: CLOUDEVENT_TYPE,
    datacontenttype: "application/json",
    [CLOUDEVENT_DIGEST_ATTR]: receiptDigest(receipt),
    data: receipt,
  };
  const ts = timestamp(receipt);
  if (ts !== undefined && ts !== null) event.time = ts;
  return event;
}

/** Map the receipt to an OTel GenAI `gen_ai.*` attribute object. */
export function toOtelGenaiAttributes(receipt: Receipt): Receipt {
  requireObject(receipt);
  const { sig } = signatureBits(receipt);
  const attrs: Receipt = {};
  const op = field(receipt, "type", "action_type");
  if (op !== undefined && op !== null) attrs["gen_ai.operation.name"] = op;
  const agent = agentId(receipt);
  if (agent !== undefined && agent !== null) attrs["gen_ai.agent.id"] = agent;
  const tool = field(receipt, "tool_name");
  if (tool !== undefined && tool !== null) attrs["gen_ai.tool.name"] = tool;
  const callId = actionId(receipt);
  if (callId !== undefined && callId !== null) attrs["gen_ai.tool.call.id"] = callId;
  const conversation = field(receipt, "session_id");
  if (conversation !== undefined && conversation !== null) {
    attrs["gen_ai.conversation.id"] = conversation;
  }
  attrs[OTEL_RECEIPT_ATTR] = canonicalString(receipt);
  attrs[OTEL_DIGEST_ATTR] = receiptDigest(receipt);
  if (sig !== undefined && sig !== null) attrs[OTEL_SIGNATURE_ATTR] = sig;
  return attrs;
}

/** Wrap the receipt as a C2PA third-party assertion using the raw-bytes rule. */
export function toC2paAssertion(receipt: Receipt): Receipt {
  requireObject(receipt);
  return {
    label: C2PA_ASSERTION_LABEL,
    data: {
      receipt,
      verify: verifyUrn(receipt),
    },
    hash_binding: {
      alg: "sha256",
      rule: "raw-bytes",
      hash: sha256Hex(receipt),
    },
  };
}

/**
 * asqav sha256 `receiptCommitment`, not the ERC-8004 keccak256 `requestHash`. See doors.py.
 */
export function toErc8004ValidationRequest(
  receipt: Receipt,
  opts: { validator?: string; agentId?: string } = {},
): Receipt {
  requireObject(receipt);
  const resolvedAgent = opts.agentId !== undefined ? opts.agentId : agentId(receipt);
  return {
    function: "validationRequest",
    args: {
      validator: opts.validator !== undefined ? opts.validator : ERC8004_ZERO_ADDRESS,
      agentId: resolvedAgent ?? null,
      requestURI: verifyUrn(receipt),
      receiptCommitment: "0x" + sha256Hex(receipt),
    },
    hashAlgorithm: "asqav-commitment-sha256",
    receipt,
  };
}

/** Return every door keyed by standard name; one inner object, N skins. */
export function wrapAll(receipt: Receipt): Receipt {
  requireObject(receipt);
  return {
    w3c_vc: toW3cVc(receipt),
    cloudevent: toCloudEvent(receipt),
    otel_genai: toOtelGenaiAttributes(receipt),
    c2pa: toC2paAssertion(receipt),
    erc8004: toErc8004ValidationRequest(receipt),
  };
}

export function receiptFromVc(vc: Receipt): Receipt {
  return vc.credentialSubject as Receipt;
}

export function receiptFromCloudEvent(event: Receipt): Receipt {
  return event.data as Receipt;
}

export function receiptFromOtelGenaiAttributes(attrs: Receipt): Receipt {
  return JSON.parse(attrs[OTEL_RECEIPT_ATTR] as string) as Receipt;
}

export function receiptFromC2paAssertion(assertion: Receipt): Receipt {
  return (assertion.data as Receipt).receipt as Receipt;
}

export function receiptFromErc8004ValidationRequest(request: Receipt): Receipt {
  return request.receipt as Receipt;
}

/** Recover the inner receipt from any door envelope, detected by shape. */
export function extractReceipt(envelope: Receipt): Receipt {
  requireObject(envelope);
  if ("credentialSubject" in envelope) return receiptFromVc(envelope);
  if (envelope.type === CLOUDEVENT_TYPE || ("specversion" in envelope && "data" in envelope)) {
    return receiptFromCloudEvent(envelope);
  }
  if (OTEL_RECEIPT_ATTR in envelope) return receiptFromOtelGenaiAttributes(envelope);
  if (envelope.label === C2PA_ASSERTION_LABEL) return receiptFromC2paAssertion(envelope);
  if (envelope.function === "validationRequest") {
    return receiptFromErc8004ValidationRequest(envelope);
  }
  throw new Error("unrecognised door envelope");
}
