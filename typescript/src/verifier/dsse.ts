/**
 * Offline DSSE attestation verifier: signing stays remote, verification is air-gapped. Signed bytes are
 * the PAE, a byte-for-byte port of `core/dsse.py`. Fail-closed: anything unresolved is a FAIL.
 */

import { verifySignature } from "./crypto.js";
import { DuplicateMemberError, parseJsonStrict } from "./canonical.js";
import { b64decode, resolveKey, resolveRevokedAt } from "./vrShim.js";

/** DSSE payloadType for an in-toto Statement (mirrors `core/dsse.py`). */
export const IN_TOTO_PAYLOAD_TYPE = "application/vnd.in-toto+json";

/** in-toto Statement v1 `_type` sentinel (mirrors `core/dsse.py`). */
export const IN_TOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1";

// Same revoked set as `vrShim.checkKeyStatus`; an attestation from one of these
// keys must not PASS offline.
const REVOKED_KEY_STATUSES = new Set(["revoked", "suspended", "compromised"]);

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

/**
 * DSSE Pre-Authentication Encoding of (payloadType, body): `"DSSEv1" SP LEN(type) SP type SP LEN(body)
 * SP body`, with 0x20 separators and ASCII-decimal lengths. Byte-compatible with `core/dsse.py`.
 */
export function buildPae(payloadType: string | Uint8Array, body: Uint8Array): Uint8Array {
  const enc = new TextEncoder();
  const typeBytes = typeof payloadType === "string" ? enc.encode(payloadType) : payloadType;
  const parts: Uint8Array[] = [
    enc.encode("DSSEv1"),
    enc.encode(String(typeBytes.length)),
    typeBytes,
    enc.encode(String(body.length)),
    body,
  ];
  let total = parts.length - 1; // one 0x20 separator between each pair
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let off = 0;
  for (let i = 0; i < parts.length; i += 1) {
    if (i > 0) out[off++] = 0x20;
    out.set(parts[i], off);
    off += parts[i].length;
  }
  return out;
}

/**
 * Pull `subject[0].digest.sha256` from an in-toto Statement, lowercased, or null when absent. A push
 * guard asserts it equals the pushed commit sha, binding the attestation to the bytes that landed.
 */
export function extractSubjectDigest(statement: unknown): string | null {
  if (!isRecord(statement)) return null;
  const subject = statement.subject;
  if (!Array.isArray(subject) || subject.length === 0) return null;
  const first = subject[0];
  if (!isRecord(first)) return null;
  const digest = first.digest;
  if (!isRecord(digest)) return null;
  const sha = digest.sha256;
  if (typeof sha !== "string" || sha.length === 0) return null;
  return sha.toLowerCase();
}

/** One axis of an attestation verdict; attestations are PASS/FAIL only, never skip. */
export interface AttestationAxis {
  result: "PASS" | "FAIL";
  note: string;
}

export interface AttestationAxes {
  signature: AttestationAxis;
  key_status: AttestationAxis;
  structure: AttestationAxis;
}

export interface AttestationVerdict {
  verdict: "PASS" | "FAIL";
  subjectDigest: string | null;
  predicateType: string | null;
  axes: AttestationAxes;
  reason: string;
}

export interface VerifyAttestationOptions {
  /** Expected DSSE payloadType (default: the in-toto Statement type). */
  payloadType?: string;
}

interface KeyEvaluation {
  good: boolean;
  note: string;
  pub: Uint8Array | null;
  alg: string | null;
}

/** Fail-closed gate on the JWKS key: present, not revoked, status outside the revoked set. */
function evaluateKeyStatus(jwks: Record<string, unknown> | null, kid: string): KeyEvaluation {
  const [pub, status, alg] = resolveKey(jwks, kid);
  if (pub === null) {
    return { good: false, note: `no key published for kid ${JSON.stringify(kid)}`, pub: null, alg: null };
  }
  const revokedAt = resolveRevokedAt(jwks, kid);
  if (revokedAt) {
    return { good: false, note: `key ${kid} revoked at ${revokedAt}; attestation cannot be trusted`, pub, alg };
  }
  const s = (status ?? "").toLowerCase();
  if (REVOKED_KEY_STATUSES.has(s)) {
    return { good: false, note: `key ${kid} status ${JSON.stringify(status)} is not active`, pub, alg };
  }
  return { good: true, note: `key ${kid} status ${JSON.stringify(status ?? "active")} is active`, pub, alg };
}

interface StructureOutcome {
  axis: AttestationAxis;
  payloadBytes: Uint8Array | null;
  signatures: unknown[];
  subjectDigest: string | null;
  predicateType: string | null;
}

function checkAttestationStructure(envelope: unknown, expectedType: string): StructureOutcome {
  const bad = (note: string): StructureOutcome => ({
    axis: { result: "FAIL", note },
    payloadBytes: null,
    signatures: [],
    subjectDigest: null,
    predicateType: null,
  });
  if (!isRecord(envelope)) return bad("envelope is not an object");
  if (envelope.payloadType !== expectedType) {
    return bad(`payloadType ${JSON.stringify(envelope.payloadType)} is not ${JSON.stringify(expectedType)}`);
  }
  if (typeof envelope.payload !== "string") return bad("payload is not a base64 string");
  const sigs = envelope.signatures;
  if (!Array.isArray(sigs) || sigs.length === 0) return bad("signatures array is missing or empty");
  let payloadBytes: Uint8Array;
  try {
    payloadBytes = b64decode(envelope.payload);
  } catch (exc) {
    return bad(`payload is not valid base64: ${(exc as Error).message}`);
  }
  let statement: unknown;
  try {
    // Strict ingest (419): a duplicated member name is a terminal parse failure.
    statement = parseJsonStrict(new TextDecoder().decode(payloadBytes));
  } catch (exc) {
    if (exc instanceof DuplicateMemberError) return bad(`payload rejected: ${exc.message}`);
    return bad("payload does not decode to a JSON document");
  }
  if (!isRecord(statement)) return bad("decoded payload is not a JSON object");
  if (statement._type !== IN_TOTO_STATEMENT_TYPE) {
    return bad(`statement _type ${JSON.stringify(statement._type)} is not ${JSON.stringify(IN_TOTO_STATEMENT_TYPE)}`);
  }
  const subjectDigest = extractSubjectDigest(statement);
  if (subjectDigest === null) return bad("statement subject[0].digest.sha256 is missing");
  const predicateType = typeof statement.predicateType === "string" ? statement.predicateType : null;
  if (predicateType === null) return bad("statement predicateType is missing");
  return {
    axis: {
      result: "PASS",
      note: `in-toto Statement v1; predicateType ${predicateType}; subject digest ${subjectDigest}`,
    },
    payloadBytes,
    signatures: sigs,
    subjectDigest,
    predicateType,
  };
}

/**
 * Verify a DSSE attestation envelope fully offline against an in-memory JWKS. PASS needs a well-formed
 * Statement plus one signature that verifies against an active key. No network call is made.
 */
export function verifyAttestation(
  envelope: unknown,
  jwks: Record<string, unknown> | null,
  opts: VerifyAttestationOptions = {},
): AttestationVerdict {
  const expectedType = opts.payloadType ?? IN_TOTO_PAYLOAD_TYPE;
  const structure = checkAttestationStructure(envelope, expectedType);
  const { payloadBytes, signatures, subjectDigest, predicateType } = structure;

  let sigAxis: AttestationAxis = { result: "FAIL", note: "no signature verified" };
  let keyAxis: AttestationAxis = { result: "FAIL", note: "no key resolved" };
  let anyVerifies = false;
  let trusted = false;
  const sigNotes: string[] = [];

  if (payloadBytes !== null) {
    const paeBytes = buildPae(expectedType, payloadBytes);
    for (const entry of signatures) {
      if (!isRecord(entry)) {
        sigNotes.push("signature entry is not an object");
        continue;
      }
      const kid = typeof entry.keyid === "string" ? entry.keyid : "";
      const key = evaluateKeyStatus(jwks, kid);
      if (key.good) {
        keyAxis = { result: "PASS", note: key.note };
      } else if (keyAxis.result !== "PASS") {
        keyAxis = { result: "FAIL", note: key.note };
      }
      if (typeof entry.sig !== "string") {
        sigNotes.push(`kid ${kid || "?"}: sig field is not a string`);
        continue;
      }
      if (key.pub === null) {
        sigNotes.push(`kid ${kid || "?"}: ${key.note}`);
        continue;
      }
      let sigBytes: Uint8Array;
      try {
        sigBytes = b64decode(entry.sig);
      } catch {
        sigNotes.push(`kid ${kid || "?"}: sig is not valid base64`);
        continue;
      }
      const outcome = verifySignature(key.alg ?? "ML-DSA-65", key.pub, paeBytes, sigBytes);
      if (outcome.result === "PASS") {
        anyVerifies = true;
        sigAxis = { result: "PASS", note: `kid ${kid || "?"}: ${outcome.note}` };
        if (key.good) trusted = true;
      } else {
        sigNotes.push(`kid ${kid || "?"}: ${outcome.note}`);
      }
    }
    if (!anyVerifies) {
      sigAxis = { result: "FAIL", note: sigNotes.length > 0 ? sigNotes.join("; ") : "no signature verified" };
    }
  }

  const verdict: "PASS" | "FAIL" = structure.axis.result === "PASS" && trusted ? "PASS" : "FAIL";
  let reason: string;
  if (verdict === "PASS") {
    reason = `DSSE attestation verified over PAE; subject digest ${subjectDigest}`;
  } else if (structure.axis.result === "FAIL") {
    reason = `structure: ${structure.axis.note}`;
  } else if (!anyVerifies) {
    reason = `signature: ${sigAxis.note}`;
  } else {
    reason = `key_status: ${keyAxis.note}`;
  }

  return {
    verdict,
    subjectDigest,
    predicateType,
    axes: { signature: sigAxis, key_status: keyAxis, structure: structure.axis },
    reason,
  };
}
