/**
 * Offline DSSE attestation verifier - verify a POST /v1/attest envelope locally,
 * with no round-trip to any Asqav server. Signing stays remote; verification is
 * air-gapped.
 *
 * The envelope is a DSSE (Dead Simple Signing Envelope) wrapping an in-toto
 * Statement v1, signed with ML-DSA-65. The signed bytes are the DSSE
 * Pre-Authentication Encoding (PAE), not the raw JSON:
 *
 *   PAE(type, body) = "DSSEv1" SP LEN(type) SP type SP LEN(body) SP body
 *
 * SP is a single 0x20 space and LEN is the ASCII-decimal byte length. This is a
 * byte-for-byte port of the backend `core/dsse.py` `pae()`, so an envelope the
 * cloud signs re-derives to the identical bytes here.
 *
 * Verify recomputes PAE over the DECODED payload bytes, exactly as the backend
 * `verify_dsse_envelope` does: canonicalization never sits on the trust path, so
 * a verifier never re-encodes the statement and cannot diverge from the signer.
 *
 * Verdict is fail-closed: a missing key, a revoked key, an unsupported algorithm,
 * or a signature that does not check is a FAIL, never a PASS or a skip.
 */

import { DuplicateJsonMemberError, parseJsonPreservingFloats } from "./canonical.js";
import { verifySignature } from "./crypto.js";
import {
  CLASSIFICATION,
  INVALID,
  PASS,
  REASON_CLASSES,
  UNVERIFIABLE,
  type Classification,
} from "./taxonomy.js";
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
 * DSSE Pre-Authentication Encoding of (payloadType, body).
 *
 * Returns `"DSSEv1" SP LEN(type) SP type SP LEN(body) SP body` with single 0x20
 * separators and ASCII-decimal byte lengths - the exact bytes a signer signs and
 * a verifier re-derives. Byte-compatible with `core/dsse.py` `pae()`.
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
 * Pull the sha256 hex digest an in-toto Statement binds its subject to.
 *
 * Returns `subject[0].digest.sha256` lowercased, or null when absent. A push
 * guard asserts this equals the pushed commit sha to bind the attestation to the
 * exact bytes that landed.
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

/**
 * One axis of an attestation verdict. Criterion 418: a failure is INVALID (a
 * binding the check refuted) or UNVERIFIABLE (a recomputation that could not
 * complete); the two never collapse, and neither is ever the PASS outcome.
 */
export interface AttestationAxis {
  result: "PASS" | "INVALID" | "UNVERIFIABLE";
  note: string;
  reasonCode: string;
}

export interface AttestationAxes {
  signature: AttestationAxis;
  key_status: AttestationAxis;
  structure: AttestationAxis;
}

export interface AttestationVerdict {
  verdict: "PASS" | "INVALID" | "UNVERIFIABLE";
  /** Wire classification mirroring the verdict: valid/invalid/unverifiable. */
  classification: Classification;
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
  /** Criterion 418 class token for a failed gate. */
  reasonCode: string;
}

/** Fail-closed gate on the JWKS key: present, not revoked, status outside the revoked set. */
function evaluateKeyStatus(jwks: Record<string, unknown> | null, kid: string): KeyEvaluation {
  const [pub, status, alg] = resolveKey(jwks, kid);
  if (pub === null) {
    return {
      good: false,
      note: `no key published for kid ${JSON.stringify(kid)}`,
      pub: null,
      alg: null,
      reasonCode: "key_unresolvable",
    };
  }
  const revokedAt = resolveRevokedAt(jwks, kid);
  if (revokedAt) {
    return {
      good: false,
      note: `key ${kid} revoked at ${revokedAt}; attestation cannot be trusted`,
      pub,
      alg,
      reasonCode: "key_changed",
    };
  }
  const s = (status ?? "").toLowerCase();
  if (REVOKED_KEY_STATUSES.has(s)) {
    return {
      good: false,
      note: `key ${kid} status ${JSON.stringify(status)} is not active`,
      pub,
      alg,
      reasonCode: "key_changed",
    };
  }
  return {
    good: true,
    note: `key ${kid} status ${JSON.stringify(status ?? "active")} is active`,
    pub,
    alg,
    reasonCode: "none",
  };
}

interface StructureOutcome {
  axis: AttestationAxis;
  payloadBytes: Uint8Array | null;
  signatures: unknown[];
  subjectDigest: string | null;
  predicateType: string | null;
}

function checkAttestationStructure(envelope: unknown, expectedType: string): StructureOutcome {
  // Structural failures are UNVERIFIABLE: the check cannot complete on them
  const bad = (note: string, reasonCode: string): StructureOutcome => ({
    axis: { result: UNVERIFIABLE, note, reasonCode },
    payloadBytes: null,
    signatures: [],
    subjectDigest: null,
    predicateType: null,
  });
  if (!isRecord(envelope)) return bad("envelope is not an object", "member_malformed");
  if (envelope.payloadType !== expectedType) {
    return bad(
      `payloadType ${JSON.stringify(envelope.payloadType)} is not ${JSON.stringify(expectedType)}`,
      "member_malformed",
    );
  }
  if (typeof envelope.payload !== "string") return bad("payload is not a base64 string", "member_malformed");
  const sigs = envelope.signatures;
  if (!Array.isArray(sigs) || sigs.length === 0) {
    return bad("signatures array is missing or empty", "member_malformed");
  }
  let payloadBytes: Uint8Array;
  try {
    payloadBytes = b64decode(envelope.payload);
  } catch (exc) {
    return bad(`payload is not valid base64: ${(exc as Error).message}`, "member_malformed");
  }
  let statement: unknown;
  try {
    // Strict ingest (criterion 419): a duplicate member inside the statement is
    // a terminal parse failure, before any binding check runs
    statement = parseJsonPreservingFloats(new TextDecoder().decode(payloadBytes));
  } catch (exc) {
    const reason = exc instanceof DuplicateJsonMemberError ? "duplicate_member" : "parse_failed";
    return bad("payload does not decode to a JSON document", reason);
  }
  if (!isRecord(statement)) return bad("decoded payload is not a JSON object", "member_malformed");
  if (statement._type !== IN_TOTO_STATEMENT_TYPE) {
    return bad(
      `statement _type ${JSON.stringify(statement._type)} is not ${JSON.stringify(IN_TOTO_STATEMENT_TYPE)}`,
      "member_malformed",
    );
  }
  const subjectDigest = extractSubjectDigest(statement);
  if (subjectDigest === null) return bad("statement subject[0].digest.sha256 is missing", "member_malformed");
  const predicateType = typeof statement.predicateType === "string" ? statement.predicateType : null;
  if (predicateType === null) return bad("statement predicateType is missing", "member_malformed");
  return {
    axis: {
      result: PASS,
      note: `in-toto Statement v1; predicateType ${predicateType}; subject digest ${subjectDigest}`,
      reasonCode: "none",
    },
    payloadBytes,
    signatures: sigs,
    subjectDigest,
    predicateType,
  };
}

/**
 * Verify a DSSE attestation envelope fully offline against an in-memory JWKS.
 *
 * Steps: decode the payload to the in-toto Statement; for each signature resolve
 * the JWKS key (fail-closed on missing/revoked); recompute PAE over the decoded
 * payload bytes and check the ML-DSA-65 signature. PASS requires a well-formed
 * Statement plus at least one signature that both verifies and resolves to an
 * active key. No network call is made.
 *
 * @param envelope - Parsed DSSE envelope ({payloadType, payload, signatures}).
 * @param jwks - JWKS object previously fetched via `fetchJwks()`.
 * @param opts - Optional overrides (expected payloadType).
 */
export function verifyAttestation(
  envelope: unknown,
  jwks: Record<string, unknown> | null,
  opts: VerifyAttestationOptions = {},
): AttestationVerdict {
  const expectedType = opts.payloadType ?? IN_TOTO_PAYLOAD_TYPE;
  const structure = checkAttestationStructure(envelope, expectedType);
  const { payloadBytes, signatures, subjectDigest, predicateType } = structure;

  let sigAxis: AttestationAxis = { result: UNVERIFIABLE, note: "no signature verified", reasonCode: "signature_unchecked" };
  let keyAxis: AttestationAxis = { result: UNVERIFIABLE, note: "no key resolved", reasonCode: "key_unresolvable" };
  let anyVerifies = false;
  let trusted = false;
  const sigNotes: string[] = [];
  // Every failure reason seen; the verdict's class derives from these (418)
  const failureReasons: string[] = [];
  if (structure.axis.result !== PASS) failureReasons.push(structure.axis.reasonCode);

  if (payloadBytes !== null) {
    const paeBytes = buildPae(expectedType, payloadBytes);
    for (const entry of signatures) {
      if (!isRecord(entry)) {
        sigNotes.push("signature entry is not an object");
        failureReasons.push("member_malformed");
        continue;
      }
      const kid = typeof entry.keyid === "string" ? entry.keyid : "";
      const key = evaluateKeyStatus(jwks, kid);
      if (key.good) {
        keyAxis = { result: PASS, note: key.note, reasonCode: "none" };
      } else if (keyAxis.result !== PASS) {
        keyAxis = { result: REASON_CLASSES[key.reasonCode] ?? UNVERIFIABLE, note: key.note, reasonCode: key.reasonCode };
        failureReasons.push(key.reasonCode);
      }
      if (typeof entry.sig !== "string") {
        sigNotes.push(`kid ${kid || "?"}: sig field is not a string`);
        failureReasons.push("member_malformed");
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
        failureReasons.push("signature_malformed");
        continue;
      }
      const outcome = verifySignature(key.alg ?? "ML-DSA-65", key.pub, paeBytes, sigBytes);
      if (outcome.result === PASS) {
        anyVerifies = true;
        sigAxis = { result: PASS, note: `kid ${kid || "?"}: ${outcome.note}`, reasonCode: "none" };
        if (key.good) trusted = true;
      } else {
        sigNotes.push(`kid ${kid || "?"}: ${outcome.note}`);
        failureReasons.push(outcome.reasonCode);
      }
    }
    if (!anyVerifies) {
      // The signature axis class is the worst class seen: an INVALID binding
      // failure dominates an UNVERIFIABLE could-not-check (criterion 418)
      const worst = failureReasons.some((r) => REASON_CLASSES[r] === INVALID)
        ? INVALID
        : UNVERIFIABLE;
      const worstReason = failureReasons.find((r) => REASON_CLASSES[r] === worst) ?? "signature_unchecked";
      sigAxis = {
        result: worst,
        note: sigNotes.length > 0 ? sigNotes.join("; ") : "no signature verified",
        reasonCode: worstReason,
      };
    }
  }

  // Criterion 418: PASS needs structure plus a trusted verified signature; any
  // INVALID-class failure dominates, otherwise the verdict is UNVERIFIABLE
  let verdict: "PASS" | "INVALID" | "UNVERIFIABLE";
  if (structure.axis.result === PASS && trusted) {
    verdict = PASS;
  } else if (failureReasons.some((r) => REASON_CLASSES[r] === INVALID)) {
    verdict = INVALID;
  } else {
    verdict = UNVERIFIABLE;
  }
  let reason: string;
  if (verdict === PASS) {
    reason = `DSSE attestation verified over PAE; subject digest ${subjectDigest}`;
  } else if (structure.axis.result !== PASS) {
    reason = `structure: ${structure.axis.note}`;
  } else if (!anyVerifies) {
    reason = `signature: ${sigAxis.note}`;
  } else {
    reason = `key_status: ${keyAxis.note}`;
  }

  return {
    verdict,
    classification: CLASSIFICATION[verdict],
    subjectDigest,
    predicateType,
    axes: { signature: sigAxis, key_status: keyAxis, structure: structure.axis },
    reason,
  };
}
