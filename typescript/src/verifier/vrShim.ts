/**
 * A focused port of the standalone `verify_receipt.py` helpers the Asqav-native
 * and ACTA adapters reuse, so the TS oracle reproduces the same bytes and the
 * same structure verdict as the Python surface.
 *
 * Only the pieces the adapters touch are ported: base64 decoding, envelope
 * normalisation, the Asqav-native structure check, JWKS key resolution, and the
 * first-receipt seed constant.
 */

/** Mirrors `core/integrity.py` FIRST_RECEIPT_SEED (64 zeros). */
export const FIRST_RECEIPT_SEED = "0".repeat(64);

const REQUIRED_FIELDS = [
  "type",
  "issued_at",
  "issuer_id",
  "action_ref",
  "payload_digest",
  "policy_digest",
  "previousReceiptHash",
  "decision",
] as const;

const ALLOWED_TYPES = new Set([
  "protectmcp:decision",
  "protectmcp:restraint",
  "protectmcp:lifecycle",
  "protectmcp:lifecycle:configuration_change",
  "protectmcp:lifecycle:risk_acceptance",
  "protectmcp:lifecycle:code_authorship",
  "protectmcp:acknowledgment",
  "protectmcp:observation",
  "protectmcp:observation:result_bound",
]);

/** Decode standard or url-safe base64, padding-tolerant (mirrors `_b64decode`). */
export function b64decode(value: string): Uint8Array {
  let s = value.replace(/-/g, "+").replace(/_/g, "/");
  s += "=".repeat((-s.length % 4 + 4) % 4);
  return new Uint8Array(Buffer.from(s, "base64"));
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

/**
 * Remap a hosted /verify response into the canonical 3-key envelope. Already
 * canonical or non-hosted shapes pass through unchanged (mirrors
 * `normalise_envelope`).
 */
export function normaliseEnvelope(raw: Record<string, unknown>): Record<string, unknown> {
  const sig = raw.signature;
  if (isRecord(sig) && isRecord(raw.payload)) {
    return raw;
  }
  const payload = raw.payload;
  if (!isRecord(payload)) {
    return raw;
  }
  let sigObj = raw.signature_envelope;
  if (!isRecord(sigObj)) {
    if (isRecord(sig)) {
      sigObj = sig;
    } else {
      sigObj = {
        alg: raw.algorithm ?? "ML-DSA-65",
        kid: payload.issuer_id ?? "",
        sig: typeof sig === "string" ? sig : "",
      };
    }
  }
  return {
    payload,
    signature: sigObj,
    anchors: raw.anchors ?? [],
  };
}

/** Asqav-native structure check; returns `[result, note]` (mirrors `check_structure`). */
export function checkStructure(payload: Record<string, unknown>): readonly ["PASS" | "FAIL", string] {
  const missing = REQUIRED_FIELDS.filter((f) => !(f in payload));
  if (missing.length > 0) {
    return ["FAIL", `missing required fields: ${missing.join(",")}`];
  }
  const rt = payload.type;
  if (typeof rt !== "string" || !ALLOWED_TYPES.has(rt)) {
    return ["FAIL", `type ${JSON.stringify(rt)} outside the allowed namespace`];
  }
  return ["PASS", `required fields present; type ${rt}`];
}

/** Return `[publicKeyBytes, status, alg]` for `kid` from a JWKS dict (mirrors `resolve_key`). */
export function resolveKey(
  jwks: Record<string, unknown> | null,
  kid: string,
): readonly [Uint8Array | null, string | null, string | null] {
  const keys = (jwks?.keys as Array<Record<string, unknown>> | undefined) ?? [];
  for (const k of keys) {
    if (kid && (kid === k.issuer_id || kid === k.kid)) {
      return [b64decode(k.public_key as string), (k.status as string) ?? null, (k.alg as string) ?? null];
    }
  }
  return [null, null, null];
}
