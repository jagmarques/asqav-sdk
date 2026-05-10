/**
 * Compliance Receipts wire-shape projection helpers.
 *
 * Pure helpers for offline analysis of a receipt: extract the
 * spec-shape signature envelope `{alg, kid, sig}` and the anchors[]
 * array. The cloud emits the three-key envelope directly under
 * compliance_mode; the helpers below return the cloud-supplied values
 * and undefined on non-compliance receipts.
 */

/** Spec-shape signature envelope. */
export interface SignatureEnvelope {
  alg: string;
  kid: string;
  sig: string;
}

/** Spec-shape anchor entry. */
export interface AnchorEntry {
  type: "rfc3161" | "opentimestamps" | string;
  value: string;
  serial?: string;
  tsa?: string;
  status?: string;
  commit_hash?: string | null;
  [key: string]: unknown;
}

/** Minimal duck-typed slice of a SignatureResponse / verify dict the
 * projection helpers read. */
export interface ProjectableResponse {
  signature?: string | SignatureEnvelope;
  anchors?: AnchorEntry[] | null;
  payload?: Record<string, unknown> | null;
  complianceMode?: boolean;
  compliance_mode?: boolean;
  [key: string]: unknown;
}

/** Return `{alg, kid, sig}` when the response carries the object form,
 *  else undefined. The cloud emits `signature` as the object form under
 *  compliance_mode and as a base64 string in flat mode. */
export function signatureEnvelopeFromResponse(
  response: ProjectableResponse | null | undefined,
): SignatureEnvelope | undefined {
  if (response == null) return undefined;
  const sig = response.signature;
  if (sig && typeof sig === "object" && "alg" in sig && "kid" in sig && "sig" in sig) {
    return { ...(sig as SignatureEnvelope) };
  }
  return undefined;
}

/** Return the `anchors[]` array when present, else undefined. */
export function anchorsFromResponse(
  response: ProjectableResponse | null | undefined,
): AnchorEntry[] | undefined {
  if (response == null) return undefined;
  if (Array.isArray(response.anchors)) {
    return response.anchors.map((e) => ({ ...e }));
  }
  return undefined;
}

/** Helper for callers that hold raw anchor bytes locally. */
export function encodeAnchorValue(raw: Uint8Array): string {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(raw).toString("base64");
  }
  let s = "";
  for (const b of raw) s += String.fromCharCode(b);
  return btoa(s);
}
