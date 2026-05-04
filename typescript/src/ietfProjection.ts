/**
 * IETF Compliance Receipts -01 wire-shape projection helpers.
 *
 * Mirror of `asqav.ietf_projection` (Python) and
 * `asqav_cloud.core.ietf_projection`. Pure functions that re-shape the
 * legacy flat fields into the spec-shape JSON structures the IETF
 * profile defines (Section 4 "JSON Receipt Structure" + Section 4.4
 * "Anchors").
 *
 * Callers use these helpers to produce a draft-conformant view of a
 * receipt for offline verification or audit-pack export when working
 * against an older cloud that does not yet emit `signatureObject` /
 * `anchors` directly.
 */

/** Spec-shape signature envelope per Section 4. */
export interface SignatureEnvelope {
  alg: string;
  kid: string;
  sig: string;
}

/** Spec-shape anchor entry per Section 4.4. */
export interface AnchorEntry {
  type: "rfc3161" | "opentimestamps" | string;
  value: string;
  serial?: string;
  tsa?: string;
  status?: string;
  [key: string]: unknown;
}

/** Minimal duck-typed slice of a SignatureResponse / verify dict the
 * projection helpers read. */
export interface ProjectableResponse {
  algorithm?: string;
  signature?: string;
  key_id?: string;
  kid?: string;
  complianceMode?: boolean;
  compliance_mode?: boolean;
  signatureObject?: SignatureEnvelope | null;
  anchors?: AnchorEntry[] | null;
  rfc3161_serial?: string;
  rfc3161_tsa?: string;
  rfc3161Serial?: string;
  rfc3161Tsa?: string;
  bitcoinAnchor?: { status?: string } | null;
  bitcoin_anchor?: { status?: string } | null;
  [key: string]: unknown;
}

function pick<T>(obj: ProjectableResponse, ...names: string[]): T | undefined {
  for (const n of names) {
    const v = (obj as Record<string, unknown>)[n];
    if (v !== undefined && v !== null) return v as T;
  }
  return undefined;
}

function isComplianceMode(obj: ProjectableResponse): boolean {
  return Boolean(pick<boolean>(obj, "complianceMode", "compliance_mode"));
}

/** Project a SignatureResponse / verify dict into `{alg, kid, sig}`.
 *
 * Returns the cloud-supplied `signatureObject` when present so callers
 * do not double-project; otherwise builds the envelope from the flat
 * fields. Returns `undefined` for non-compliance receipts so legacy
 * callers stay byte-stable. */
export function signatureObjectFromResponse(
  response: ProjectableResponse | null | undefined,
): SignatureEnvelope | undefined {
  if (response == null) return undefined;
  if (response.signatureObject) {
    // Spread to a fresh object so callers cannot mutate the source.
    return { ...response.signatureObject };
  }
  if (!isComplianceMode(response)) return undefined;
  return {
    alg: pick<string>(response, "algorithm") ?? "",
    kid: pick<string>(response, "key_id", "kid") ?? "",
    sig: pick<string>(response, "signature") ?? "",
  };
}

/** Project a SignatureResponse / verify dict into `anchors[]`.
 *
 * Returns the cloud-supplied list when present (deep-copied), otherwise
 * builds entries from the flat columns. Returns `undefined` for
 * non-compliance receipts; returns `[]` (not undefined) when
 * complianceMode is on but no anchor columns are populated. */
export function anchorsFromResponse(
  response: ProjectableResponse | null | undefined,
): AnchorEntry[] | undefined {
  if (response == null) return undefined;
  if (Array.isArray(response.anchors)) {
    return response.anchors.map((e) => ({ ...e }));
  }
  if (!isComplianceMode(response)) return undefined;

  const anchors: AnchorEntry[] = [];

  const rfc3161Serial = pick<string>(response, "rfc3161Serial", "rfc3161_serial");
  const rfc3161Tsa = pick<string>(response, "rfc3161Tsa", "rfc3161_tsa");
  if (rfc3161Serial || rfc3161Tsa) {
    const entry: AnchorEntry = { type: "rfc3161", value: "" };
    if (rfc3161Serial) entry.serial = rfc3161Serial;
    if (rfc3161Tsa) entry.tsa = rfc3161Tsa;
    anchors.push(entry);
  }

  const bitcoinAnchor = pick<{ status?: string }>(
    response,
    "bitcoinAnchor",
    "bitcoin_anchor",
  );
  const status = bitcoinAnchor?.status;
  if (status) {
    anchors.push({ type: "opentimestamps", value: "", status });
  }

  return anchors;
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
