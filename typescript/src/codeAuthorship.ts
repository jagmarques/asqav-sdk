/**
 * SDK half of the un-bypassable code-authorship path.
 *
 * POST /v1/code-authorship is the authoritative ingress. The client supplies an
 * ADVISORY change digest (sha256 of `git diff base..head`) and the server
 * re-fetches the commit, recomputes the canonical diff, and signs an in-toto
 * Statement whose `subject[0].digest.sha256` is the SERVER-recomputed hash. The
 * client digest is never trusted. `digest_match` only reports whether the
 * advisory value agreed with the server's.
 *
 * The authoritative capture layer for a code-authorship receipt is
 * `github_sha_pull`. A receipt whose capture layer is `in_process_sdk` or
 * `passive_telemetry` is a client self-report and can NEVER be an authoritative
 * decision receipt (observation only), mirroring the cloud's
 * `observation_decision_not_allowed` rule.
 */

import { createHash } from "node:crypto";

import { request } from "./index.js";

/** API-key scope required to call POST /v1/code-authorship. */
export const CODE_AUTHORSHIP_WRITE_SCOPE = "code_authorship:write";

/** Endpoint path (joined onto the configured API base). */
export const CODE_AUTHORSHIP_PATH = "/code-authorship";

/** predicateType of the authoritative code-authorship in-toto Statement. */
export const CODE_AUTHORSHIP_PREDICATE_TYPE =
  "https://asqav.com/attestation/code-authorship/v1";

/** in-toto Statement v1 type carried by the envelope. */
export const INTOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1";

/** asset_class the server stamps on the code-authorship predicate. */
export const CODE_AUTHORSHIP_ASSET_CLASS = "code";

/**
 * The ONLY capture layer that makes a code-authorship receipt authoritative.
 * The server stamps it after re-fetching the commit and recomputing the diff.
 */
export const AUTHORITATIVE_CAPTURE_LAYER = "github_sha_pull";

/**
 * Capture layers that are client self-reports. A code-authorship receipt
 * carrying one of these is observation only and never an authoritative
 * decision receipt.
 */
export const OBSERVATION_ONLY_CAPTURE_LAYERS = [
  "in_process_sdk",
  "passive_telemetry",
] as const;

/** Verdict tokens for verifyCodeAuthorshipEnvelope. */
export const VERDICT_PASS = "PASS";
export const VERDICT_REJECT = "REJECT";

/**
 * Compute the ADVISORY change digest `sha256:<hex>`.
 *
 * The advisory digest is sha256 of `git diff base..head`. When `diffText` is
 * supplied it is hashed directly. With no diff the bare head sha is hashed so
 * the field is always a well-formed `sha256:<64 hex>` existence proof. The
 * server recomputes its own digest and signs that. This value only lets the
 * server report `digest_match`.
 */
export function computeAdvisoryDigest(
  headSha: string,
  diffText?: string,
): string {
  const material = diffText !== undefined && diffText !== "" ? diffText : headSha;
  return "sha256:" + createHash("sha256").update(material, "utf8").digest("hex");
}

function bareHex(digest: unknown): string | undefined {
  if (typeof digest !== "string" || digest === "") {
    return undefined;
  }
  return digest.split(":").slice(1).join(":") || digest;
}

/** Parsed response from POST /v1/code-authorship. */
export interface CodeAuthorshipResult {
  /** The server-signed in-toto Statement. */
  envelope: Record<string, unknown>;
  /** The signed receipt carrying the signature envelope. */
  receipt: Record<string, unknown>;
  kid?: string;
  jwksUrl?: string;
  /** The server-recomputed diff hash, `sha256:<hex>`. */
  serverDigest?: string;
  /** Whether the advisory client digest agreed with the server digest. */
  digestMatch: boolean;
  /** The SERVER digest bound into `subject[0].digest.sha256` (bare hex). */
  subjectDigest?: string;
  captureLayer?: string;
  assetClass?: string;
  advisoryClientDigest?: string;
  raw: Record<string, unknown>;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

/** Project the wire response into a result, reading the Statement fields. */
export function parseCodeAuthorshipResponse(
  data: Record<string, unknown>,
): CodeAuthorshipResult {
  const envelope = asRecord(data.envelope);
  const receipt = asRecord(data.receipt);

  let subjectDigest: string | undefined;
  const subject = envelope.subject;
  if (Array.isArray(subject) && subject.length > 0) {
    const digestObj = asRecord(asRecord(subject[0]).digest);
    subjectDigest = asString(digestObj.sha256);
  }

  const predicate = asRecord(envelope.predicate);

  return {
    envelope,
    receipt,
    kid: asString(data.kid),
    jwksUrl: asString(data.jwks_url),
    serverDigest: asString(data.server_digest),
    digestMatch: Boolean(data.digest_match),
    subjectDigest,
    captureLayer: asString(predicate.capture_layer),
    assetClass: asString(predicate.asset_class),
    advisoryClientDigest: asString(predicate.advisory_client_digest),
    raw: data,
  };
}

/** True when the bound subject digest equals the server-recomputed digest. */
export function subjectMatchesServer(result: CodeAuthorshipResult): boolean {
  if (result.subjectDigest === undefined || result.serverDigest === undefined) {
    return false;
  }
  return result.subjectDigest === bareHex(result.serverDigest);
}

/** Options for submitCodeAuthorship. `changeDigest` is advisory. */
export interface CodeAuthorshipOptions {
  repo: string;
  commitSha: string;
  baseSha?: string;
  changeDigest?: string;
  changeClass?: string;
  author?: string;
  anchor?: string;
}

/**
 * POST an advisory code-authorship record to /v1/code-authorship.
 *
 * `changeDigest` is ADVISORY (sha256 of `git diff base..head`). The server
 * recomputes the authoritative digest and signs it. Requires `init({ apiKey })`
 * with a key holding the `code_authorship:write` scope. Returns the parsed
 * authoritative envelope.
 */
export async function submitCodeAuthorship(
  options: CodeAuthorshipOptions,
): Promise<CodeAuthorshipResult> {
  const body: Record<string, unknown> = {
    repo: options.repo,
    commit_sha: options.commitSha,
  };
  if (options.baseSha) body.base_sha = options.baseSha;
  if (options.changeDigest) body.change_digest = options.changeDigest;
  if (options.changeClass) body.change_class = options.changeClass;
  if (options.author) body.author = options.author;
  if (options.anchor) body.anchor = options.anchor;

  const data = await request<Record<string, unknown>>(
    "POST",
    CODE_AUTHORSHIP_PATH,
    body,
  );
  return parseCodeAuthorshipResponse(data);
}

/** Outcome of the code-authorship envelope check. */
export interface CodeAuthorshipVerification {
  verdict: string;
  /** True only when the capture layer is `github_sha_pull`. */
  authoritative: boolean;
  /** True when the capture layer is a client self-report (observation only). */
  observationOnly: boolean;
  captureLayer?: string;
  subjectDigest?: string;
  reasons: string[];
}

/**
 * Verify the code-authorship structural + capture-layer invariant.
 *
 * Checks the in-toto Statement shape (`_type`, `predicateType`), that
 * `subject[0].digest.sha256` is present, and the capture-layer rule:
 * `github_sha_pull` is authoritative, `in_process_sdk` / `passive_telemetry`
 * are observation only and never authoritative. The cryptographic DSSE
 * signature is verified by the standalone verifier or the hosted /verify. The
 * cloud is the authoritative verifier and this helper is a convenience.
 */
export function verifyCodeAuthorshipEnvelope(
  envelope: unknown,
): CodeAuthorshipVerification {
  const reasons: string[] = [];

  if (envelope === null || typeof envelope !== "object" || Array.isArray(envelope)) {
    return {
      verdict: VERDICT_REJECT,
      authoritative: false,
      observationOnly: false,
      captureLayer: undefined,
      subjectDigest: undefined,
      reasons: ["envelope_not_an_object"],
    };
  }

  const env = envelope as Record<string, unknown>;

  if (env._type !== INTOTO_STATEMENT_TYPE) {
    reasons.push("code_authorship_envelope_not_intoto_statement");
  }
  if (env.predicateType !== CODE_AUTHORSHIP_PREDICATE_TYPE) {
    reasons.push("code_authorship_wrong_predicate_type");
  }

  let subjectDigest: string | undefined;
  const subject = env.subject;
  if (Array.isArray(subject) && subject.length > 0) {
    const digestObj = asRecord(asRecord(subject[0]).digest);
    subjectDigest = asString(digestObj.sha256);
  }
  if (!subjectDigest) {
    reasons.push("code_authorship_missing_subject_digest");
  }

  const predicate = asRecord(env.predicate);
  const captureLayer = asString(predicate.capture_layer);

  const observationOnly = (
    OBSERVATION_ONLY_CAPTURE_LAYERS as readonly string[]
  ).includes(captureLayer ?? "");
  const authoritative = captureLayer === AUTHORITATIVE_CAPTURE_LAYER;

  if (observationOnly) {
    reasons.push("observation_capture_layer_not_authoritative");
  } else if (captureLayer === undefined) {
    reasons.push("code_authorship_missing_capture_layer");
  } else if (!authoritative) {
    reasons.push("code_authorship_capture_layer_not_github_sha_pull");
  }

  return {
    verdict: reasons.length === 0 ? VERDICT_PASS : VERDICT_REJECT,
    authoritative,
    observationOnly,
    captureLayer,
    subjectDigest,
    reasons,
  };
}
