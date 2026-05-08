/**
 * asqav - AI agent governance.
 *
 * Thin TypeScript SDK for asqav.com. All ML-DSA cryptography happens server-side.
 *
 * Quick start:
 *   import { init, Agent } from "asqav";
 *
 *   init({ apiKey: "sk_..." });
 *   const agent = await Agent.create({ name: "my-agent" });
 *   const sig = await agent.sign({ actionType: "api:call", context: { model: "gpt-4" } });
 */

import { createHash, createHmac } from "node:crypto";
import {
  SUPPORTED_ALGORITHMS,
  isSupportedAlgorithm,
  type SupportedAlgorithm,
} from "./algorithms.js";
import { canonicalizeAction } from "./canonicalize.js";
import { _dispatchAfter, _dispatchBefore } from "./hooks.js";
import { canonicalJson } from "./jcs.js";
import { type Mode, resolveMode } from "./mode.js";

/** Allowed values for `receiptType` per the IETF Compliance Receipts profile. */
export const RECEIPT_TYPE_NAMESPACE = [
  "protectmcp:decision",
  "protectmcp:restraint",
  "protectmcp:lifecycle",
] as const;
export type ReceiptType = (typeof RECEIPT_TYPE_NAMESPACE)[number];

/** DORA RTS JC 2024-33 Annex II field 3.23 canonical incident classification.
 *
 * The Joint Committee of the European Supervisory Authorities published
 * the final RTS on classification of major ICT-related incidents on
 * 17 July 2024 (JC 2024-33). Annex II field 3.23 fixes the controlled
 * vocabulary of `incident_class` to exactly six values. Earlier DORA
 * preliminary drafts (and the SDK's previous releases) circulated a
 * 12-value list; the cloud accepts those legacy tokens and normalises
 * them to the canonical six server-side, so this SDK forwards the
 * caller's value verbatim and only rejects tokens that are neither
 * canonical nor a known legacy alias.
 */
export const DORA_INCIDENT_CLASS_NAMESPACE = [
  "cybersecurity_related",
  "process_failure",
  "system_failure",
  "external_event",
  "payment_related",
  "other",
] as const;
export type DoraIncidentClass = (typeof DORA_INCIDENT_CLASS_NAMESPACE)[number];

/** Legacy 12-value vocabulary from pre-final DORA drafts. Mirrored from
 * the cloud's alias dict (cloud PR #194) so the SDK can accept either
 * form without a network roundtrip. The cloud performs the authoritative
 * normalisation; this map is purely for client-side acceptance checks.
 */
export const LEGACY_DORA_ALIASES: Readonly<Record<string, DoraIncidentClass>> =
  Object.freeze({
    malicious_actions: "cybersecurity_related",
    cyberattack: "cybersecurity_related",
    unauthorised_access: "cybersecurity_related",
    process_failure_internal: "process_failure",
    human_error: "process_failure",
    system_failure_internal: "system_failure",
    software_malfunction: "system_failure",
    hardware_failure: "system_failure",
    third_party_failure: "external_event",
    natural_disaster: "external_event",
    payment_fraud: "payment_related",
    unknown: "other",
  });

/** Sandbox state vocabulary per the High-Risk gate. */
export type SandboxState = "enabled" | "disabled" | "unavailable";

/** Risk class controlled vocabulary. */
export type RiskClass = "low" | "medium" | "high" | "unknown";

/** Policy decision vocabulary; deny / rate_limit require `reason`. */
export type PolicyDecision = "permit" | "deny" | "rate_limit";

/** Spec-shape decision vocabulary; legacy `policyDecision` uses "permit". */
export type Decision = "allow" | "deny" | "rate_limit";

/** Map a legacy `policyDecision` token to the spec-shape `decision`. */
export const DECISION_MAP: Readonly<Record<string, Decision>> = Object.freeze({
  permit: "allow",
  allow: "allow",
  deny: "deny",
  rate_limit: "rate_limit",
});

/** Translate `policyDecision` to the spec-shape `decision`. Falls back
 * to `"deny"` so a misconfigured caller never publishes an
 * out-of-vocabulary token to a draft-strict verifier.
 */
export function mapPolicyDecisionToDecision(
  policyDecision: string | null | undefined,
): Decision {
  return DECISION_MAP[(policyDecision ?? "").toLowerCase()] ?? "deny";
}

export { clearHooks, registerAfter, registerBefore } from "./hooks.js";
export type { AfterHook, BeforeHook } from "./hooks.js";
export { canonicalize, canonicalizeAction, hashAction } from "./canonicalize.js";
export { canonicalJson } from "./jcs.js";
export {
  verifyChain,
  deriveChainHash,
  FIRST_RECEIPT_SEED,
  type ChainRecord,
  type ChainStepResult,
  type ChainVerificationResult,
  type VerifyChainOptions,
} from "./replay.js";
export {
  SUPPORTED_ALGORITHMS,
  LOCAL_SIGNING_ALGORITHMS,
  isSupportedAlgorithm,
  isLocalSigningAlgorithm,
  generateKeypair,
  signMessage,
  verifyMessage,
  type SupportedAlgorithm,
  type LocalSigningAlgorithm,
  type LocalKeypair,
} from "./algorithms.js";
export { resolveMode, isAsqavCloudHost, type Mode } from "./mode.js";
export { BudgetTracker, type BudgetCheckResult, type BudgetTrackerOptions } from "./budget.js";

// Compliance Receipts wire-shape projection helpers.
export {
  signatureEnvelopeFromResponse,
  anchorsFromResponse,
  encodeAnchorValue,
  type SignatureEnvelope,
  type AnchorEntry,
  type ProjectableResponse,
} from "./ietfProjection.js";

import {
  signatureEnvelopeFromResponse as _signatureEnvelopeFromResponse,
  anchorsFromResponse as _anchorsFromResponse,
  type SignatureEnvelope,
  type AnchorEntry,
} from "./ietfProjection.js";

/** Return the spec-shape envelope `{alg, kid, sig}` for a
 * SignatureResponse, or undefined for legacy receipts. */
export function signatureEnvelope(
  response: SignatureResponse | null | undefined,
): SignatureEnvelope | undefined {
  return _signatureEnvelopeFromResponse(response as never);
}

/** Return the cloud-supplied `anchors[]` array, or undefined for
 * legacy receipts. */
export function anchors(
  response: SignatureResponse | null | undefined,
): AnchorEntry[] | undefined {
  return _anchorsFromResponse(response as never);
}

const DEFAULT_BASE_URL = "https://api.asqav.com/api/v1";
/** Metadata keys allowed alongside a hash-only request. Mirrors the
 * Python whitelist; the server enforces its own. */
const HASH_ONLY_METADATA_WHITELIST = new Set<string>([
  "agent_id",
  "session_id",
  "action_type",
  "model_name",
  "tool_name",
  "parent_id",
]);

interface Config {
  apiKey: string | null;
  baseUrl: string;
  mode: Mode;
  orgSalt: Uint8Array | null;
}

const config: Config = {
  apiKey: null,
  baseUrl: DEFAULT_BASE_URL,
  mode: "full-payload",
  orgSalt: null,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export class AsqavError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AsqavError";
  }
}

export class AuthenticationError extends AsqavError {
  constructor(message = "Invalid API key") {
    super(message);
    this.name = "AuthenticationError";
  }
}

export class RateLimitError extends AsqavError {
  constructor(message = "Rate limit exceeded") {
    super(message);
    this.name = "RateLimitError";
  }
}

export class APIError extends AsqavError {
  statusCode: number;
  constructor(message: string, statusCode: number) {
    super(message);
    this.name = "APIError";
    this.statusCode = statusCode;
  }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface InitOptions {
  apiKey?: string;
  baseUrl?: string;
  /**
   * Wire mode for ``Agent.sign``. Defaults to ``"auto"``: hash-only for
   * ``*.asqav.com`` cloud URLs, full-payload otherwise. Set to
   * ``"hash-only"`` or ``"full-payload"`` to override. Also reads the
   * ``ASQAV_MODE`` env var when ``"auto"``.
   */
  mode?: "auto" | "hash-only" | "full-payload";
  /**
   * Optional 32-byte per-organization salt. When set, hash-only mode
   * uses HMAC-SHA-256 instead of plain SHA-256.
   */
  orgSalt?: Uint8Array;
}

export interface AgentCreateOptions {
  name: string;
  /** One of `ml-dsa-65` (default), `ml-dsa-44`, `ml-dsa-87`, `ed25519`,
   * or `es256`. Validated client-side; the cloud is the source of
   * truth on which algorithms are honored on the receipt-signing
   * path. */
  algorithm?: SupportedAlgorithm | string;
  capabilities?: string[];
}

export interface UserIntent {
  signature: string;
  public_key: string;
  algorithm: 'ed25519' | 'ecdsa-p256' | 'webauthn' | string;
  key_id?: string;
  signed_message: string;
  signed_at: string;
}

export interface SignOptions {
  actionType: string;
  context?: Record<string, unknown>;
  /** Optional tool name when this action is a tool call. Surfaced in
   * the dashboard graph view. Only the name travels; tool args do not. */
  toolName?: string;
  /** Optional model name when this action is an LLM call (e.g. "gpt-4").
   * Only the name travels; the prompt does not. */
  modelName?: string;
  /** Optional parent action ID for linking child actions to their parent
   * in a workflow. Powers the agent graph view. */
  parentId?: string;
  /** Optional list of agent_ids in the same org expected to countersign
   * this record. Each peer fetches the record and calls
   * ``agent.countersign(signatureId)``. */
  coSigners?: string[];
  /** Optional user-intent envelope. The end user signs a digest of the
   * action and the SDK passes it through to the backend verbatim. */
  userIntent?: UserIntent;
  /** Optional opaque nonce for replay protection. Unique per agent. */
  nonce?: string;
  /** Optional validity window in seconds. Sets `valid_until = signed_at + validSeconds`
   * on the server; verifying the record after expiry returns
   * `verified=false` with `validation_label="signature_expired"`. */
  validSeconds?: number;
  /** Counterparty pin: base64 executor pubkey expected on attestation;
   * other keys -> 403 `executor_key_mismatch`. */
  expectedExecutorPubkeyB64?: string;

  // -------------------------------------------------------------------
  // IETF Compliance Receipts profile fields
  // (draft-marques-asqav-compliance-receipts-00). All optional client-
  // side; the cloud applies the per-field MUST/REQUIRED rules when
  // ``complianceMode=true``. Wire keys are snake_case per the IETF
  // profile (e.g. ``action_ref``).
  // -------------------------------------------------------------------

  /** Emit the receipt under the IETF profile. When true, the cloud uses
   * fail-closed anchoring, raises the retention floor, and writes the
   * v3-20 envelope fields. Defaults to false. */
  complianceMode?: boolean;

  /** `sha256:<hex>` of the canonical Action object. When omitted under
   * `complianceMode=true`, the SDK computes it client-side as
   * `"sha256:" + sha256(canonicalJson(action))` where `action` is
   * `{action_type, context}`. */
  actionRef?: string;

  /** High-Risk gate. Caller-supplied; defaults to `"unavailable"` when
   * not provided and the receipt is High-Risk. */
  sandboxState?: SandboxState;

  /** Logical task iteration id, distinct from `session_id`. REQUIRED
   * for multi-step receipts. */
  iterationId?: string;

  /** Risk classification controlled vocabulary. */
  riskClass?: RiskClass;

  /** Incident class; canonical string or array of strings for
   * multi-regime cases. */
  incidentClass?: string | string[];

  /** Names a legal entity. Resolved server-side from
   * `Organization.legal_entity` when omitted. */
  issuerId?: string;

  /** Digest of the request payload. Accepts the legacy
   * `"sha256:<hex>"` string or the wire object form
   * `{ hash, size?, preview? }`. */
  payloadDigest?: string | { hash: string; size?: number; preview?: string };

  /** Receipt namespace. One of `protectmcp:decision` |
   * `protectmcp:restraint` | `protectmcp:lifecycle`. Defaults to
   * `protectmcp:decision`. Validated client-side. */
  receiptType?: ReceiptType;

  /** Machine-readable code for `policy_decision in {deny, rate_limit}`.
   * Drawn from the IETF rejection-reason vocabulary; the cloud is the
   * source of truth on which codes are accepted. */
  reason?: string;

  /** Policy decision. `permit` (default), `deny`, or `rate_limit`. When
   * not `permit`, `reason` is required. */
  policyDecision?: PolicyDecision;
}

export interface CoSignature {
  agentId: string;
  signature: string;
  signedAt: string;
}

export interface SignatureResponse {
  /** Polymorphic. Base64 string in legacy mode, `{alg, kid, sig}` object
   * form under compliance_mode. Use `signatureEnvelope()` for the dict
   * form. */
  signature: string | SignatureEnvelope;
  signatureId: string;
  actionId: string;
  timestamp: string;
  verificationUrl: string;
  algorithm?: string;
  chainHash?: string;
  /** Agents the original signer expects to countersign this record. */
  requiredCoSigners?: string[];
  /** Signatures appended by countersigning peers, in arrival order. */
  coSignatures?: CoSignature[];
  /** URL pattern (relative) for peers to POST their countersignature to. */
  countersignUrl?: string;
  /** True when the server validated a user_intent envelope on this record. */
  userIntentVerified?: boolean;
  /** True when the cloud emitted this receipt under the IETF profile. */
  complianceMode?: boolean;
  /** IETF §5.7 chain link: `sha256(canonicalJson(predecessor_envelope))`,
   * lowercase 64-hex. First record per agent is `"0".repeat(64)`. */
  previousReceiptHash?: string;
  /** `sha256:<hex>` of the canonical Action object. */
  actionRef?: string;
  /** `sha256:<hex>` of the policy artefact retained for this receipt. */
  policyDigest?: string;
  /** Receipt namespace echoed from the request. */
  receiptType?: string;
  /** Resolved legal entity for this receipt. */
  issuerId?: string;
  /** Legacy `policy_decision` value echoed from the request (one of
   * `"permit" | "deny" | "rate_limit"`). Kept for backward compat with
   * tooling built against the pre-spec implementation. */
  policyDecision?: PolicyDecision;
  /** Spec-shape `decision`: echoed from cloud or mapped from
   * `policyDecision` for older servers. Undefined for non-compliance. */
  decision?: Decision;
  /** Compliance Receipts envelope: the canonical signed dict. None on
   * legacy receipts. */
  payload?: Record<string, unknown>;
  /** Compliance Receipts envelope: the type-discriminated anchors
   * array. None on legacy receipts; `[]` when compliance_mode is on
   * before anchors land. */
  anchors?: AnchorEntry[];
}

export interface SessionResponse {
  sessionId: string;
  agentId: string;
  status: string;
  startedAt: string;
  endedAt?: string;
}

/** Granular sub-checks the cloud returns alongside `verified`. The
 * `validationLabel` string is the dominant failure reason when
 * `verified=false`, e.g. `signature_expired` or `signer_key_changed`.
 *
 * The IETF Compliance Receipts profile sub-axes (`chainValid`,
 * `anchorStatusOts`, `anchorStatusRfc3161`, `signedAtSkewSeconds`,
 * `missingFields`) are populated only when the cloud emits them on a
 * compliance-mode receipt; older receipts and older deployments leave
 * them undefined for back-compat. */
export interface VerificationDetail {
  signerKeyMatch: boolean;
  signatureValid: boolean;
  algorithmMatch: boolean;
  agentActive: boolean;
  validationLabel:
    | "valid"
    | "invalid_signature"
    | "signature_expired"
    | "signer_key_changed"
    | "algorithm_mismatch"
    | "agent_inactive"
    | "agent_unknown"
    | "chain_break"
    | "anchor_invalid"
    | "missing_required_field"
    | "signed_at_skew"
    | string;
  /** Skew vs verifier wall clock; beyond SKEW_BOUND_SECONDS (300s) -> reject. */
  signedAtSkewSeconds?: number;
  /** True when the cloud could rederive the chain hash from the stored
   * `signed_envelope` and the predecessor matches. None on legacy
   * (pre-IETF) records. */
  chainValid?: boolean;
  /** Tri-state OTS status. `pending` means the proof is still inside
   * the upgrade window; `invalid` means corruption or stale; `valid`
   * mirrors the Bitcoin block-confirmed case. */
  anchorStatusOts?: "valid" | "pending" | "invalid";
  /** RFC 3161 anchor outcome. Deterministic at issuance; never
   * `pending`. */
  anchorStatusRfc3161?: "valid" | "pending" | "invalid";
  /** REQUIRED-fields presence check. When the record is
   * `compliance_mode=true` and any spec-mandated field has been NULLed
   * post-issuance, surfaces the missing column names. Empty list /
   * undefined means every required field is present. */
  missingFields?: string[];
}

export interface VerificationResponse {
  signatureId: string;
  agentId: string;
  agentName?: string;
  actionId?: string;
  actionType?: string;
  payload?: unknown;
  signature?: string;
  algorithm: string;
  signedAt: string;
  verified: boolean;
  verificationUrl?: string;
  verificationDetail?: VerificationDetail;
}

export interface AppliedAttestationOptions {
  signatureId: string;
  appliedAt: string;
  outcome: "success" | "fail" | "partial";
  errorCode?: string | null;
  executorPubkeyB64: string;
  executorAlgorithm: string;
  signatureB64: string;
}

export interface AppliedAttestationResponse {
  signatureId: string;
  appliedAttestation: {
    appliedAt: string;
    outcome: string;
    errorCode: string | null;
    executorPubkeyB64: string;
    executorAlgorithm: string;
    signatureB64: string;
  };
}

export interface ListRejectedAttemptsOptions {
  failureReason?: string;
  agentId?: string;
  hours?: number;
  limit?: number;
  offset?: number;
}

export interface RejectedAttempt {
  id: number;
  createdAt: string;
  organizationId: string | null;
  agentId: string | null;
  endpoint: string;
  actionType: string | null;
  failureReason: string;
  statusCode: number;
  suppliedSignerKey: string | null;
  suppliedSignaturePrefix: string | null;
  ip: string | null;
  userAgent: string | null;
  requestId: string | null;
  detail: string | null;
}

export interface RejectedAttemptList {
  items: RejectedAttempt[];
  total: number;
  limit: number;
  offset: number;
}

export interface ListSessionsOptions {
  agentId?: string;
  status?: string;
  limit?: number;
}

export interface ExportAuditOptions {
  startDate?: string;
  endDate?: string;
  agentId?: string;
}

export interface PreflightResult {
  cleared: boolean;
  agentActive: boolean;
  policyAllowed: boolean;
  reasons: string[];
  explanation: string;
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

export function init(options: InitOptions = {}): void {
  const apiKey = options.apiKey ?? process.env.ASQAV_API_KEY ?? null;
  if (!apiKey) {
    throw new AuthenticationError(
      "API key required. Set ASQAV_API_KEY or pass apiKey to init(). Get yours at asqav.com",
    );
  }
  config.apiKey = apiKey;
  config.baseUrl = options.baseUrl ?? config.baseUrl ?? DEFAULT_BASE_URL;
  config.mode = resolveMode(
    config.baseUrl,
    process.env.ASQAV_MODE ?? null,
    options.mode ?? "auto",
  );
  config.orgSalt = options.orgSalt ?? null;
}

function ensureInitialized(): void {
  if (!config.apiKey) {
    throw new AuthenticationError("Call init() first. Get your API key at asqav.com");
  }
}

// ---------------------------------------------------------------------------
// HTTP request helper with retry
// ---------------------------------------------------------------------------

const RETRY_STATUSES = new Set([429, 500, 502, 503, 504]);
const MAX_ATTEMPTS = 3;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function request<T = unknown>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  ensureInitialized();

  const url = config.baseUrl.replace(/\/+$/, "") + (path.startsWith("/") ? path : `/${path}`);
  const headers: Record<string, string> = {
    "X-API-Key": config.apiKey as string,
    "Content-Type": "application/json",
    Accept: "application/json",
  };

  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    let response: Response;
    try {
      response = await fetch(url, {
        method,
        headers,
        body: body !== undefined ? JSON.stringify(body) : undefined,
      });
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (attempt < MAX_ATTEMPTS) {
        await sleep(2 ** (attempt - 1) * 250);
        continue;
      }
      throw new APIError(`Network error: ${lastError.message}`, 0);
    }

    if (response.status === 401) {
      throw new AuthenticationError("Invalid API key");
    }
    if (response.status === 429) {
      if (attempt < MAX_ATTEMPTS) {
        await sleep(2 ** (attempt - 1) * 500);
        continue;
      }
      throw new RateLimitError("Rate limit exceeded. Upgrade at asqav.com/pricing");
    }
    if (RETRY_STATUSES.has(response.status) && attempt < MAX_ATTEMPTS) {
      await sleep(2 ** (attempt - 1) * 500);
      continue;
    }
    if (response.status >= 400) {
      let detail: string = response.statusText || "Unknown error";
      try {
        const json = (await response.json()) as { error?: string; detail?: string };
        detail = json.error ?? json.detail ?? detail;
      } catch {
        try {
          detail = await response.text();
        } catch {
          // ignore
        }
      }
      throw new APIError(detail, response.status);
    }

    if (response.status === 204) {
      return undefined as T;
    }

    try {
      return (await response.json()) as T;
    } catch {
      return undefined as T;
    }
  }

  throw new APIError(`Request failed after ${MAX_ATTEMPTS} attempts`, 0);
}

// ---------------------------------------------------------------------------
// Sign body builder (mode-aware)
// ---------------------------------------------------------------------------

interface BuildSignBodyArgs {
  actionType: string;
  context: Record<string, unknown>;
  sessionId: string | null;
  agentId: string;
  userIntent?: UserIntent;
}

async function buildSignBody(args: BuildSignBodyArgs): Promise<Record<string, unknown>> {
  if (config.mode === "hash-only") {
    const canonical = canonicalizeAction(args.actionType, args.context);
    const payloadSize = canonical.byteLength;
    const salt = config.orgSalt ?? undefined;
    const hex = salt
      ? createHmac("sha256", Buffer.from(salt)).update(canonical).digest("hex")
      : createHash("sha256").update(canonical).digest("hex");
    const digest = `sha256:${hex}`;
    const metadata: Record<string, unknown> = {
      agent_id: args.agentId,
      action_type: args.actionType,
    };
    if (args.sessionId !== null) metadata.session_id = args.sessionId;
    // Opt-in via underscored sentinel keys; nothing else from context travels.
    const ctx = args.context as Record<string, unknown>;
    if (typeof ctx._model_name === "string") metadata.model_name = ctx._model_name;
    if (typeof ctx._tool_name === "string") metadata.tool_name = ctx._tool_name;
    if (typeof ctx._parent_id === "string") metadata.parent_id = ctx._parent_id;
    for (const k of Object.keys(metadata)) {
      if (!HASH_ONLY_METADATA_WHITELIST.has(k)) delete metadata[k];
    }
    const hashBody: Record<string, unknown> = {
      action_type: args.actionType,
      hash: digest,
      hash_algo: "sha256",
      payload_size: payloadSize,
      metadata,
      session_id: args.sessionId,
    };
    if (args.userIntent !== undefined) hashBody.user_intent = args.userIntent;
    return hashBody;
  }
  const fullBody: Record<string, unknown> = {
    action_type: args.actionType,
    context: args.context,
    session_id: args.sessionId,
  };
  if (args.userIntent !== undefined) fullBody.user_intent = args.userIntent;
  return fullBody;
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

interface AgentData {
  agent_id: string;
  name: string;
  public_key: string;
  key_id: string;
  algorithm: string;
  capabilities: string[];
  created_at: string | number;
}

export class Agent {
  readonly agentId: string;
  readonly name: string;
  readonly publicKey: string;
  readonly keyId: string;
  readonly algorithm: string;
  readonly capabilities: string[];
  readonly createdAt: string | number;
  private sessionId: string | null = null;

  private constructor(data: AgentData) {
    this.agentId = data.agent_id;
    this.name = data.name;
    this.publicKey = data.public_key;
    this.keyId = data.key_id;
    this.algorithm = data.algorithm;
    this.capabilities = data.capabilities ?? [];
    this.createdAt = data.created_at;
  }

  /** Internal: rebuild an Agent from a server payload. */
  static attach(data: AgentData): Agent {
    return new Agent(data);
  }

  static async create(options: AgentCreateOptions): Promise<Agent> {
    const algorithm = options.algorithm ?? "ml-dsa-65";
    // Cloud accepts ml-dsa-{44,65,87}, ed25519, es256.
    if (!isSupportedAlgorithm(algorithm)) {
      throw new AsqavError(
        `unsupported_algorithm: '${algorithm}'. Use one of: ${SUPPORTED_ALGORITHMS.join(", ")}`,
      );
    }
    const data = await request<AgentData>("POST", "/agents/create", {
      name: options.name,
      algorithm,
      capabilities: options.capabilities ?? [],
    });
    return new Agent(data);
  }

  static async get(agentId: string): Promise<Agent> {
    const data = await request<AgentData>("GET", `/agents/${agentId}`);
    return new Agent(data);
  }

  async sign(options: SignOptions): Promise<SignatureResponse> {
    let initialContext = options.context ?? {};
    // Surface explicit kwargs into the underscored sentinel keys so the
    // hash-only metadata builder picks them up. Mirrors the Python SDK.
    if (
      options.toolName !== undefined
      || options.modelName !== undefined
      || options.parentId !== undefined
    ) {
      initialContext = { ...initialContext };
      if (options.toolName !== undefined) initialContext._tool_name = options.toolName;
      if (options.modelName !== undefined) initialContext._model_name = options.modelName;
      if (options.parentId !== undefined) initialContext._parent_id = options.parentId;
    }
    const finalContext = _dispatchBefore(options.actionType, initialContext);

    const complianceMode = options.complianceMode === true;

    // Validate the IETF receipt namespace client-side so callers fail
    // fast instead of waiting for the server's 422. The cloud is still
    // the source of truth; we mirror the same vocabulary.
    if (options.receiptType !== undefined
      && !(RECEIPT_TYPE_NAMESPACE as readonly string[]).includes(options.receiptType)) {
      throw new AsqavError(
        `invalid_receipt_type: must be one of ${RECEIPT_TYPE_NAMESPACE.join(", ")}`,
      );
    }
    if (options.policyDecision !== undefined
      && options.policyDecision !== "permit"
      && !options.reason) {
      throw new AsqavError(
        "missing_reason: policy_decision=deny|rate_limit requires a `reason` code",
      );
    }
    // DORA RTS JC 2024-33 Annex II field 3.23 vocabulary check (six
    // canonical values, plus the legacy 12-value alias set the cloud
    // still accepts). Reject anything else before the HTTP roundtrip.
    if (options.incidentClass !== undefined && options.incidentClass !== "") {
      const incidentValues = Array.isArray(options.incidentClass)
        ? options.incidentClass
        : [options.incidentClass];
      for (const ic of incidentValues) {
        if (
          !(DORA_INCIDENT_CLASS_NAMESPACE as readonly string[]).includes(ic)
          && !(ic in LEGACY_DORA_ALIASES)
        ) {
          throw new AsqavError(
            `invalid_incident_class: '${ic}' must be one of ${DORA_INCIDENT_CLASS_NAMESPACE.join(", ")} `
              + `or a known legacy alias ${Object.keys(LEGACY_DORA_ALIASES).sort().join(", ")}`,
          );
        }
      }
    }

    // Compute action_ref client-side when the caller is in compliance
    // mode and did not supply one. Per spec:
    //   action_ref = "sha256:" + sha256(canonicalJson(action))
    // where `action = {action_type, context}` matches the cloud-side
    // shape used by `hash_action`.
    let actionRef = options.actionRef;
    if (complianceMode && actionRef === undefined) {
      const action = { action_type: options.actionType, context: finalContext };
      const hex = createHash("sha256")
        .update(canonicalJson(action))
        .digest("hex");
      actionRef = `sha256:${hex}`;
    }

    const body = await buildSignBody({
      actionType: options.actionType,
      context: finalContext,
      sessionId: this.sessionId,
      agentId: this.agentId,
      userIntent: options.userIntent,
    });
    if (options.coSigners && options.coSigners.length > 0) {
      body.co_signers = [...options.coSigners];
    }
    if (options.nonce !== undefined) {
      body.nonce = options.nonce;
    }
    if (options.validSeconds !== undefined) {
      body.valid_seconds = options.validSeconds;
    }
    if (options.expectedExecutorPubkeyB64 !== undefined) {
      body.expected_executor_pubkey_b64 = options.expectedExecutorPubkeyB64;
    }

    // IETF Compliance Receipts profile fields. Wire-format names are
    // snake_case per draft-marques-asqav-compliance-receipts-00.
    // `compliance_mode` always travels (default false) so the cloud
    // knows whether to apply the profile rules. The other fields ride
    // along when present; the cloud applies the per-field MUST checks.
    if (complianceMode) body.compliance_mode = true;
    if (actionRef !== undefined) body.action_ref = actionRef;
    if (options.sandboxState !== undefined) body.sandbox_state = options.sandboxState;
    if (options.iterationId !== undefined) body.iteration_id = options.iterationId;
    if (options.riskClass !== undefined) body.risk_class = options.riskClass;
    if (options.incidentClass !== undefined) body.incident_class = options.incidentClass;
    if (options.issuerId !== undefined) body.issuer_id = options.issuerId;
    if (options.payloadDigest !== undefined) body.payload_digest = options.payloadDigest;
    if (options.receiptType !== undefined) body.receipt_type = options.receiptType;
    if (options.reason !== undefined) body.reason = options.reason;
    if (options.policyDecision !== undefined) {
      body.policy_decision = options.policyDecision;
      // Send spec-shape `decision` under compliance_mode; older clouds ignore.
      if (complianceMode) {
        body.decision = mapPolicyDecisionToDecision(options.policyDecision);
      }
    }

    const data = await request<{
      signature: string;
      signature_id: string;
      action_id: string;
      timestamp: string;
      verification_url: string;
      algorithm?: string;
      chain_hash?: string;
      record_hash?: string;
      required_co_signers?: string[];
      co_signatures?: Array<{ agent_id: string; signature: string; signed_at: string }>;
      countersign_url?: string;
      user_intent_verified?: boolean;
      compliance_mode?: boolean;
      previousReceiptHash?: string;
      previous_receipt_hash?: string;
      action_ref?: string;
      policy_digest?: string;
      receipt_type?: string;
      issuer_id?: string;
      policy_decision?: string;
      decision?: string;
      payload?: Record<string, unknown>;
      anchors?: AnchorEntry[];
    }>("POST", `/agents/${this.agentId}/sign`, body);

    const response: SignatureResponse = {
      signature: data.signature,
      signatureId: data.signature_id,
      actionId: data.action_id,
      timestamp: data.timestamp,
      verificationUrl: data.verification_url,
      algorithm: data.algorithm,
      chainHash: data.chain_hash ?? data.record_hash,
      requiredCoSigners: data.required_co_signers,
      coSignatures: data.co_signatures?.map((s) => ({
        agentId: s.agent_id,
        signature: s.signature,
        signedAt: s.signed_at,
      })),
      countersignUrl: data.countersign_url,
      userIntentVerified: data.user_intent_verified,
      complianceMode: data.compliance_mode,
      // Accept either camelCase (per the IETF profile JSON wire) or
      // snake_case (legacy alias for one release).
      previousReceiptHash: data.previousReceiptHash ?? data.previous_receipt_hash,
      actionRef: data.action_ref,
      policyDigest: data.policy_digest,
      receiptType: data.receipt_type,
      issuerId: data.issuer_id,
      policyDecision: data.policy_decision as PolicyDecision | undefined,
      // Prefer cloud-emitted `decision`; map locally for older clouds.
      decision: (data.decision as Decision | undefined) ?? (
        data.compliance_mode
          ? mapPolicyDecisionToDecision(data.policy_decision)
          : undefined
      ),
      payload: data.payload,
      anchors: data.anchors,
    };

    _dispatchAfter(options.actionType, response);
    return response;
  }

  async countersign(signatureId: string): Promise<SignatureResponse> {
    const data = await request<{
      signature: string;
      signature_id: string;
      action_id: string;
      timestamp: string;
      verification_url: string;
      algorithm?: string;
      chain_hash?: string;
      record_hash?: string;
      required_co_signers?: string[];
      co_signatures?: Array<{ agent_id: string; signature: string; signed_at: string }>;
      countersign_url?: string;
      user_intent_verified?: boolean;
    }>("POST", `/agents/${this.agentId}/countersign/${signatureId}`, {});

    return {
      signature: data.signature,
      signatureId: data.signature_id,
      actionId: data.action_id,
      timestamp: data.timestamp,
      verificationUrl: data.verification_url,
      algorithm: data.algorithm,
      chainHash: data.chain_hash ?? data.record_hash,
      requiredCoSigners: data.required_co_signers,
      coSignatures: data.co_signatures?.map((s) => ({
        agentId: s.agent_id,
        signature: s.signature,
        signedAt: s.signed_at,
      })),
      countersignUrl: data.countersign_url,
      userIntentVerified: data.user_intent_verified,
    };
  }

  async startSession(): Promise<SessionResponse> {
    const data = await request<{
      session_id: string;
      agent_id: string;
      status: string;
      started_at: string;
    }>("POST", "/sessions/", { agent_id: this.agentId });
    this.sessionId = data.session_id;
    return {
      sessionId: data.session_id,
      agentId: data.agent_id,
      status: data.status,
      startedAt: data.started_at,
    };
  }

  async endSession(options: { status?: string } = {}): Promise<SessionResponse> {
    if (!this.sessionId) {
      throw new AsqavError("No active session");
    }
    const sessionId = this.sessionId;
    const data = await request<{
      agent_id: string;
      status: string;
      started_at: string;
      ended_at?: string;
    }>("PATCH", `/sessions/${sessionId}`, { status: options.status ?? "completed" });
    this.sessionId = null;
    return {
      sessionId,
      agentId: data.agent_id,
      status: data.status,
      startedAt: data.started_at,
      endedAt: data.ended_at,
    };
  }

  async revoke(options: { reason?: string } = {}): Promise<void> {
    await request("POST", `/agents/${this.agentId}/revoke`, {
      reason: options.reason ?? "manual",
    });
  }

  /**
   * Pre-flight check combining revocation/suspension status and policy.
   * Fail-open: if a sub-check errors, it is recorded in `reasons` but
   * does not block. Mirrors the Python `Agent.preflight`.
   */
  async preflight(actionType: string): Promise<PreflightResult> {
    let agentActive = true;
    let policyAllowed = true;
    const reasons: string[] = [];

    try {
      const status = await request<{ revoked?: boolean; suspended?: boolean; suspended_reason?: string }>(
        "GET",
        `/agents/${this.agentId}/status`,
      );
      if (status.revoked) {
        agentActive = false;
        reasons.push("agent is revoked");
      }
      if (status.suspended) {
        agentActive = false;
        const suffix = status.suspended_reason ? ` (${status.suspended_reason})` : "";
        reasons.push(`agent is suspended${suffix}`);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      reasons.push(`status check failed (${msg}) - skipped`);
    }

    try {
      const policies = await request<Array<{
        is_active?: boolean;
        action_pattern?: string;
        action?: string;
        name?: string;
      }>>("GET", "/policies");
      const list = Array.isArray(policies) ? policies : [];
      for (const p of list) {
        if (!p.is_active) continue;
        const pattern = p.action_pattern ?? "";
        const matches = pattern === "*" || actionType.startsWith(pattern.replace(/\*+$/, ""));
        if (matches && (p.action === "block" || p.action === "block_and_alert")) {
          policyAllowed = false;
          reasons.push(`blocked by policy: ${p.name ?? "unknown"}`);
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      reasons.push(`policy check failed (${msg}) - skipped`);
    }

    const cleared = agentActive && policyAllowed;
    let explanation: string;
    if (cleared) {
      explanation = "Allowed: agent is active and action is permitted by policy";
    } else if (!agentActive && reasons.includes("agent is revoked")) {
      explanation = "Blocked: agent has been revoked";
    } else if (!agentActive && reasons.some((r) => r.startsWith("agent is suspended"))) {
      const suspendReason = reasons.find((r) => r.startsWith("agent is suspended")) ?? "";
      explanation = `Blocked: ${suspendReason}`;
    } else if (!policyAllowed) {
      explanation = "Blocked: action not permitted by current policy rules";
    } else {
      explanation = reasons.length ? `Blocked: ${reasons.join("; ")}` : "Blocked";
    }

    return { cleared, agentActive, policyAllowed, reasons, explanation };
  }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

export async function verifySignature(signatureId: string): Promise<VerificationResponse> {
  const data = await request<{
    signature_id: string;
    agent_id: string;
    agent_name?: string;
    action_id?: string;
    action_type?: string;
    payload?: unknown;
    signature?: string;
    algorithm: string;
    signed_at: string;
    verified: boolean;
    verification_url?: string;
    verification_detail?: {
      signer_key_match: boolean;
      signature_valid: boolean;
      algorithm_match: boolean;
      agent_active: boolean;
      validation_label: string;
      signed_at_skew_seconds?: number;
      chain_valid?: boolean;
      anchor_status_ots?: "valid" | "pending" | "invalid";
      anchor_status_rfc3161?: "valid" | "pending" | "invalid";
      missing_fields?: string[];
    };
  }>("GET", `/verify/${signatureId}`);

  return {
    signatureId: data.signature_id,
    agentId: data.agent_id,
    agentName: data.agent_name,
    actionId: data.action_id,
    actionType: data.action_type,
    payload: data.payload,
    signature: data.signature,
    algorithm: data.algorithm,
    signedAt: data.signed_at,
    verified: data.verified,
    verificationUrl: data.verification_url,
    verificationDetail: data.verification_detail
      ? {
          signerKeyMatch: data.verification_detail.signer_key_match,
          signatureValid: data.verification_detail.signature_valid,
          algorithmMatch: data.verification_detail.algorithm_match,
          agentActive: data.verification_detail.agent_active,
          validationLabel: data.verification_detail.validation_label,
          signedAtSkewSeconds: data.verification_detail.signed_at_skew_seconds,
          chainValid: data.verification_detail.chain_valid,
          anchorStatusOts: data.verification_detail.anchor_status_ots,
          anchorStatusRfc3161: data.verification_detail.anchor_status_rfc3161,
          missingFields: data.verification_detail.missing_fields,
        }
      : undefined,
  };
}

/** Post the executor side of an action.
 *
 * The executor signs the canonical bytes
 * `{applied_at, error_code, outcome, signature_id}` (sorted-keys JSON,
 * no whitespace) with its identity key and posts the base64 of pubkey +
 * signature back. If the original signer pinned an expected executor
 * public key on `agent.sign({ expectedExecutorPubkeyB64 })`, any
 * attestation from a different key is rejected with 403.
 */
export async function postAppliedAttestation(
  options: AppliedAttestationOptions,
): Promise<AppliedAttestationResponse> {
  const data = await request<{
    signature_id: string;
    applied_attestation: {
      applied_at: string;
      outcome: string;
      error_code: string | null;
      executor_pubkey_b64: string;
      executor_algorithm: string;
      signature_b64: string;
    };
  }>("POST", `/signatures/${options.signatureId}/applied-attestation`, {
    applied_at: options.appliedAt,
    outcome: options.outcome,
    error_code: options.errorCode ?? null,
    executor_pubkey_b64: options.executorPubkeyB64,
    executor_algorithm: options.executorAlgorithm,
    signature_b64: options.signatureB64,
  });
  return {
    signatureId: data.signature_id,
    appliedAttestation: {
      appliedAt: data.applied_attestation.applied_at,
      outcome: data.applied_attestation.outcome,
      errorCode: data.applied_attestation.error_code,
      executorPubkeyB64: data.applied_attestation.executor_pubkey_b64,
      executorAlgorithm: data.applied_attestation.executor_algorithm,
      signatureB64: data.applied_attestation.signature_b64,
    },
  };
}

/** List rejected sign / verify / replay attempts for the caller's
 * org. Pro+ tier. Public verify rejections (org NULL) are filtered out
 * by design - admins query those separately via maintenance. */
export async function listRejectedAttempts(
  options: ListRejectedAttemptsOptions = {},
): Promise<RejectedAttemptList> {
  const params = new URLSearchParams();
  if (options.failureReason) params.set("failure_reason", options.failureReason);
  if (options.agentId) params.set("agent_id", options.agentId);
  if (options.hours !== undefined) params.set("hours", String(options.hours));
  if (options.limit !== undefined) params.set("limit", String(options.limit));
  if (options.offset !== undefined) params.set("offset", String(options.offset));
  const query = params.toString();
  const path = query
    ? `/observability/rejected-attempts?${query}`
    : "/observability/rejected-attempts";

  const data = await request<{
    items: Array<{
      id: number;
      created_at: string;
      organization_id: string | null;
      agent_id: string | null;
      endpoint: string;
      action_type: string | null;
      failure_reason: string;
      status_code: number;
      supplied_signer_key: string | null;
      supplied_signature_prefix: string | null;
      ip: string | null;
      user_agent: string | null;
      request_id: string | null;
      detail: string | null;
    }>;
    total: number;
    limit: number;
    offset: number;
  }>("GET", path);

  return {
    total: data.total,
    limit: data.limit,
    offset: data.offset,
    items: data.items.map((r) => ({
      id: r.id,
      createdAt: r.created_at,
      organizationId: r.organization_id,
      agentId: r.agent_id,
      endpoint: r.endpoint,
      actionType: r.action_type,
      failureReason: r.failure_reason,
      statusCode: r.status_code,
      suppliedSignerKey: r.supplied_signer_key,
      suppliedSignaturePrefix: r.supplied_signature_prefix,
      ip: r.ip,
      userAgent: r.user_agent,
      requestId: r.request_id,
      detail: r.detail,
    })),
  };
}

export async function listSessions(
  options: ListSessionsOptions = {},
): Promise<SessionResponse[]> {
  const params = new URLSearchParams();
  if (options.limit !== undefined) params.set("limit", String(options.limit));
  if (options.status) params.set("status", options.status);
  if (options.agentId) params.set("agent_id", options.agentId);

  const query = params.toString();
  const path = query ? `/sessions/?${query}` : "/sessions/";
  const data = await request<
    Array<{
      session_id: string;
      agent_id: string;
      status: string;
      started_at: string;
      ended_at?: string;
    }>
  >("GET", path);

  return (data ?? []).map((s) => ({
    sessionId: s.session_id,
    agentId: s.agent_id,
    status: s.status,
    startedAt: s.started_at,
    endedAt: s.ended_at,
  }));
}

export async function exportAuditJson(
  options: ExportAuditOptions = {},
): Promise<unknown> {
  const params = new URLSearchParams();
  if (options.startDate) params.set("start_date", options.startDate);
  if (options.endDate) params.set("end_date", options.endDate);
  if (options.agentId) params.set("agent_id", options.agentId);
  const query = params.toString();
  const path = query ? `/export/json?${query}` : "/export/json";
  return request<unknown>("GET", path);
}

// ---------------------------------------------------------------------------
// Internal config exposure for tests only
// ---------------------------------------------------------------------------

/** @internal - reset module state. Used in tests. */
export function _resetForTests(): void {
  config.apiKey = null;
  config.baseUrl = DEFAULT_BASE_URL;
  config.mode = "full-payload";
  config.orgSalt = null;
}

/** @internal - inspect or override mode in tests. */
export function _setModeForTests(mode: Mode, orgSalt?: Uint8Array | null): void {
  config.mode = mode;
  config.orgSalt = orgSalt ?? null;
}

/** @internal - read current mode in tests. */
export function _getModeForTests(): Mode {
  return config.mode;
}
