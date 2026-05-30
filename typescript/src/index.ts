/**
 * asqav - AI agent governance.
 *
 * Thin TypeScript SDK for asqav.com. All ML-DSA cryptography happens server-side.
 *
 * Quick start:
 *   import { init, Agent } from "@asqav/sdk";
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
  "protectmcp:lifecycle:configuration_change",
  "protectmcp:acknowledgment",
  "protectmcp:observation",
  // observation receipts with a result_digest use the :result_bound suffix for auditor indexing.
  "protectmcp:observation:result_bound",
] as const;
export type ReceiptType = (typeof RECEIPT_TYPE_NAMESPACE)[number];

/** Receipt type emitted by an acknowledging agent (B) over an originating
 * receipt; the binding object travels on B's signed payload. */
export const ACKNOWLEDGMENT_RECEIPT_TYPE = "protectmcp:acknowledgment" as const;

/** DORA RTS JC 2024-33 Annex II field 3.23 canonical incident classification.
 *
 * The Joint Committee of the European Supervisory Authorities published
 * the final RTS on classification of major ICT-related incidents on
 * 17 July 2024 (JC 2024-33). Annex II field 3.23 fixes the controlled
 * vocabulary of `incident_class` to exactly six values. The cloud
 * accepts only these canonical tokens; this SDK forwards the caller's
 * value verbatim and rejects anything outside the canonical set.
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

/** HIPAA Security Rule canonical incident classification.
 *
 * HIPAA 45 CFR 164.304 is one of the canonical source regimes for
 * `incident_class` alongside DORA, NYDFS, and CIRCIA. A Covered Entity
 * SHOULD be able to emit a HIPAA token without falling back to a DORA
 * category.
 */
export const HIPAA_INCIDENT_CLASS_NAMESPACE = [
  "hipaa_security_incident",
] as const;
export type HipaaIncidentClass = (typeof HIPAA_INCIDENT_CLASS_NAMESPACE)[number];

/** Union of every canonical incident_class token the SDK accepts. The
 * cloud's `core/incident_vocabulary.py` is authoritative; this mirror
 * spares callers a network round-trip on obvious mistakes.
 */
export const INCIDENT_CLASS_NAMESPACE = [
  ...DORA_INCIDENT_CLASS_NAMESPACE,
  ...HIPAA_INCIDENT_CLASS_NAMESPACE,
] as const;
export type IncidentClass = (typeof INCIDENT_CLASS_NAMESPACE)[number];

/** Sandbox state vocabulary per the High-Risk gate. Closes GAP-E3 on the
 * TypeScript SDK: the cloud already enforces this enum, the SDK now
 * refuses out-of-vocabulary values before the HTTP roundtrip. */
export const SANDBOX_STATE_NAMESPACE = ["enabled", "disabled", "unavailable"] as const;
export type SandboxState = (typeof SANDBOX_STATE_NAMESPACE)[number];

/** Producer-side `capture_topology` vocabulary; mirrors the cloud
 * SignRequest Literal and the IETF -04 capture-topologies appendix. */
export const CAPTURE_TOPOLOGY_NAMESPACE = [
  "in_process_sdk",
  "network_proxy",
  "browser_extension",
  "ebpf_observer",
  "mcp_proxy",
  "passive_telemetry",
] as const;
export type CaptureTopology = (typeof CAPTURE_TOPOLOGY_NAMESPACE)[number];

/** Durable-anchoring witnesses Asqav ships. The cloud `witness_policy`
 * extension accepts only this set; `rekor` is explicitly rejected because
 * it is not a shipped witness. Mirrors the live governance.json contract. */
export const WITNESS_NAMESPACE = ["rfc3161", "opentimestamps"] as const;
export type Witness = (typeof WITNESS_NAMESPACE)[number];

/** Caller-declared N-of-M durable-anchoring quorum. `required` must be in
 * `[1, witnesses.length]`; `witnesses` must be a non-empty subset of the
 * two shipped witnesses. `rekor` is rejected. */
export interface WitnessPolicy {
  required: number;
  witnesses: Witness[];
}

/** Risk class controlled vocabulary. */
export type RiskClass = "low" | "medium" | "high" | "unknown";

/** Policy decision vocabulary; deny / rate_limit require `reason`. `none`
 * is the lifecycle opt-out gated to `protectmcp:lifecycle` receipts. */
export type PolicyDecision = "permit" | "deny" | "rate_limit" | "none";

/** Spec-shape decision vocabulary; `observation` is reserved to
 * lifecycle receipts emitted under `policyDecision="none"` (IETF -04). */
export type Decision = "allow" | "deny" | "rate_limit" | "observation";

/** Map an original `policyDecision` token to the spec-shape `decision`.
 * The `none -> observation` row is gated to lifecycle receipts by the
 * no-false-attestation rule on the cloud side. */
export const DECISION_MAP: Readonly<Record<string, Decision>> = Object.freeze({
  permit: "allow",
  allow: "allow",
  deny: "deny",
  rate_limit: "rate_limit",
  none: "observation",
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
export {
  canonicalize,
  canonicalizeAction,
  canonicalizeToolArgs,
  hashAction,
} from "./canonicalize.js";
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

// Counterparty acknowledgment binding (IETF -04 counterparty_binding extension).
export {
  computeCounterpartyBinding,
  computeEnvelopeHash,
  verifyCounterpartyBinding,
  type CounterpartyBinding,
  type CounterpartyBindingVerification,
  type ComputeCounterpartyBindingOptions,
} from "./counterparty.js";

import {
  signatureEnvelopeFromResponse as _signatureEnvelopeFromResponse,
  anchorsFromResponse as _anchorsFromResponse,
  type SignatureEnvelope,
  type AnchorEntry,
} from "./ietfProjection.js";

/** Return the spec-shape envelope `{alg, kid, sig}` for a
 * SignatureResponse, or undefined for non-compliance receipts. */
export function signatureEnvelope(
  response: SignatureResponse | null | undefined,
): SignatureEnvelope | undefined {
  return _signatureEnvelopeFromResponse(response as never);
}

/** Return the cloud-supplied `anchors[]` array, or undefined for
 * non-compliance receipts. */
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

// === Errors ===

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

// === Public types ===

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
  /** Optional resolvable man_<token> mandate this sign is authorised under.
   * The cloud re-verifies the issuer signature, scope, server-clock window,
   * and revocation, then builds the signed authorized_under_mandate
   * attestation. The attestation is never a request input. */
  mandateId?: string;
  /** Optional user-intent envelope. The end user signs a digest of the
   * action and the SDK passes it through to the backend verbatim. */
  userIntent?: UserIntent;
  /** Optional opaque nonce for replay protection. Unique per agent. */
  nonce?: string;
  /** Optional validity window in seconds. Sets `valid_until = signed_at + validSeconds`
   * on the server; verifying the record after expiry returns
   * `verified=false` with `validation_label="signature_expired"`.
   * Mutually exclusive with `expiresAt` (rule 10). Pass exactly one of
   * the two: `validSeconds` for "expire N seconds after signing", or
   * `expiresAt` for an explicit horizon. Passing both throws
   * `expiry_collision_guard`. Passing neither falls back to the
   * server-side default (`valid_seconds=86400`). */
  validSeconds?: number;
  /** Counterparty pin: base64 executor pubkey expected on attestation;
   * other keys -> 403 `executor_key_mismatch`. */
  expectedExecutorPubkeyB64?: string;

  // === IETF Compliance Receipts profile fields ===

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

  /** Digest of the request payload. Accepts the
   * `"sha256:<hex>"` string or the wire object form
   * `{ hash, size?, preview? }`. */
  payloadDigest?: string | { hash: string; size?: number; preview?: string };

  /** Receipt namespace. One of `protectmcp:decision` |
   * `protectmcp:restraint` | `protectmcp:lifecycle` |
   * `protectmcp:acknowledgment` | `protectmcp:observation`. Defaults to
   * `protectmcp:decision`. Validated client-side. */
  receiptType?: ReceiptType;

  /** Machine-readable code for `policy_decision in {deny, rate_limit}`.
   * Drawn from the IETF rejection-reason vocabulary; the cloud is the
   * source of truth on which codes are accepted. */
  reason?: string;

  /** Policy decision. `permit` (default), `deny`, `rate_limit`, or `none`.
   * `none` is the lifecycle opt-out and requires `receiptType` in the
   * `protectmcp:lifecycle` namespace. When `deny` or `rate_limit`,
   * `reason` is required. */
  policyDecision?: PolicyDecision;

  /** Producer-side topology. One of `in_process_sdk`, `network_proxy`,
   * `browser_extension`, `ebpf_observer`, `mcp_proxy`, `passive_telemetry`.
   * Stamped on the audit-pack manifest entry; never on the signed payload.
   * `passive_telemetry` requires `receiptType='protectmcp:observation'`
   * (false-attestation guard). */
  captureTopology?: CaptureTopology;

  // === NSA CSI U/OO/6030316-26 alignment (cloud 0.5.0) ===

  /** `sha256:<hex>` of the tool output. Caller-supplied and forwarded
   * verbatim to the cloud. */
  resultDigest?: string;

  /** Receipt validity horizon. ISO-8601 string or POSIX number. The cloud
   * owns absolute time-binding and accepts only a duration, so the SDK
   * converts this absolute horizon to a client-computed `valid_seconds` and
   * sends that instead. Mutually exclusive with `validSeconds` (rule 10);
   * see `validSeconds` for precedence. */
  expiresAt?: string | number;

  /** Optional JSON schema describing the tool surface. When provided
   * alongside `toolName` and `toolFingerprint` is omitted, the SDK
   * computes the fingerprint client-side via JCS canonicalization. */
  toolSchema?: Record<string, unknown>;

  /** Explicit fingerprint over the canonical `{tool_name, schema}` blob.
   * When omitted but `toolName` + `toolSchema` are present, the SDK
   * derives it client-side. MUST be 32 bare lowercase hex chars
   * (SHA-256[:32], no `sha256:` prefix), matching the cloud wire form; use
   * `computeToolFingerprint` to avoid drift. */
  toolFingerprint?: string;

  /** `sha256:<hex>` of the agent's runtime configuration manifest.
   * REQUIRED for `receiptType='protectmcp:lifecycle:configuration_change'`
   * (false-attestation rule 9). MUST match `sha256:<64-hex>` (rule 11);
   * use `computeConfigManifestDigest` to avoid drift. */
  configManifestDigest?: string;

  /** `sha256:<hex>` of the agent's CVE inventory snapshot at signing time.
   * MUST match `sha256:<64-hex>` (rule 11); use
   * `computeCveInventoryDigest` to avoid drift. */
  cveInventoryDigest?: string;

  /** Build-provenance 4-tuple. `sha256:<hex>` of the executable that
   * invoked the action. Binds build-side provenance into the signed
   * receipt. MUST match `sha256:<64-hex>` (rule 11). */
  executableHash?: string;

  /** Build-provenance 4-tuple. `sha256:<hex>` of the canonical CycloneDX
   * or SPDX SBOM document. MUST match `sha256:<64-hex>` (rule 11). */
  sbomDigest?: string;

  /** Build-provenance 4-tuple. https URL to the SLSA attestation envelope
   * for the executing build. */
  slsaProvenancePointer?: string;

  /** Build-provenance 4-tuple. https URL to the in-toto, Sigstore, or
   * Rekor entry covering the executing build. */
  supplyChainPointer?: string;

  /** Caller-supplied list of MITRE ATT&CK technique ids (e.g.
   * `["T1059", "T1078"]`). Self-declared; never Asqav-verified. Cloud
   * sets `framework_mappings_self_declared=true` whenever any of the six
   * taxonomy lists is populated. */
  mitreTechniques?: string[];

  /** Caller-supplied list of MITRE ATLAS ids for AI-system threats
   * (e.g. `["AML.T0051", "AML.T0043"]`). Self-declared. */
  mitreAtlas?: string[];

  /** Caller-supplied list of OWASP Top 10 for LLM ids
   * (e.g. `["LLM01", "LLM02"]`). Self-declared. */
  owaspLlmTop10?: string[];

  /** Caller-supplied list of NIST AI RMF function ids and subcategories
   * (e.g. `["GOVERN-1.1", "MEASURE-2.7"]`). Self-declared. */
  nistAiRmf?: string[];

  /** Caller-supplied list of ISO/IEC 42001 control ids (e.g.
   * `["A.6.2.6"]`). Self-declared. */
  iso42001?: string[];

  /** Caller-supplied list of EU AI Act article ids (e.g.
   * `["Article-12", "Article-15"]`). Self-declared. */
  euAiActArticles?: string[];

  /** Caller-supplied base64-encoded RFC 3161 TimeStampResp (DER).
   * Preserved on the receipt for offline TSA chain verification
   * independent of cloud-issued anchors. */
  rfc3161Timestamp?: string;

  /** Caller-declared N-of-M durable-anchoring quorum. Shape
   * `{ required, witnesses: [<subset of "rfc3161", "opentimestamps">] }`.
   * `required` must be in `[1, witnesses.length]`; `witnesses` must be a
   * non-empty subset of the two shipped witnesses. `rekor` is rejected
   * (not a shipped witness). The receipt reaches `witness_quorum_met`
   * only when `required` witnesses hold a real inclusion proof. Projected
   * to the wire as snake_case `witness_policy`. */
  witnessPolicy?: WitnessPolicy;
}

export interface CoSignature {
  agentId: string;
  signature: string;
  signedAt: string;
}

export interface SignatureResponse {
  /** Polymorphic. Base64 string in non-compliance mode, `{alg, kid, sig}` object
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
  /** IETF chain link: `sha256(canonicalJson(predecessor_envelope))`,
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
  /** Original `policy_decision` value echoed from the request (one of
   * `"permit" | "deny" | "rate_limit"`). Kept for backward compat with
   * tooling built against the pre-spec implementation. */
  policyDecision?: PolicyDecision;
  /** Spec-shape `decision`: echoed from cloud or mapped from
   * `policyDecision` for older servers. Undefined for non-compliance. */
  decision?: Decision;
  /** Compliance Receipts envelope: the canonical signed dict. None on
   * non-compliance receipts. */
  payload?: Record<string, unknown>;
  /** Compliance Receipts envelope: the type-discriminated anchors
   * array. None on non-compliance receipts; `[]` when compliance_mode is on
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
 * The IETF profile sub-axes (`chainValid`, `anchorValidOts`,
 * `anchorValidRfc3161`, `anchorStatusOts`, `anchorStatusRfc3161`,
 * `signedAtSkewSeconds`, `missingFields`, `policyDigestResolved`,
 * `duplicateEmissionCandidate`, `regimesSatisfied`) populate only
 * when the cloud emits them on a compliance-mode receipt;
 * non-compliance-mode receipts leave them undefined. */
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
    | "agent_revoked_before_issuance"
    | "chain_break"
    | "anchor_invalid"
    | "missing_required_field"
    | "signed_at_skew"
    | string;
  /** Skew vs verifier wall clock; beyond the cloud's bound -> reject. */
  signedAtSkewSeconds?: number;
  /** True when the cloud could rederive the chain hash from the stored
   * `signed_envelope` and the predecessor matches. Undefined on
   * non-compliance-mode records. */
  chainValid?: boolean;
  /** OTS proof bool form. True when the OpenTimestamps proof rederived. */
  anchorValidOts?: boolean;
  /** Timestamp anchor bool form. True when the timestamp rederived. */
  anchorValidRfc3161?: boolean;
  /** Tri-state OTS status. `pending` means inside the upgrade window;
   * `invalid` means corruption or stale; `valid` mirrors the Bitcoin
   * block-confirmed case. */
  anchorStatusOts?: "valid" | "pending" | "invalid";
  /** Timestamp anchor outcome. Deterministic at issuance; never
   * `pending`. */
  anchorStatusRfc3161?: "valid" | "pending" | "invalid";
  /** REQUIRED-fields presence check. Surfaces column names NULLed
   * post-issuance on a compliance-mode receipt; empty / undefined means
   * every required field is present. */
  missingFields?: string[];
  /** True when the policy digest at issuance resolved to a known artefact. */
  policyDigestResolved?: boolean;
  /** True / non-empty when more than one receipt exists for the
   * (action_ref, issuer_id) pair; reporting flag, not a verification
   * downgrade. */
  duplicateEmissionCandidate?: boolean | string;
  /** Regulator tokens the cloud derived for the receipt (e.g.
   * `eu_ai_act`, `dora`). Empty / undefined on non-compliance-mode
   * receipts. */
  regimesSatisfied?: string[];
}

/** Anchor attestation state on a verification response. `status`
 * vocabulary: none | pending | confirmed | failed. `anchorTxRef` and
 * `anchorBlockHeight` populate once the OpenTimestamps proof upgrades
 * to a confirmed attestation. Field names are provider-agnostic so the
 * response shape survives a future switch off OpenTimestamps + Bitcoin. */
export interface BitcoinAnchorStatus {
  status: "none" | "pending" | "confirmed" | "failed" | string;
  anchorTxRef?: string | null;
  anchorBlockHeight?: number | null;
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
  /** Receipt kind. Always `"signature"` for this response shape. */
  type?: string;
  /** Bitcoin anchor state; `status: "none"` when the receipt has no anchor. */
  bitcoinAnchor?: BitcoinAnchorStatus;
  /** IETF projection of the signed envelope: `{alg, kid}`. Undefined on
   * non-compliance-mode receipts. */
  signatureEnvelope?: { alg: string; kid: string } & Record<string, string>;
  /** IETF anchors projection: list of `{type, value, ...}` per anchor.
   * `type` is `"opentimestamps"` or `"rfc3161"`. Undefined on
   * non-compliance-mode receipts. */
  anchors?: Array<
    {
      type: "opentimestamps" | "rfc3161" | string;
      value: string;
    } & Record<string, unknown>
  >;
  /** Algorithm registry version in force at issuance. Undefined on
   * rows lacking the column. */
  algorithmRegistryVersion?: string;
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

// === init ===

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

// === HTTP request helper with retry ===

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

// === Sign body builder (mode-aware) ===

/**
 * Merge explicit kwargs (`toolName`, `modelName`, `parentId`) into the
 * caller-supplied context as underscored sentinel keys so the hash-only
 * metadata builder picks them up. Returns a fresh object only when one
 * of the kwargs was present; otherwise returns the original ref. Mirrors
 * the Python SDK's surface-into-context behaviour.
 */
function surfaceKwargsIntoContext(options: SignOptions): Record<string, unknown> {
  const initial = options.context ?? {};
  if (
    options.toolName === undefined
    && options.modelName === undefined
    && options.parentId === undefined
  ) {
    return initial;
  }
  const merged: Record<string, unknown> = { ...initial };
  if (options.toolName !== undefined) merged._tool_name = options.toolName;
  if (options.modelName !== undefined) merged._model_name = options.modelName;
  if (options.parentId !== undefined) merged._parent_id = options.parentId;
  return merged;
}

/**
 * Fail-fast vocabulary checks on caller input before the HTTP roundtrip.
 * The cloud remains source of truth; these mirror the canonical sets so
 * obvious mistakes do not consume a network call. Throws AsqavError on
 * the first offending field.
 */
function validateSignOptions(options: SignOptions): void {
  if (
    options.receiptType !== undefined
    && !(RECEIPT_TYPE_NAMESPACE as readonly string[]).includes(options.receiptType)
  ) {
    throw new AsqavError(
      `invalid_receipt_type: must be one of ${RECEIPT_TYPE_NAMESPACE.join(", ")}`,
    );
  }
  if (
    (options.policyDecision === "deny" || options.policyDecision === "rate_limit")
    && !options.reason
  ) {
    throw new AsqavError(
      "missing_reason: policy_decision=deny|rate_limit requires a `reason` code",
    );
  }
  if (
    options.sandboxState !== undefined
    && !(SANDBOX_STATE_NAMESPACE as readonly string[]).includes(options.sandboxState)
  ) {
    throw new AsqavError(
      `invalid_sandbox_state: '${options.sandboxState}' must be one of ${SANDBOX_STATE_NAMESPACE.join(", ")}`,
    );
  }
  if (
    options.captureTopology !== undefined
    && !(CAPTURE_TOPOLOGY_NAMESPACE as readonly string[]).includes(options.captureTopology)
  ) {
    throw new AsqavError(
      `invalid_capture_topology: '${options.captureTopology}' must be one of ${CAPTURE_TOPOLOGY_NAMESPACE.join(", ")}`,
    );
  }
  // Rule 8 lockstep with the cloud SignRequest validator: passive_telemetry
  // pairs only with protectmcp:observation or protectmcp:observation:result_bound.
  if (
    options.captureTopology === "passive_telemetry"
    && options.receiptType !== undefined
    && options.receiptType !== "protectmcp:observation"
    && options.receiptType !== "protectmcp:observation:result_bound"
  ) {
    const offending = options.receiptType.split(":").slice(1).join(":") || options.receiptType;
    throw new AsqavError(
      `false_attestation_guard: capture_topology=passive_telemetry receipts must use receipt_type=protectmcp:observation[:result_bound], not :${offending} (rule 8)`,
    );
  }
  // Rule 9 lockstep with the cloud SignRequest cross-field validator.
  if (
    options.receiptType === "protectmcp:lifecycle:configuration_change"
    && options.configManifestDigest === undefined
  ) {
    throw new AsqavError(
      "configuration_change_missing_config_manifest_digest: receipt_type=protectmcp:lifecycle:configuration_change requires config_manifest_digest (sha256:<64 hex>).",
    );
  }
  if (
    options.receiptType === "protectmcp:observation:result_bound"
    && options.resultDigest === undefined
  ) {
    throw new AsqavError(
      "result_bound_missing_result_digest: receipt_type=protectmcp:observation:result_bound requires result_digest (sha256:<64 hex>).",
    );
  }
  // Rule 10: an absolute horizon and a duration together are ambiguous; reject both.
  if (options.validSeconds !== undefined && options.expiresAt !== undefined) {
    throw new AsqavError(
      "expiry_collision_guard: pass either valid_seconds or expires_at, not both (rule 10)",
    );
  }
  // Rule 11 lockstep: per-field tokens mirror cloud <field>_not_sha256_wire_form.
  const _digestChecks: Array<[string, string | undefined]> = [
    ["config_manifest_digest", options.configManifestDigest],
    ["cve_inventory_digest", options.cveInventoryDigest],
    ["executable_hash", options.executableHash],
    ["sbom_digest", options.sbomDigest],
  ];
  for (const [name, value] of _digestChecks) {
    if (value !== undefined && !SHA256_HEX_RE.test(value)) {
      throw new AsqavError(
        `${name}_not_sha256_wire_form: must look like 'sha256:<64 lowercase hex>'.`,
      );
    }
  }
  if (options.toolFingerprint !== undefined && !TOOL_FINGERPRINT_RE.test(options.toolFingerprint)) {
    throw new AsqavError(
      "tool_fingerprint_not_32_hex_chars: must be 32 lowercase hex chars (SHA-256[:32]).",
    );
  }
  const _pointerChecks: Array<[string, string | undefined]> = [
    ["slsa_provenance_pointer", options.slsaProvenancePointer],
    ["supply_chain_pointer", options.supplyChainPointer],
  ];
  for (const [name, value] of _pointerChecks) {
    if (value !== undefined && !(value.startsWith("https://") || value.startsWith("http://"))) {
      throw new AsqavError(
        `pointer_url_guard: ${name} must be an http(s) URL`,
      );
    }
  }
  // Threat-framework taxonomy validators. Lockstep with the cloud
  // SignRequest cross-field validator: lists must be non-empty
  // arrays of strings (each entry up to 128 chars); rfc3161Timestamp
  // must be valid base64. Verbatim guard tokens kept in sync with
  // the cloud and Python SDK so SDK errors round-trip through the
  // conformance vectors.
  const _taxonomyChecks: Array<[string, string[] | undefined]> = [
    ["mitre_techniques", options.mitreTechniques],
    ["mitre_atlas", options.mitreAtlas],
    ["owasp_llm_top10", options.owaspLlmTop10],
    ["nist_ai_rmf", options.nistAiRmf],
    ["iso_42001", options.iso42001],
    ["eu_ai_act_articles", options.euAiActArticles],
  ];
  for (const [name, value] of _taxonomyChecks) {
    if (value === undefined) continue;
    if (!Array.isArray(value) || value.length === 0) {
      throw new AsqavError(
        `${name}_must_be_non_empty_list: pass a non-empty list of strings or omit the field.`,
      );
    }
    for (const item of value) {
      if (typeof item !== "string" || item.length === 0 || item.length > 128) {
        throw new AsqavError(
          `${name}_entry_invalid: each entry must be a non-empty string of length <= 128.`,
        );
      }
    }
  }
  if (options.rfc3161Timestamp !== undefined) {
    const t = options.rfc3161Timestamp;
    if (typeof t !== "string" || t.length === 0) {
      throw new AsqavError(
        "rfc3161_timestamp_not_base64: must be a non-empty base64-encoded TimeStampResp.",
      );
    }
    // Tight base64 shape check; reject any non-base64 char or bad padding.
    if (!/^[A-Za-z0-9+/]+={0,2}$/.test(t) || (t.length % 4) !== 0) {
      throw new AsqavError(
        "rfc3161_timestamp_not_base64: must be valid base64.",
      );
    }
    try {
      // Node Buffer round-trip catches padding errors strict mode misses.
      const { Buffer } = require("node:buffer") as typeof import("node:buffer");
      const decoded = Buffer.from(t, "base64");
      const reencoded = decoded.toString("base64");
      // Reject when re-encoded form differs (catches non-canonical padding).
      if (reencoded !== t) {
        throw new AsqavError(
          "rfc3161_timestamp_not_base64: must be valid base64.",
        );
      }
    } catch (err) {
      if (err instanceof AsqavError) throw err;
      throw new AsqavError(
        `rfc3161_timestamp_not_base64: must be valid base64 (decode failed: ${err}).`,
      );
    }
  }
  validateWitnessPolicy(options.witnessPolicy);
  validateIncidentClass(options.incidentClass);
}

/**
 * Reject malformed `witnessPolicy` before the HTTP roundtrip. Lockstep
 * with the cloud witness_policy extension: `{ required, witnesses }` where
 * `witnesses` is a non-empty subset of the two shipped witnesses and
 * `required` is an integer in `[1, witnesses.length]`. `rekor` is rejected
 * because it is not a shipped witness. Verbatim guard tokens match the
 * Python SDK so SDK errors round-trip through the conformance vectors.
 */
function validateWitnessPolicy(value: WitnessPolicy | undefined): void {
  if (value === undefined) return;
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new AsqavError(
      'witness_policy_invalid: must be an object { required: number, witnesses: [...] }.',
    );
  }
  const { required, witnesses } = value;
  if (!Array.isArray(witnesses) || witnesses.length === 0) {
    throw new AsqavError(
      `witness_policy_witnesses_must_be_non_empty_list: witnesses must be a non-empty list drawn from ${WITNESS_NAMESPACE.join(", ")}.`,
    );
  }
  for (const w of witnesses) {
    if (!(WITNESS_NAMESPACE as readonly string[]).includes(w)) {
      throw new AsqavError(
        `witness_policy_unknown_witness: '${w}' is not a shipped witness; witnesses must be a subset of ${WITNESS_NAMESPACE.join(", ")} (rekor is rejected).`,
      );
    }
  }
  if (new Set(witnesses).size !== witnesses.length) {
    throw new AsqavError(
      "witness_policy_duplicate_witness: witnesses must not contain duplicates.",
    );
  }
  if (typeof required !== "number" || !Number.isInteger(required)) {
    throw new AsqavError(
      "witness_policy_required_must_be_int: required must be an integer in [1, witnesses.length].",
    );
  }
  if (required < 1 || required > witnesses.length) {
    throw new AsqavError(
      `witness_policy_required_out_of_range: required must be in [1, ${witnesses.length}] (got ${required}).`,
    );
  }
}

/**
 * Reject `incident_class` values outside the canonical union (HIPAA
 * token plus the six DORA tokens). Accepts a single value, an array,
 * or undefined / empty string (skip).
 */
function validateIncidentClass(value: string | string[] | undefined): void {
  if (value === undefined || value === "") return;
  const values = Array.isArray(value) ? value : [value];
  for (const ic of values) {
    if (!(INCIDENT_CLASS_NAMESPACE as readonly string[]).includes(ic)) {
      throw new AsqavError(
        `invalid_incident_class: '${ic}' must be one of ${INCIDENT_CLASS_NAMESPACE.join(", ")}`,
      );
    }
  }
}

/**
 * Derive `action_ref` locally when omitted under compliance mode; matches
 * the cloud's `hash_action` shape (`sha256:<hex>` over canonical JSON of
 * `{action_type, context}`). Returns the caller's value verbatim when
 * compliance mode is off or `actionRef` is already set.
 */
function deriveActionRef(
  complianceMode: boolean,
  actionRef: string | undefined,
  actionType: string,
  context: Record<string, unknown>,
): string | undefined {
  if (!complianceMode || actionRef !== undefined) return actionRef;
  const action = { action_type: actionType, context };
  const hex = createHash("sha256").update(canonicalJson(action)).digest("hex");
  return `sha256:${hex}`;
}

/** Wire format for every caller-supplied digest field
 * (`tool_fingerprint`, `config_manifest_digest`, `cve_inventory_digest`,
 * `result_digest`). The cloud accepts the same exact-match form so the
 * SDK surfaces bad inputs before the HTTP roundtrip; the regex matches
 * the helper outputs byte-for-byte. */
const SHA256_HEX_RE = /^sha256:[a-f0-9]{64}$/;

/** Wire format for `tool_fingerprint`: 32 bare lowercase hex chars
 * (SHA-256[:32]). The cloud validates this field with the bare form, not
 * the self-describing `sha256:<hex>` form used by the other digests. */
const TOOL_FINGERPRINT_RE = /^[0-9a-f]{32}$/;

/** Compute a deterministic 32-bare-hex fingerprint over `{tool_name,
 * schema}` per NSA CSI U/OO/6030316-26. The fingerprint is
 * byte-deterministic under JCS (RFC 8785) so the SDK and cloud agree when
 * the cloud rehashes for tamper detection. The cloud requires the first 32
 * hex chars of the SHA-256 digest, bare (no `sha256:` prefix), matching
 * `TOOL_FINGERPRINT_RE`. */
export function computeToolFingerprint(
  toolName: string,
  toolSchema: Record<string, unknown> | undefined,
): string {
  const payload = canonicalJson({ tool_name: toolName, schema: toolSchema ?? {} });
  const hex = createHash("sha256").update(payload).digest("hex");
  return hex.slice(0, 32);
}

/** Compute the canonical `sha256:<hex>` of a runtime configuration
 * manifest. NSA CSI U/OO/6030316-26 anchors the "silent capability creep"
 * audit on `protectmcp:lifecycle:configuration_change` receipts (rule 9).
 * This helper produces a digest byte-deterministic under JCS (RFC 8785)
 * so two auditors who receive the same manifest object produce identical
 * digests regardless of insertion order. Output matches `SHA256_HEX_RE`
 * byte-for-byte. */
export function computeConfigManifestDigest(
  manifest: Record<string, unknown>,
): string {
  const hex = createHash("sha256").update(canonicalJson(manifest)).digest("hex");
  return `sha256:${hex}`;
}

/** Compute the canonical `sha256:<hex>` of a CVE inventory snapshot. NSA
 * CSI U/OO/6030316-26 anchors the "known-vulnerable invocation" audit on
 * receipts that carry `cve_inventory_digest`. The helper takes the list
 * of CVE records (any JSON-serialisable shape) and produces a digest
 * byte-deterministic under JCS (RFC 8785). Output matches
 * `SHA256_HEX_RE` byte-for-byte. */
export function computeCveInventoryDigest(cveList: unknown[]): string {
  const hex = createHash("sha256").update(canonicalJson(cveList)).digest("hex");
  return `sha256:${hex}`;
}

/** Return a 24-hex-char (12 random bytes) nonce for replay protection.
 * The cloud verifier rejects duplicate nonces per `(agent_id, action_ref)`
 * inside the validity window. */
export function generateNonce(): string {
  const { randomBytes } = require("node:crypto") as typeof import("node:crypto");
  return randomBytes(12).toString("hex");
}

/**
 * Tack the optional non-IETF wire fields (`co_signers`, `nonce`,
 * `valid_seconds`, `expected_executor_pubkey_b64`) onto the body when
 * the caller supplied them. Mutates `body` in place for parity with the
 * inline original. Auto-generates `nonce` for cloud-side replay
 * protection (NSA CSI alignment) when the caller omits one.
 */
function applyOptionalWireFields(
  body: Record<string, unknown>,
  options: SignOptions,
): void {
  if (options.coSigners && options.coSigners.length > 0) {
    body.co_signers = [...options.coSigners];
  }
  if (options.mandateId !== undefined) {
    body.mandate_id = options.mandateId;
  }
  body.nonce = options.nonce ?? generateNonce();
  // Convert an absolute expiresAt to a client-computed valid_seconds (cloud owns time-binding).
  if (options.validSeconds !== undefined) {
    body.valid_seconds = options.validSeconds;
  } else if (options.expiresAt !== undefined) {
    const expiresAtMs = typeof options.expiresAt === "number"
      ? options.expiresAt * 1000
      : Date.parse(options.expiresAt);
    const seconds = Math.ceil((expiresAtMs - Date.now()) / 1000);
    body.valid_seconds = Math.max(1, seconds);
  }
  if (options.expectedExecutorPubkeyB64 !== undefined) {
    body.expected_executor_pubkey_b64 = options.expectedExecutorPubkeyB64;
  }
}

/** Snake_case wire-key projection of the optional IETF profile fields
 * that do not depend on `complianceMode`. Defined as a const table so
 * the `applyComplianceFields` loop stays single-pass. */
const IETF_OPTIONAL_FIELD_MAP: ReadonlyArray<{
  wire: string;
  read: (o: SignOptions) => unknown;
}> = [
  { wire: "sandbox_state", read: (o) => o.sandboxState },
  { wire: "iteration_id", read: (o) => o.iterationId },
  { wire: "risk_class", read: (o) => o.riskClass },
  { wire: "incident_class", read: (o) => o.incidentClass },
  { wire: "issuer_id", read: (o) => o.issuerId },
  { wire: "payload_digest", read: (o) => o.payloadDigest },
  { wire: "receipt_type", read: (o) => o.receiptType },
  { wire: "reason", read: (o) => o.reason },
  // NSA CSI U/OO/6030316-26 alignment (cloud 0.5.0).
  { wire: "result_digest", read: (o) => o.resultDigest },
  { wire: "config_manifest_digest", read: (o) => o.configManifestDigest },
  { wire: "cve_inventory_digest", read: (o) => o.cveInventoryDigest },
  { wire: "executable_hash", read: (o) => o.executableHash },
  { wire: "sbom_digest", read: (o) => o.sbomDigest },
  { wire: "slsa_provenance_pointer", read: (o) => o.slsaProvenancePointer },
  { wire: "supply_chain_pointer", read: (o) => o.supplyChainPointer },
  // Threat-framework taxonomy mappings (cloud 0.5.1+).
  { wire: "mitre_techniques", read: (o) => o.mitreTechniques },
  { wire: "mitre_atlas", read: (o) => o.mitreAtlas },
  { wire: "owasp_llm_top10", read: (o) => o.owaspLlmTop10 },
  { wire: "nist_ai_rmf", read: (o) => o.nistAiRmf },
  { wire: "iso_42001", read: (o) => o.iso42001 },
  { wire: "eu_ai_act_articles", read: (o) => o.euAiActArticles },
  { wire: "rfc3161_timestamp", read: (o) => o.rfc3161Timestamp },
  // Durable-anchoring quorum (cloud witness_policy extension).
  { wire: "witness_policy", read: (o) => o.witnessPolicy },
  {
    wire: "tool_fingerprint",
    read: (o) =>
      o.toolFingerprint
      ?? (o.toolName !== undefined && o.toolSchema !== undefined
        ? computeToolFingerprint(o.toolName, o.toolSchema)
        : undefined),
  },
];

/**
 * Project IETF Compliance Receipts profile fields onto the request body
 * using snake_case wire keys. Adds `compliance_mode`, `action_ref`, the
 * non-conditional optional fields (`sandbox_state`, `iteration_id`,
 * `risk_class`, `incident_class`, `issuer_id`, `payload_digest`,
 * `receipt_type`, `reason`), then the decision / topology fields that
 * depend on `complianceMode`.
 */
function applyComplianceFields(
  body: Record<string, unknown>,
  options: SignOptions,
  complianceMode: boolean,
  actionRef: string | undefined,
): void {
  if (complianceMode) body.compliance_mode = true;
  if (actionRef !== undefined) body.action_ref = actionRef;
  for (const { wire, read } of IETF_OPTIONAL_FIELD_MAP) {
    const value = read(options);
    if (value !== undefined) body[wire] = value;
  }
  applyDecisionFields(body, options, complianceMode);
}

/** Attach `policy_decision`, the spec-shape `decision` mirror, and the
 * compliance-only `capture_topology` to the body. Split out so
 * `applyComplianceFields` stays a flat loop plus one helper call. */
function applyDecisionFields(
  body: Record<string, unknown>,
  options: SignOptions,
  complianceMode: boolean,
): void {
  if (options.policyDecision !== undefined) {
    body.policy_decision = options.policyDecision;
    if (complianceMode) {
      body.decision = mapPolicyDecisionToDecision(options.policyDecision);
    }
  }
  if (complianceMode && options.captureTopology !== undefined) {
    body.capture_topology = options.captureTopology;
  }
}

/** Wire shape returned by `POST /agents/:id/sign`. Defined once so the
 * response-shaping helper can be reused and type-checked. */
interface SignWireResponse {
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
}

/**
 * Shape the snake_case wire response into the camelCase
 * `SignatureResponse` returned to callers. Prefers cloud-supplied
 * `decision`; maps from `policy_decision` for older clouds under
 * compliance_mode. Accepts either casing on `previous_receipt_hash`.
 */
function mapSignWireToResponse(data: SignWireResponse): SignatureResponse {
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
    complianceMode: data.compliance_mode,
    previousReceiptHash: data.previousReceiptHash ?? data.previous_receipt_hash,
    actionRef: data.action_ref,
    policyDigest: data.policy_digest,
    receiptType: data.receipt_type,
    issuerId: data.issuer_id,
    policyDecision: data.policy_decision as PolicyDecision | undefined,
    decision: (data.decision as Decision | undefined) ?? (
      data.compliance_mode
        ? mapPolicyDecisionToDecision(data.policy_decision)
        : undefined
    ),
    payload: data.payload,
    anchors: data.anchors,
  };
}

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

// === Agent ===

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
    const initialContext = surfaceKwargsIntoContext(options);
    const finalContext = _dispatchBefore(options.actionType, initialContext);
    const complianceMode = options.complianceMode !== false;

    validateSignOptions(options);

    const actionRef = deriveActionRef(
      complianceMode,
      options.actionRef,
      options.actionType,
      finalContext,
    );

    const body = await buildSignBody({
      actionType: options.actionType,
      context: finalContext,
      sessionId: this.sessionId,
      agentId: this.agentId,
      userIntent: options.userIntent,
    });
    applyOptionalWireFields(body, options);
    applyComplianceFields(body, options, complianceMode, actionRef);

    const data = await request<SignWireResponse>(
      "POST",
      `/agents/${this.agentId}/sign`,
      body,
    );
    const response = mapSignWireToResponse(data);
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

// === Standalone functions ===

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
      anchor_valid_ots?: boolean;
      anchor_valid_rfc3161?: boolean;
      anchor_status_ots?: "valid" | "pending" | "invalid";
      anchor_status_rfc3161?: "valid" | "pending" | "invalid";
      missing_fields?: string[];
      policy_digest_resolved?: boolean;
      duplicate_emission_candidate?: boolean | string;
      regimes_satisfied?: string[];
    };
    type?: string;
    bitcoin_anchor?: {
      status: string;
      anchor_tx_ref?: string | null;
      anchor_block_height?: number | null;
    };
    signature_envelope?: { alg: string; kid: string } & Record<string, string>;
    anchors?: Array<
      {
        type: "opentimestamps" | "rfc3161" | string;
        value: string;
      } & Record<string, unknown>
    >;
    algorithm_registry_version?: string;
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
          anchorValidOts: data.verification_detail.anchor_valid_ots,
          anchorValidRfc3161: data.verification_detail.anchor_valid_rfc3161,
          anchorStatusOts: data.verification_detail.anchor_status_ots,
          anchorStatusRfc3161: data.verification_detail.anchor_status_rfc3161,
          missingFields: data.verification_detail.missing_fields,
          policyDigestResolved: data.verification_detail.policy_digest_resolved,
          duplicateEmissionCandidate: data.verification_detail.duplicate_emission_candidate,
          regimesSatisfied: data.verification_detail.regimes_satisfied,
        }
      : undefined,
    type: data.type,
    bitcoinAnchor: data.bitcoin_anchor
      ? {
          status: data.bitcoin_anchor.status,
          anchorTxRef: data.bitcoin_anchor.anchor_tx_ref,
          anchorBlockHeight: data.bitcoin_anchor.anchor_block_height,
        }
      : undefined,
    signatureEnvelope: data.signature_envelope,
    anchors: data.anchors,
    algorithmRegistryVersion: data.algorithm_registry_version,
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

// === Internal config exposure for tests only ===

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
