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

import { canonicalize, hashAction } from "./canonicalize.js";
import { _dispatchAfter, _dispatchBefore } from "./hooks.js";
import { type Mode, resolveMode } from "./mode.js";

export { clearHooks, registerAfter, registerBefore } from "./hooks.js";
export type { AfterHook, BeforeHook } from "./hooks.js";
export { canonicalize, hashAction } from "./canonicalize.js";
export { resolveMode, isAsqavCloudHost, type Mode } from "./mode.js";

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
  algorithm?: string;
  capabilities?: string[];
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
}

export interface CoSignature {
  agentId: string;
  signature: string;
  signedAt: string;
}

export interface SignatureResponse {
  signature: string;
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
}

export interface SessionResponse {
  sessionId: string;
  agentId: string;
  status: string;
  startedAt: string;
  endedAt?: string;
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
    const digest = await hashAction(
      args.actionType,
      args.context,
      config.orgSalt ?? undefined,
    );
    const metadata: Record<string, unknown> = {
      agent_id: args.agentId,
      action_type: args.actionType,
    };
    if (args.sessionId !== null) metadata.session_id = args.sessionId;
    // Optional opt-in: caller adds ``_model_name`` / ``_tool_name`` /
    // ``_parent_id`` to the context. Other context keys are NEVER
    // copied. These three identifiers power the agent graph view in
    // the dashboard - only the name strings travel, never prompts or
    // tool args.
    const ctx = args.context as Record<string, unknown>;
    if (typeof ctx._model_name === "string") metadata.model_name = ctx._model_name;
    if (typeof ctx._tool_name === "string") metadata.tool_name = ctx._tool_name;
    if (typeof ctx._parent_id === "string") metadata.parent_id = ctx._parent_id;
    // Defensive: drop anything that snuck in outside the whitelist.
    for (const k of Object.keys(metadata)) {
      if (!HASH_ONLY_METADATA_WHITELIST.has(k)) delete metadata[k];
    }
    const hashBody: Record<string, unknown> = {
      action_type: args.actionType,
      hash: digest,
      hash_algo: "sha256",
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
    const data = await request<AgentData>("POST", "/agents/create", {
      name: options.name,
      algorithm: options.algorithm ?? "ml-dsa-65",
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
