#!/usr/bin/env node
/**
 * asqav CLI for TypeScript / Node.
 *
 * Mirrors the Python `asqav` CLI surface:
 *   asqav --version
 *   asqav verify <signature_id>
 *   asqav agents list / create <name>
 *   asqav sessions list / end <session_id>
 *   asqav replay <agent_id> <session_id>
 *   asqav preflight <agent_id> <action_type>
 *   asqav budget check / record
 *   asqav approve <session_id> <entity_id>
 *   asqav compliance frameworks / export
 *
 * Pro-only commands (replay, preflight, budget, approve) and the
 * Business-only `compliance export` are gated client-side via the same
 * GET /account endpoint the Python CLI uses; the server is still the
 * source of truth.
 */

import {
  Agent,
  APIError,
  BudgetTracker,
  generateKeypair,
  init,
  listSessions,
  request,
  verifyChain,
  verifySignature,
  type LocalSigningAlgorithm,
} from "./index.js";

const CLI_VERSION = "0.3.2";
const TIER_RANK: Record<string, number> = {
  free: 0,
  pro: 1,
  business: 2,
  enterprise: 3,
};

function die(msg: string, code = 1): never {
  process.stderr.write(`${msg}\n`);
  process.exit(code);
}

function ensureApiKey(): void {
  if (!process.env.ASQAV_API_KEY) {
    die("Error: ASQAV_API_KEY environment variable required.");
  }
  init({ apiKey: process.env.ASQAV_API_KEY });
}

async function fetchTier(): Promise<string | null> {
  try {
    const data = await request<{ tier?: string }>("GET", "/account");
    if (data && typeof data.tier === "string" && data.tier.length > 0) {
      return data.tier.toLowerCase();
    }
  } catch {
    // Older self-hosted deployments without /account; skip the gate.
  }
  return null;
}

async function requireTier(minTier: string): Promise<void> {
  const tier = await fetchTier();
  if (tier === null) return;
  const have = TIER_RANK[tier] ?? 0;
  const need = TIER_RANK[minTier.toLowerCase()] ?? 0;
  if (have >= need) return;
  const titled = minTier.charAt(0).toUpperCase() + minTier.slice(1);
  die(
    `Error: this command requires the ${titled} tier (current: ${tier}). Upgrade at https://asqav.com/pricing.`,
    2,
  );
}

function parseFlag(args: string[], name: string): string | undefined {
  for (let i = 0; i < args.length; i++) {
    if (args[i] === `--${name}`) return args[i + 1];
    const prefix = `--${name}=`;
    if (args[i]?.startsWith(prefix)) return args[i].slice(prefix.length);
  }
  return undefined;
}

function hasFlag(args: string[], name: string): boolean {
  return args.includes(`--${name}`);
}

function positional(args: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === undefined) continue;
    if (a.startsWith("--")) {
      // skip flag and its value if no `=`
      if (!a.includes("=")) i++;
      continue;
    }
    out.push(a);
  }
  return out;
}

// === Commands ===

async function cmdVerify(args: string[]): Promise<void> {
  const [sigId] = positional(args);
  if (!sigId) die("Usage: asqav verify <signature_id> [--output text|json]");
  const output = parseFlag(args, "output") ?? "text";
  // Verify is public, but `request` short-circuits on missing config; init with any key.
  init({ apiKey: process.env.ASQAV_API_KEY ?? "anon" });
  try {
    const r = await verifySignature(sigId);
    if (output === "json") {
      process.stdout.write(`${JSON.stringify(r, null, 2)}\n`);
      return;
    }
    process.stdout.write(`Signature: ${r.verified ? "valid" : "invalid"}\n`);
    if (r.agentId) process.stdout.write(`Agent: ${r.agentName ?? "?"} (${r.agentId})\n`);
    if (r.actionType) process.stdout.write(`Action: ${r.actionType}\n`);
    process.stdout.write(`Algorithm: ${r.algorithm}\n`);
    const d = r.verificationDetail;
    if (d) {
      process.stdout.write(`Validation: ${d.validationLabel}\n`);
      if (d.chainValid !== undefined) {
        process.stdout.write(`Chain valid: ${d.chainValid}\n`);
      }
      if (d.anchorStatusOts !== undefined) {
        process.stdout.write(`Anchor (OTS): ${d.anchorStatusOts}\n`);
      }
      if (d.anchorStatusRfc3161 !== undefined) {
        process.stdout.write(`Anchor (RFC 3161): ${d.anchorStatusRfc3161}\n`);
      }
      if (d.signedAtSkewSeconds !== undefined) {
        process.stdout.write(`signed_at skew: ${d.signedAtSkewSeconds}s\n`);
      }
      if (d.missingFields && d.missingFields.length > 0) {
        process.stdout.write(`Missing fields: ${d.missingFields.join(", ")}\n`);
      }
    }
  } catch (err) {
    if (err instanceof APIError && err.statusCode === 404) {
      die(`Signature not found: ${sigId}`);
    }
    die(`Error: ${(err as Error).message}`);
  }
}

async function cmdAgentsList(): Promise<void> {
  ensureApiKey();
  try {
    const data = await request<unknown>("GET", "/agents");
    const list = Array.isArray(data) ? data : (data as { agents?: unknown[] })?.agents ?? [];
    if (!Array.isArray(list) || list.length === 0) {
      process.stdout.write("No agents found.\n");
      return;
    }
    for (const a of list as Array<{ name?: string; agent_id?: string; algorithm?: string }>) {
      process.stdout.write(`  ${a.name ?? "unknown"} (${a.agent_id ?? "unknown"}) [${a.algorithm ?? "unknown"}]\n`);
    }
  } catch (err) {
    die(`Error: ${(err as Error).message}`);
  }
}

async function cmdAgentsCreate(args: string[]): Promise<void> {
  const [name] = positional(args);
  if (!name) die("Usage: asqav agents create <name>");
  ensureApiKey();
  try {
    const a = await Agent.create({ name });
    process.stdout.write(`Agent created: ${a.name} (${a.agentId})\n`);
    process.stdout.write(`Algorithm: ${a.algorithm}\n`);
    process.stdout.write(`Key ID: ${a.keyId}\n`);
  } catch (err) {
    die(`Error: ${(err as Error).message}`);
  }
}

async function cmdAgentRevoke(args: string[]): Promise<void> {
  const [agentId] = positional(args);
  const reason = parseFlag(args, "reason") ?? "manual";
  if (!agentId) die("Usage: asqav agents revoke <agent_id> [--reason X]");
  ensureApiKey();
  try {
    const agent = await Agent.get(agentId);
    await agent.revoke({ reason });
    process.stdout.write(`Revoked ${agentId}\n`);
  } catch (err) {
    die(`Error: ${(err as Error).message}`);
  }
}

async function cmdSessionsList(args: string[]): Promise<void> {
  ensureApiKey();
  const limit = parseFlag(args, "limit");
  const status = parseFlag(args, "status");
  const agentId = parseFlag(args, "agent");
  const sessions = await listSessions({
    limit: limit ? Number(limit) : undefined,
    status,
    agentId,
  });
  if (sessions.length === 0) {
    process.stdout.write("No sessions found.\n");
    return;
  }
  for (const s of sessions) {
    process.stdout.write(`  ${s.sessionId}  agent=${s.agentId}  status=${s.status}  started=${s.startedAt}\n`);
  }
}

async function cmdSessionsEnd(args: string[]): Promise<void> {
  const [sessionId] = positional(args);
  const status = parseFlag(args, "status") ?? "completed";
  if (!sessionId) die("Usage: asqav sessions end <session_id> [--status completed]");
  ensureApiKey();
  try {
    await request("PATCH", `/sessions/${sessionId}`, { status });
    process.stdout.write(`Session ${sessionId} -> ${status}\n`);
  } catch (err) {
    die(`Error: ${(err as Error).message}`);
  }
}

async function cmdReplay(args: string[]): Promise<void> {
  const [agentId, sessionId] = positional(args);
  if (!agentId || !sessionId) {
    die("Usage: asqav replay <agent_id> <session_id> [--json]");
  }
  ensureApiKey();
  await requireTier("pro");
  try {
    const data = await request<{ steps?: unknown[]; chain_integrity?: boolean }>(
      "GET",
      `/agents/${agentId}/sessions/${sessionId}/replay`,
    );
    if (hasFlag(args, "json")) {
      process.stdout.write(`${JSON.stringify(data, null, 2)}\n`);
    } else {
      const steps = Array.isArray(data.steps) ? data.steps.length : 0;
      process.stdout.write(`Steps: ${steps}\n`);
      process.stdout.write(`Chain integrity: ${data.chain_integrity ? "ok" : "FAILED"}\n`);
    }
    if (data.chain_integrity === false) process.exit(1);
  } catch (err) {
    die(`Error: ${(err as Error).message}`);
  }
}

async function cmdPreflight(args: string[]): Promise<void> {
  const [agentId, actionType] = positional(args);
  if (!agentId || !actionType) {
    die("Usage: asqav preflight <agent_id> <action_type> [--json]");
  }
  ensureApiKey();
  await requireTier("pro");
  try {
    const agent = await Agent.get(agentId);
    const r = await agent.preflight(actionType);
    if (hasFlag(args, "json")) {
      process.stdout.write(`${JSON.stringify(r, null, 2)}\n`);
    } else {
      process.stdout.write(`${r.cleared ? "CLEARED" : "BLOCKED"}  ${r.explanation}\n`);
      for (const reason of r.reasons) process.stdout.write(`  - ${reason}\n`);
    }
    if (!r.cleared) process.exit(1);
  } catch (err) {
    die(`Error: ${(err as Error).message}`);
  }
}

async function cmdBudgetCheck(args: string[]): Promise<void> {
  const agentId = parseFlag(args, "agent-id");
  const limit = Number(parseFlag(args, "limit"));
  const estimated = Number(parseFlag(args, "estimated-cost"));
  const currentSpend = Number(parseFlag(args, "current-spend") ?? "0");
  const currency = parseFlag(args, "currency") ?? "USD";
  if (!agentId || !Number.isFinite(limit) || !Number.isFinite(estimated)) {
    die("Usage: asqav budget check --agent-id <id> --limit N --estimated-cost N [--current-spend N] [--currency USD]");
  }
  ensureApiKey();
  await requireTier("pro");
  const agent = await Agent.get(agentId);
  const tracker = new BudgetTracker({ agent, limit, currency });
  // Apply existing spend by recording a synthetic priming amount in memory only.
  // The CLI doesn't sign anything during 'check'; this matches the Python parity.
  (tracker as unknown as { spend: number }).spend = currentSpend;
  const r = tracker.check(estimated);
  if (hasFlag(args, "json")) {
    process.stdout.write(`${JSON.stringify(r, null, 2)}\n`);
  } else {
    process.stdout.write(`${r.allowed ? "ALLOWED" : "DENIED"}  remaining=${r.remaining.toFixed(4)} ${currency}\n`);
    if (r.reason) process.stdout.write(`  reason: ${r.reason}\n`);
  }
  if (!r.allowed) process.exit(1);
}

async function cmdBudgetRecord(args: string[]): Promise<void> {
  const agentId = parseFlag(args, "agent-id");
  const action = parseFlag(args, "action");
  const limit = Number(parseFlag(args, "limit"));
  const cost = Number(parseFlag(args, "actual-cost"));
  const currentSpend = Number(parseFlag(args, "current-spend") ?? "0");
  const currency = parseFlag(args, "currency") ?? "USD";
  if (!agentId || !action || !Number.isFinite(limit) || !Number.isFinite(cost)) {
    die("Usage: asqav budget record --agent-id <id> --action <type> --actual-cost N --limit N [--current-spend N] [--currency USD]");
  }
  ensureApiKey();
  await requireTier("pro");
  try {
    const agent = await Agent.get(agentId);
    const tracker = new BudgetTracker({ agent, limit, currency });
    (tracker as unknown as { spend: number }).spend = currentSpend;
    const sig = await tracker.record(action, cost);
    process.stdout.write(`signature_id: ${sig.signatureId}\n`);
    process.stdout.write(`action_id:    ${sig.actionId}\n`);
    process.stdout.write(`verify_url:   ${sig.verificationUrl}\n`);
  } catch (err) {
    die(`Error: ${(err as Error).message}`);
  }
}

async function cmdApprove(args: string[]): Promise<void> {
  const [sessionId, entityId] = positional(args);
  if (!sessionId || !entityId) {
    die("Usage: asqav approve <session_id> <entity_id> [--json]");
  }
  ensureApiKey();
  await requireTier("pro");
  try {
    const data = await request<unknown>("POST", `/sessions/${sessionId}/approve`, { entity_id: entityId });
    if (hasFlag(args, "json")) {
      process.stdout.write(`${JSON.stringify(data, null, 2)}\n`);
    } else {
      const d = data as { signatures_collected?: number; approvals_required?: number; status?: string };
      process.stdout.write(`approvals: ${d.signatures_collected ?? "?"}/${d.approvals_required ?? "?"}\n`);
      process.stdout.write(`status:    ${d.status ?? "?"}\n`);
    }
  } catch (err) {
    die(`Error: ${(err as Error).message}`);
  }
}

async function cmdComplianceFrameworks(): Promise<void> {
  const frameworks: Array<[string, string]> = [
    ["eu_ai_act", "EU AI Act, Articles 12 + 26"],
    ["dora", "DORA, Article 17"],
    ["nydfs_500", "NYDFS Part 500"],
    ["colorado_ai", "Colorado AI Act SB 24-205"],
    ["texas_traiga", "Texas TRAIGA HB 149"],
    ["nist_ai_rmf", "NIST AI RMF"],
    ["circia", "CIRCIA Covered Cyber Incident"],
    ["hipaa_security", "HIPAA Security Rule 164.312(b)"],
    ["sec_17a4a", "SEC 17 CFR 240.17a-4(a) - 6 year"],
    ["sec_17a4b", "SEC 17 CFR 240.17a-4(b) - 3 year"],
  ];
  for (const [k, name] of frameworks) {
    process.stdout.write(`${k}\t${name}\n`);
  }
}

async function cmdComplianceExport(args: string[]): Promise<void> {
  const session = parseFlag(args, "session");
  const framework = parseFlag(args, "framework") ?? "eu_ai_act";
  const output = parseFlag(args, "output");
  if (!session || !output) {
    die("Usage: asqav compliance export --session <id> --output <path> [--framework eu_ai_act]");
  }
  ensureApiKey();
  await requireTier("business");
  try {
    const data = await request<unknown>(
      "GET",
      `/export/compliance-bundle?session_id=${encodeURIComponent(session)}&framework=${encodeURIComponent(framework)}`,
    );
    const fs = await import("node:fs/promises");
    await fs.writeFile(output, `${JSON.stringify(data, null, 2)}\n`, "utf8");
    const d = data as { receipt_count?: number; merkle_root?: string };
    process.stdout.write(`wrote ${d.receipt_count ?? "?"} receipts to ${output}\n`);
    if (d.merkle_root) process.stdout.write(`merkle_root: ${d.merkle_root}\n`);
  } catch (err) {
    die(`Error: ${(err as Error).message}`);
  }
}

// === IETF Compliance Receipts profile commands ===

async function readJsonInput(value: string): Promise<unknown> {
  if (value === "-") {
    const chunks: Buffer[] = [];
    for await (const chunk of process.stdin) {
      chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : (chunk as Buffer));
    }
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  }
  const fs = await import("node:fs/promises");
  const text = await fs.readFile(value, "utf8");
  return JSON.parse(text);
}

async function cmdSign(args: string[]): Promise<void> {
  const agentId = parseFlag(args, "agent-id");
  const actionType = parseFlag(args, "action-type");
  if (!agentId || !actionType) {
    die("Usage: asqav sign --agent-id ID --action-type T [--compliance-mode] [--action-json PATH]");
  }
  const complianceMode = hasFlag(args, "compliance-mode");
  const actionRef = parseFlag(args, "action-ref");
  const actionJson = parseFlag(args, "action-json");
  const sandboxState = parseFlag(args, "sandbox-state");
  const iterationId = parseFlag(args, "iteration-id");
  const riskClass = parseFlag(args, "risk-class");
  const incidentClass = parseFlag(args, "incident-class");
  const issuerId = parseFlag(args, "issuer-id");
  const receiptType = parseFlag(args, "receipt-type") ?? "protectmcp:decision";
  const reason = parseFlag(args, "reason");
  // --decision (allow|deny|rate_limit) wins over legacy --policy-decision.
  const decisionFlag = parseFlag(args, "decision");
  const decisionMap: Record<string, string> = {
    allow: "permit",
    deny: "deny",
    rate_limit: "rate_limit",
  };
  const policyDecision =
    (decisionFlag !== undefined ? decisionMap[decisionFlag] : undefined)
    ?? parseFlag(args, "policy-decision")
    ?? "permit";
  const sessionId = parseFlag(args, "session-id");
  const validSecondsStr = parseFlag(args, "valid-seconds");
  const policyArtefactPath = parseFlag(args, "policy-artefact");
  const output = parseFlag(args, "output") ?? "text";

  if ((policyDecision === "deny" || policyDecision === "rate_limit") && !reason) {
    die("Error: --reason is required when --policy-decision is deny or rate_limit.");
  }

  ensureApiKey();
  let context: Record<string, unknown> | undefined;
  if (actionJson) {
    try {
      context = (await readJsonInput(actionJson)) as Record<string, unknown>;
    } catch (err) {
      die(`Error reading action JSON: ${(err as Error).message}`);
    }
  }
  if (policyArtefactPath) {
    try {
      const artefact = await readJsonInput(policyArtefactPath);
      context = { ...(context ?? {}), _policy_artefact: artefact };
    } catch (err) {
      die(`Error reading policy artefact: ${(err as Error).message}`);
    }
  }
  const actionId = parseFlag(args, "action-id");
  if (actionId) {
    context = { ...(context ?? {}), _action_id: actionId };
  }

  try {
    const agent = await Agent.get(agentId);
    if (sessionId) {
      // Internal field used by buildSignBody when the session is bound.
      (agent as unknown as { sessionId: string }).sessionId = sessionId;
    }
    const sig = await agent.sign({
      actionType: actionType,
      context,
      complianceMode,
      actionRef,
      sandboxState: sandboxState as "enabled" | "disabled" | "unavailable" | undefined,
      iterationId,
      riskClass: riskClass as "low" | "medium" | "high" | "unknown" | undefined,
      incidentClass,
      issuerId,
      receiptType: receiptType as
        | "protectmcp:decision"
        | "protectmcp:restraint"
        | "protectmcp:lifecycle",
      reason,
      policyDecision: policyDecision as "permit" | "deny" | "rate_limit",
      validSeconds: validSecondsStr ? Number(validSecondsStr) : undefined,
    });
    if (output === "json") {
      process.stdout.write(`${JSON.stringify(sig, null, 2)}\n`);
      return;
    }
    process.stdout.write(`signature_id: ${sig.signatureId}\n`);
    process.stdout.write(`action_id:    ${sig.actionId}\n`);
    process.stdout.write(`algorithm:    ${sig.algorithm ?? "?"}\n`);
    process.stdout.write(`signed_at:    ${sig.timestamp}\n`);
    if (sig.policyDigest) process.stdout.write(`policy_digest: ${sig.policyDigest}\n`);
    if (sig.complianceMode) {
      process.stdout.write(`compliance_mode: true\n`);
      if (sig.receiptType) process.stdout.write(`receipt_type:  ${sig.receiptType}\n`);
      if (sig.actionRef) process.stdout.write(`action_ref:    ${sig.actionRef}\n`);
      if (sig.previousReceiptHash) {
        process.stdout.write(`previous_receipt_hash: ${sig.previousReceiptHash}\n`);
      }
      if (sig.decision) {
        process.stdout.write(`decision:      ${sig.decision}\n`);
      }
    }
    process.stdout.write(`verify_url:   ${sig.verificationUrl}\n`);
  } catch (err) {
    die(`Error signing: ${(err as Error).message}`);
  }
}

async function cmdAuditPackExport(args: string[]): Promise<void> {
  const start = parseFlag(args, "start");
  const end = parseFlag(args, "end");
  const outputFile = parseFlag(args, "output-file");
  const organizationId = parseFlag(args, "organization-id");
  const onlyCompliance = !hasFlag(args, "no-only-compliance");
  if (!start || !end || !outputFile) {
    die("Usage: asqav audit-pack export --start ISO --end ISO --output-file PATH [--organization-id ID] [--no-only-compliance]");
  }
  ensureApiKey();
  try {
    const body: Record<string, unknown> = { start, end, only_compliance: onlyCompliance };
    if (organizationId) body.organization_id = organizationId;
    const bundle = await request<{
      receipt_count?: number;
      bundle_digest?: string;
      bundle_signature_algorithm?: string;
    }>("POST", "/audit-pack/export", body);
    const fs = await import("node:fs/promises");
    await fs.writeFile(outputFile, `${JSON.stringify(bundle, null, 2)}\n`, "utf8");
    process.stdout.write(`wrote ${bundle.receipt_count ?? "?"} receipts to ${outputFile}\n`);
    if (bundle.bundle_digest) process.stdout.write(`bundle_digest: ${bundle.bundle_digest}\n`);
    if (bundle.bundle_signature_algorithm) {
      process.stdout.write(`algorithm:     ${bundle.bundle_signature_algorithm}\n`);
    }
  } catch (err) {
    die(`Error exporting audit pack: ${(err as Error).message}`);
  }
}

async function cmdAuditPackPolicy(args: string[]): Promise<void> {
  let [digest] = positional(args);
  if (!digest) die("Usage: asqav audit-pack policy <sha256:hex>");
  if (!digest.startsWith("sha256:")) {
    if (/^[0-9a-fA-F]{64}$/.test(digest)) {
      digest = `sha256:${digest.toLowerCase()}`;
    } else {
      die("Error: digest must be sha256:<64-hex> or a bare 64-hex string.");
    }
  }
  const output = parseFlag(args, "output") ?? "json";
  ensureApiKey();
  try {
    const data = await request<unknown>("GET", `/audit-pack/policy/${digest}`);
    if (output === "text") {
      const d = data as {
        digest?: string;
        organization_id?: string;
        agent_id?: string;
        action_type?: string;
        created_at?: string;
        artefact?: unknown;
      };
      process.stdout.write(`digest:          ${d.digest ?? "?"}\n`);
      process.stdout.write(`organization_id: ${d.organization_id ?? "?"}\n`);
      process.stdout.write(`agent_id:        ${d.agent_id ?? "?"}\n`);
      process.stdout.write(`action_type:     ${d.action_type ?? "?"}\n`);
      process.stdout.write(`created_at:      ${d.created_at ?? "?"}\n`);
      process.stdout.write(`--- artefact ---\n${JSON.stringify(d.artefact ?? {}, null, 2)}\n`);
    } else {
      process.stdout.write(`${JSON.stringify(data, null, 2)}\n`);
    }
  } catch (err) {
    die(`Error fetching artefact: ${(err as Error).message}`);
  }
}

async function cmdPayloadsErase(args: string[]): Promise<void> {
  const [signatureId] = positional(args);
  if (!signatureId) die("Usage: asqav payloads erase <signature_id> [--yes]");
  if (!hasFlag(args, "yes") && !hasFlag(args, "y")) {
    die("Refusing to erase without --yes. Erasure is idempotent but cannot be undone.");
  }
  ensureApiKey();
  try {
    await request("DELETE", `/signatures/payloads/${signatureId}`);
    process.stdout.write(`Payload erased for ${signatureId}.\n`);
    process.stdout.write(`Tombstone written; signature remains cryptographically verifiable.\n`);
  } catch (err) {
    die(`Error erasing payload: ${(err as Error).message}`);
  }
}

async function cmdOrgSetComplianceStrict(args: string[]): Promise<void> {
  const [orgId] = positional(args);
  if (!orgId) die("Usage: asqav org set-compliance-strict <org_id> --enable|--disable");
  const enable = hasFlag(args, "enable");
  const disable = hasFlag(args, "disable");
  if (enable === disable) {
    die("Error: pass exactly one of --enable or --disable.");
  }
  ensureApiKey();
  try {
    const out = await request<{ compliance_mode_strict?: boolean }>(
      "PATCH",
      `/orgs/${orgId}`,
      { compliance_mode_strict: enable },
    );
    const state = enable ? "enabled" : "disabled";
    process.stdout.write(`compliance_mode_strict ${state} for org ${orgId}.\n`);
    if (out && typeof out.compliance_mode_strict === "boolean") {
      process.stdout.write(
        `server confirms: compliance_mode_strict=${out.compliance_mode_strict}\n`,
      );
    }
  } catch (err) {
    die(`Error updating organization: ${(err as Error).message}`);
  }
}

async function cmdKeysGenerate(args: string[]): Promise<void> {
  const algorithm = parseFlag(args, "algorithm") ?? "ed25519";
  const out = parseFlag(args, "out");
  if (algorithm === "ml-dsa-65") {
    die(
      "ml-dsa-65 keypair generation runs server-side; call `asqav agents create --algorithm ml-dsa-65`.",
    );
  }
  if (algorithm !== "ed25519" && algorithm !== "es256") {
    die(`unsupported_algorithm: ${algorithm}. Supported: ed25519, es256.`);
  }
  try {
    const kp = generateKeypair(algorithm as LocalSigningAlgorithm);
    if (out) {
      const fs = await import("node:fs/promises");
      // Write the PKCS#8 DER bytes (base64-decoded) so the file is the
      // raw key material the cloud signing path expects.
      await fs.writeFile(out, Buffer.from(kp.privateKeyPkcs8B64, "base64"), { mode: 0o600 });
      process.stdout.write(`Private key (PKCS#8 DER) written to ${out} (mode 0600).\n`);
    }
    process.stdout.write(`--- public key (SPKI DER, base64) ---\n${kp.publicKeySpkiB64}\n`);
  } catch (err) {
    die(`Error generating keypair: ${(err as Error).message}`);
  }
}

async function cmdReplayVerify(args: string[]): Promise<void> {
  const [agentId, sessionId] = positional(args);
  if (!agentId || !sessionId) {
    die("Usage: asqav replay-verify <agent_id> <session_id> [--strict] [--output text|json]");
  }
  const strict = hasFlag(args, "strict");
  const output = parseFlag(args, "output") ?? "text";
  ensureApiKey();
  await requireTier("pro");
  try {
    const data = await request<{
      steps?: Array<{
        signature_id?: string;
        signed_envelope?: Record<string, unknown>;
        previous_receipt_hash?: string;
        previousReceiptHash?: string;
      }>;
    }>("GET", `/agents/${agentId}/sessions/${sessionId}/replay`);
    const steps = Array.isArray(data.steps) ? data.steps : [];
    if (steps.length === 0) {
      die("Error: no replay steps returned by the cloud.");
    }
    const legacySteps = steps.filter((s) => !s.signed_envelope);
    const records = steps
      .filter((s) => !!s.signed_envelope)
      .map((s) => ({ signedEnvelope: s.signed_envelope as Record<string, unknown> }));
    const result = verifyChain(records);
    const chainValid = result.chainIntegrity && legacySteps.length === 0;
    const strictPass = strict ? chainValid : result.chainIntegrity;
    if (output === "json") {
      process.stdout.write(
        `${JSON.stringify(
          {
            compliance_chain_valid: chainValid,
            strict,
            strict_pass: strictPass,
            step_count: steps.length,
            legacy_step_count: legacySteps.length,
            steps: result.steps,
          },
          null,
          2,
        )}\n`,
      );
    } else {
      process.stdout.write(`compliance_chain_valid: ${chainValid}\n`);
      process.stdout.write(`strict: ${strict}\n`);
      process.stdout.write(`steps: ${steps.length} (legacy: ${legacySteps.length})\n`);
      for (const step of result.steps) {
        const ok = step.chainValid ? "ok  " : "FAIL";
        process.stdout.write(`  [${ok}] step ${step.index}\n`);
      }
      if (strict && legacySteps.length > 0) {
        process.stdout.write(
          `strict mode FAILED: ${legacySteps.length} step(s) lack signed_envelope.\n`,
        );
      }
    }
    if (!strictPass) process.exit(1);
  } catch (err) {
    die(`Error: ${(err as Error).message}`);
  }
}

const SUPPORTED_MIGRATIONS = new Set(["v3-20", "v3-21", "v3-22"]);

async function cmdMigrateRun(args: string[]): Promise<void> {
  const [migration] = positional(args);
  if (!migration) die("Usage: asqav migrate run v3-20|v3-21|v3-22");
  if (!SUPPORTED_MIGRATIONS.has(migration)) {
    die(`Error: unsupported migration ${migration}. Supported: ${[...SUPPORTED_MIGRATIONS].join(", ")}`);
  }
  const maintenanceKey = process.env.ASQAV_MAINTENANCE_KEY;
  if (!maintenanceKey) {
    die("Error: ASQAV_MAINTENANCE_KEY env var is required for migration commands.");
  }
  if (!process.env.ASQAV_API_KEY) {
    die("Error: ASQAV_API_KEY env var is required.");
  }
  const baseUrl = process.env.ASQAV_API_URL ?? "https://api.asqav.com/api/v1";
  const url = `${baseUrl.replace(/\/+$/, "")}/maintenance/run-migration-${migration}`;
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "X-API-Key": process.env.ASQAV_API_KEY,
        "X-Maintenance-Key": maintenanceKey,
        "Content-Type": "application/json",
      },
      body: "{}",
    });
    const text = await response.text();
    if (response.status >= 400) {
      die(`HTTP ${response.status} from /maintenance/run-migration-${migration}: ${text}`);
    }
    try {
      const parsed = JSON.parse(text);
      process.stdout.write(`${JSON.stringify(parsed, null, 2)}\n`);
    } catch {
      process.stdout.write(`${text}\n`);
    }
  } catch (err) {
    die(`Network error: ${(err as Error).message}`);
  }
}

// === Help / usage ===

function printHelp(): void {
  process.stdout.write(`asqav - AI agent governance CLI (TypeScript build)

Usage:
  asqav --version
  asqav verify <signature_id> [--output text|json]
  asqav sign --agent-id ID --action-type T [--compliance-mode] [--action-json PATH|-]
             [--receipt-type protectmcp:decision|...] [--risk-class low|medium|high|unknown]
             [--sandbox-state enabled|disabled|unavailable] [--iteration-id ID]
             [--issuer-id LEGAL] [--policy-decision permit|deny|rate_limit]
             [--decision allow|deny|rate_limit] [--reason CODE]
             [--algorithm ml-dsa-65|ed25519|es256] [--policy-artefact PATH|-]
             [--session-id ID] [--valid-seconds N] [--output text|json]
  asqav agents list / create <name> / revoke <agent_id>
  asqav sessions list / end <session_id>
  asqav replay <agent_id> <session_id> [--json]            (Pro)
  asqav replay-verify <agent_id> <session_id> [--strict]   (Pro)  IETF chain
  asqav preflight <agent_id> <action_type> [--json]        (Pro)
  asqav budget check / record                              (Pro)
  asqav approve <session_id> <entity_id>                   (Pro)
  asqav compliance frameworks / export                     (Business)
  asqav audit-pack export --start ISO --end ISO --output-file PATH
                          [--organization-id ID] [--no-only-compliance]
  asqav audit-pack policy <sha256:hex> [--output text|json]
  asqav payloads erase <signature_id> --yes                P4 right-to-erasure
  asqav org set-compliance-strict <org_id> --enable|--disable
  asqav keys generate --algorithm ed25519|es256 [--out priv.pem]
  asqav migrate run v3-20|v3-21|v3-22                      X-Maintenance-Key required

Set ASQAV_API_KEY to authenticate. Get a key at https://asqav.com.
`);
}

// === Dispatcher ===

export async function runCli(argv: string[]): Promise<void> {
  const [first, ...rest] = argv;
  if (!first || first === "-h" || first === "--help") {
    printHelp();
    return;
  }
  if (first === "-v" || first === "--version") {
    process.stdout.write(`asqav ${CLI_VERSION}\n`);
    return;
  }

  switch (first) {
    case "verify":
      return cmdVerify(rest);
    case "sign":
      return cmdSign(rest);
    case "agents": {
      const [sub, ...rest2] = rest;
      if (sub === "list") return cmdAgentsList();
      if (sub === "create") return cmdAgentsCreate(rest2);
      if (sub === "revoke") return cmdAgentRevoke(rest2);
      die("Usage: asqav agents (list | create <name> | revoke <agent_id>)");
    }
    case "sessions": {
      const [sub, ...rest2] = rest;
      if (sub === "list") return cmdSessionsList(rest2);
      if (sub === "end") return cmdSessionsEnd(rest2);
      die("Usage: asqav sessions (list | end <session_id>)");
    }
    case "replay":
      return cmdReplay(rest);
    case "replay-verify":
      return cmdReplayVerify(rest);
    case "preflight":
      return cmdPreflight(rest);
    case "budget": {
      const [sub, ...rest2] = rest;
      if (sub === "check") return cmdBudgetCheck(rest2);
      if (sub === "record") return cmdBudgetRecord(rest2);
      die("Usage: asqav budget (check | record)");
    }
    case "approve":
      return cmdApprove(rest);
    case "compliance": {
      const [sub, ...rest2] = rest;
      if (sub === "frameworks") return cmdComplianceFrameworks();
      if (sub === "export") return cmdComplianceExport(rest2);
      die("Usage: asqav compliance (frameworks | export)");
    }
    case "audit-pack": {
      const [sub, ...rest2] = rest;
      if (sub === "export") return cmdAuditPackExport(rest2);
      if (sub === "policy") return cmdAuditPackPolicy(rest2);
      die("Usage: asqav audit-pack (export | policy <digest>)");
    }
    case "payloads": {
      const [sub, ...rest2] = rest;
      if (sub === "erase") return cmdPayloadsErase(rest2);
      die("Usage: asqav payloads erase <signature_id> --yes");
    }
    case "org": {
      const [sub, ...rest2] = rest;
      if (sub === "set-compliance-strict") return cmdOrgSetComplianceStrict(rest2);
      die("Usage: asqav org set-compliance-strict <org_id> --enable|--disable");
    }
    case "keys": {
      const [sub, ...rest2] = rest;
      if (sub === "generate") return cmdKeysGenerate(rest2);
      die("Usage: asqav keys generate --algorithm ed25519|es256 [--out PATH]");
    }
    case "migrate": {
      const [sub, ...rest2] = rest;
      if (sub === "run") return cmdMigrateRun(rest2);
      die("Usage: asqav migrate run v3-20|v3-21|v3-22");
    }
    default:
      die(`Unknown command: ${first}\nRun 'asqav --help' for usage.`);
  }
}

// Entry point when invoked via the bin shim. The compiled bin file is
// `dist/cli.js`; re-export the runner so the bin script can call it.
if (process.argv[1]?.endsWith("/asqav") || process.argv[1]?.endsWith("/cli.js") || process.argv[1]?.endsWith("/cli.mjs")) {
  runCli(process.argv.slice(2)).catch((err) => die(`Error: ${(err as Error).message}`));
}
