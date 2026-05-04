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
  init,
  listSessions,
  request,
  verifySignature,
} from "./index.js";

const CLI_VERSION = "0.2.5";
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

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

async function cmdVerify(args: string[]): Promise<void> {
  const [sigId] = positional(args);
  if (!sigId) die("Usage: asqav verify <signature_id> [--output text|json]");
  const output = parseFlag(args, "output") ?? "text";
  // Verify is public (no auth required) but the underlying request
  // helper short-circuits on missing config; init with whatever key is
  // available so the public endpoint still works.
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
  // Static list mirrors python/src/asqav/compliance.py FRAMEWORKS keys.
  const frameworks: Array<[string, string]> = [
    ["eu_ai_act_art12", "EU AI Act, Article 12 - Logging"],
    ["eu_ai_act_art14", "EU AI Act, Article 14 - Human oversight"],
    ["dora_ict", "DORA - ICT operational resilience"],
    ["soc2", "SOC 2 - CC7.2 anomaly logging"],
  ];
  for (const [k, name] of frameworks) {
    process.stdout.write(`${k}\t${name}\n`);
  }
}

async function cmdComplianceExport(args: string[]): Promise<void> {
  const session = parseFlag(args, "session");
  const framework = parseFlag(args, "framework") ?? "eu_ai_act_art12";
  const output = parseFlag(args, "output");
  if (!session || !output) {
    die("Usage: asqav compliance export --session <id> --output <path> [--framework eu_ai_act_art12]");
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

// ---------------------------------------------------------------------------
// Help / usage
// ---------------------------------------------------------------------------

function printHelp(): void {
  process.stdout.write(`asqav - AI agent governance CLI (TypeScript build)

Usage:
  asqav --version
  asqav verify <signature_id>
  asqav agents list
  asqav agents create <name>
  asqav agents revoke <agent_id> [--reason X]
  asqav sessions list [--limit N] [--status X] [--agent ID]
  asqav sessions end <session_id> [--status completed]
  asqav replay <agent_id> <session_id> [--json]            (Pro)
  asqav preflight <agent_id> <action_type> [--json]        (Pro)
  asqav budget check --agent-id ID --limit N --estimated-cost N
                     [--current-spend N] [--currency USD] [--json]   (Pro)
  asqav budget record --agent-id ID --action TYPE --actual-cost N --limit N
                      [--current-spend N] [--currency USD]           (Pro)
  asqav approve <session_id> <entity_id> [--json]          (Pro)
  asqav compliance frameworks
  asqav compliance export --session ID --output PATH [--framework KEY]  (Business)

Set ASQAV_API_KEY to authenticate. Get a key at https://asqav.com.
`);
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

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
    default:
      die(`Unknown command: ${first}\nRun 'asqav --help' for usage.`);
  }
}

// Entry point when invoked via the bin shim. The compiled bin file is
// `dist/cli.js`; re-export the runner so the bin script can call it.
if (process.argv[1]?.endsWith("/asqav") || process.argv[1]?.endsWith("/cli.js") || process.argv[1]?.endsWith("/cli.mjs")) {
  runCli(process.argv.slice(2)).catch((err) => die(`Error: ${(err as Error).message}`));
}
