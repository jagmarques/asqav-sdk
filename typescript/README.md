# asqav

TypeScript SDK for [asqav.com](https://asqav.com). All ML-DSA cryptography runs server-side. Zero native dependencies.

## Install

```bash
npm install @asqav/sdk
```

## CLI

The package ships an `asqav` binary mirroring the Python CLI surface. Set `ASQAV_API_KEY` and run:

```bash
asqav --version
asqav verify <signature_id>
asqav agents list
asqav agents create my-agent
asqav agents revoke <agent_id>
asqav sessions list [--limit N] [--status X] [--agent ID]
asqav sessions end <session_id>
asqav replay <agent_id> <session_id> [--json]              # Pro
asqav preflight <agent_id> <action_type> [--json]          # Pro
asqav budget check --agent-id ID --limit 10 --estimated-cost 0.25     # Pro
asqav budget record --agent-id ID --action api:openai --actual-cost 0.23 --limit 10  # Pro
asqav approve <session_id> <entity_id>                     # Pro
asqav compliance frameworks
asqav compliance export --session ID --output bundle.json  # Business
```

Pro/Business commands are gated client-side via `GET /account` so a free-tier key gets a clean upgrade message instead of a mid-pipeline 402. The server is the source of truth; older self-hosted deployments without `/account` skip the gate.

## Quick start

```ts
import { init, Agent } from "@asqav/sdk";

init({ apiKey: process.env.ASQAV_API_KEY });

const agent = await Agent.create({ name: "my-agent" });
const sig = await agent.sign({
  actionType: "api:call",
  context: { model: "gpt-4" },
});

console.log(sig.verificationUrl);
```

Each signed action is recorded server-side with an ML-DSA (FIPS 204) signature, a chain hash, and a public verification URL.

## Data handling modes

The SDK auto-detects whether you're pointing at the Asqav cloud or a self-hosted deployment, and selects the safer default for each:

- **Cloud (`*.asqav.com`)**: hash-only by default. The SDK builds a fingerprint of your action context, computes a SHA-256 hash locally, and sends only the hash plus a small metadata bag (action_type, agent_id, session_id, model_name, tool_name). Raw prompts and tool arguments stay on your side.
- **Self-hosted**: full-payload by default. The server can run policy checks, PII redaction, and richer audit. Recommended when you control the deployment.

Override anytime:

```ts
await init({ apiKey: "...", baseUrl: "https://api.asqav.com", mode: "hash-only" });
```

The fingerprint format is RFC 8785-style sorted JSON with no whitespace, hashed with SHA-256. See `docs/fingerprint-spec.md` and `conformance/vectors.json` for the spec and cross-language test vectors.

## Why

Without governance, an AI agent can do anything and leave no trace. With asqav, every action is signed, every policy is enforced in real time, and every audit query returns a verifiable record.

## API surface

```ts
import {
  init,
  Agent,
  verifySignature,
  listSessions,
  exportAuditJson,
  AsqavError,
  AuthenticationError,
  RateLimitError,
  APIError,
} from "@asqav/sdk";
```

### `init({ apiKey, baseUrl })`

Sets the API key for subsequent calls. Falls back to `process.env.ASQAV_API_KEY`. The default base URL is `https://api.asqav.com/api/v1`.

### `Agent.create({ name, algorithm?, capabilities? })`

Creates a new agent. The server generates the ML-DSA keypair; the private key never leaves the server.

### `Agent.get(agentId)`

Fetches an existing agent.

### `agent.sign({ actionType, context? })`

Signs an action. Returns a `SignatureResponse` with the signature bytes, signature ID, chain hash, and verification URL.

### `agent.startSession()` / `agent.endSession({ status? })`

Bracket related signings into a session for replay and compliance grouping.

### `agent.revoke({ reason? })`

Revokes the agent's credentials globally. Irreversible.

### `agent.preflight(actionType)`

Pre-flight check combining revocation/suspension status and policy. Returns `{ cleared, agentActive, policyAllowed, reasons, explanation }`. Fail-open: a sub-check that errors records a warning in `reasons` but does not block.

```ts
const decision = await agent.preflight("data:read");
if (!decision.cleared) {
  throw new Error(decision.explanation);
}
```

### `BudgetTracker`

Client-side spend budget. Each recorded spend is signed via the agent so the budget trail is itself tamper-evident. `check()` is a fail-closed arithmetic check; the SDK does not block actions on its own.

```ts
import { BudgetTracker } from "@asqav/sdk";

const budget = new BudgetTracker({ agent, limit: 10, currency: "USD" });

const decision = budget.check(0.25);
if (!decision.allowed) throw new Error(decision.reason);

await budget.record("api:openai", 0.23, { model: "gpt-4" });
```

### `verifySignature(signatureId)`

Public verification endpoint. No auth required.

### `listSessions({ agentId?, status?, limit? })`

Lists sessions, optionally filtered.

### `exportAuditJson({ startDate?, endDate?, agentId? })`

Exports the audit trail as JSON for the given filters. Pro tier and above.

## Integrations

### Vercel AI SDK

Wire Asqav signing into the [`experimental_telemetry`](https://ai-sdk.dev/docs/ai-sdk-core/telemetry) hook. Every span the AI SDK opens (text generation, tool calls, embeddings) becomes a signed Asqav action.

```ts
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { Agent, init } from "@asqav/sdk";
import { createAsqavExporter } from "@asqav/sdk/extras/vercel-ai";

init({ apiKey: process.env.ASQAV_API_KEY! });
const agent = await Agent.create({ name: "writer" });
const tracer = createAsqavExporter({ agent });

const result = await generateText({
  model: openai("gpt-4o"),
  prompt: "Write a haiku about audit trails.",
  experimental_telemetry: { isEnabled: true, tracer },
});
```

Span names are mapped to Asqav action types: `ai.generateText` -> `ai:completion`, `ai.streamText` -> `ai:completion_stream`, `ai.toolCall` -> `ai:tool_call`, errors -> `ai:error`. Signing is fire-and-forget and never blocks generation; failures log a warning instead of throwing.

## Errors

All thrown errors extend `AsqavError`. `AuthenticationError`, `RateLimitError`, and `APIError` (with `statusCode`) are exported for fine-grained handling.

## Requirements

Node 18 or newer. Uses the built-in `fetch`. No transitive runtime dependencies.

## License

MIT. Get an API key at [asqav.com](https://asqav.com).
