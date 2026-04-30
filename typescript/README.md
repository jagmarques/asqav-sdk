# asqav

TypeScript SDK for [asqav.com](https://asqav.com). All ML-DSA cryptography runs server-side. Zero native dependencies.

## Install

```bash
npm install @asqav/sdk
```

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

### `verifySignature(signatureId)`

Public verification endpoint. No auth required.

### `listSessions({ agentId?, status?, limit? })`

Lists sessions, optionally filtered.

### `exportAuditJson({ startDate?, endDate?, agentId? })`

Exports the audit trail as JSON for the given filters. Pro tier and above.

## Errors

All thrown errors extend `AsqavError`. `AuthenticationError`, `RateLimitError`, and `APIError` (with `statusCode`) are exported for fine-grained handling.

## Requirements

Node 18 or newer. Uses the built-in `fetch`. No transitive runtime dependencies.

## License

MIT. Get an API key at [asqav.com](https://asqav.com).
