# asqav

[![GitHub stars](https://img.shields.io/github/stars/jagmarques/asqav-sdk?style=social)](https://github.com/jagmarques/asqav-sdk)

TypeScript SDK for [asqav.com](https://asqav.com), the evidence layer for AI agents. Sign every agent action with ML-DSA-65 (FIPS 204), enforce policies before execution, and produce regulator-ready audit trails. All cryptography runs server-side, and the package has zero native dependencies.

## Install

```bash
npm install @asqav/sdk
```

## Quick start

Get an API key at [asqav.com](https://asqav.com), then sign your first agent action:

```ts
import { govern } from "@asqav/sdk";

// govern() is a one-call setup: init() + Agent.create() in one line
const agent = await govern({ apiKey: process.env.ASQAV_API_KEY, agentName: "my-agent" });

const sig = await agent.sign({ actionType: "api:openai:chat", context: { model: "gpt-4o" } });

console.log(sig.actionRef);             // "sha256:..." over the JCS-canonical action
console.log(sig.previousReceiptHash);   // 64 hex; "0".repeat(64) on the first record per agent
console.log(sig.verificationUrl);
```

That's it. One install, one govern call, one sign call. The receipt lands on the Asqav cloud under [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/): ML-DSA-65 (FIPS 204) signature, chain hash, retained `policy_digest`, fail-closed anchoring, and a public verification URL.

Prefer the lower-level building blocks? `init({ apiKey })` followed by `Agent.create({ name })` still works and gives more control over `algorithm`, `capabilities`, and other agent options.

### No account? Offline path

TypeScript does not ship a `local_sign` queue equivalent. For offline/air-gapped scenarios use the Python SDK's `local_sign` helper, or set `mode: "hash-only"` to minimize what leaves the process while still reaching the cloud:

```ts
await init({ apiKey: "...", baseUrl: "https://api.asqav.com", mode: "hash-only" });
```

Pass `complianceMode: false` on any `agent.sign(...)` call if you want a non-Compliance receipt.

## CLI

The package ships an `asqav` binary mirroring the Python CLI surface. Set `ASQAV_API_KEY` and run:

```bash
asqav verify <signature_id> [--output json]   # IETF axes when present
asqav sign --agent-id ID --action-type T --action-json action.json \
           --receipt-type protectmcp:decision \
           --risk-class high --issuer-id legal:Acme
           # add --no-compliance-mode to opt out of the IETF profile (default is on)
asqav agents list / create / revoke
asqav sessions list / end <session_id>
asqav replay <agent_id> <session_id> [--json]
asqav replay-verify <agent_id> <session_id> [--strict]     # IETF chain
asqav preflight <agent_id> <action_type> [--json]
asqav budget check / record
asqav approve <session_id> <entity_id>
asqav compliance frameworks / export
asqav audit-pack export --start ISO --end ISO --output-file bundle.json
asqav audit-pack policy <sha256:hex>
asqav payloads erase <signature_id> --yes                  # P4 right-to-erasure
asqav org set-compliance-strict <org_id> --enable|--disable
asqav keys generate --algorithm ed25519|es256 [--out priv.bin]
asqav migrate run v3-20|v3-21|v3-22                        # X-Maintenance-Key required
```

Every command listed here works on the free tier. The Asqav cloud is the source of truth for what your key may do.

## Data handling modes

The SDK auto-detects whether you're pointing at the Asqav cloud or a self-hosted deployment and selects the safer default for each:

- **Cloud (`*.asqav.com`)**: hash-only by default. The SDK builds a fingerprint of your action context, computes a SHA-256 hash locally, and sends only the hash plus a small metadata bag (`action_type`, `agent_id`, `session_id`, `model_name`, `tool_name`). Raw prompts and tool arguments stay on your side.
- **Self-hosted**: full-payload by default. The server can run policy checks, PII redaction, and richer audit. Recommended when you control the deployment.

Override anytime:

```ts
await init({ apiKey: "...", baseUrl: "https://api.asqav.com", mode: "hash-only" });
```

The fingerprint format is sorted JSON with no whitespace per JCS, hashed with SHA-256. See `docs/fingerprint-spec.md` and `conformance/vectors.json` for the spec and cross-language test vectors.

## Advanced / high-value actions

Once you have the basics working, you can pass compliance envelope fields for regulated or high-risk actions. The full parameter set is useful for things like large financial transfers, healthcare decisions, or anything that needs an auditor-facing trail.

```ts
const sig = await agent.sign({
  actionType: "payment.wire_transfer",
  context: { amountEur: 850000, beneficiaryIban: "DE89370400440532013000" },
  receiptType: "protectmcp:decision",
  riskClass: "high",
  issuerId: "legal:Acme GmbH",
  iterationId: "task-2026-Q2-4821",
});
```

The envelope extensions most callers reach for, camelCase on the SDK and snake_case on the JSON wire:

- `receiptType` -> `receipt_type` - `protectmcp:decision`, `protectmcp:restraint`, `protectmcp:lifecycle`, `protectmcp:lifecycle:configuration_change`, `protectmcp:acknowledgment`, `protectmcp:observation`, or `protectmcp:observation:result_bound`, which is an observation receipt that binds tool output via `resultDigest`.
- `riskClass` -> `risk_class` - controlled vocabulary: `low | medium | high | unknown`.
- `iterationId` -> `iteration_id` - logical task id, distinct from session.
- `sandboxState` -> `sandbox_state` - `enabled | disabled | unavailable` for high-risk gating.
- `incidentClass` -> `incident_class` - DORA / NYDFS / CIRCIA token, or an array of tokens.
- `issuerId` -> `issuer_id` - LEI per ISO 17442, EIN, CIK, or a W3C DID for non-LEI deployers.

## Compliance receipts: the IETF profile

Compliance Receipts are the SDK default. Each `agent.sign(...)` call produces a receipt that conforms to [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/): ML-DSA-65 signature, JCS canonicalization, retained `policy_digest`, hash-chained `previous_receipt_hash`, OpenTimestamps anchoring. Opt out with `complianceMode: false` if you want the older shape.

### Shadow AI capture with passive_telemetry

Two `receiptType` values cover the gating axis: `protectmcp:decision` records that a policy ran and gated the action. `protectmcp:observation` records that a passive monitor saw the event without gating it. Pick `observation` when the producer never had the option to block, such as a SIEM forwarder, a browser extension in observe-only mode, or a NetFlow-style proxy with no enforcement hook.

Set `captureTopology: "passive_telemetry"` to declare the producer is observing after the fact. The SDK client-side check pre-flights the Asqav cloud's full rule 8 gate: a `captureTopology: "passive_telemetry"` receipt MUST use `receiptType: "protectmcp:observation"`. Any other receiptType paired with `passive_telemetry`, namely `:decision`, `:restraint`, `:lifecycle`, `:lifecycle:configuration_change`, or `:acknowledgment`, throws `AsqavError` with the verbatim `false_attestation_guard: capture_topology=passive_telemetry receipts must use receipt_type=protectmcp:observation[:result_bound], not :<offending> (rule 8)` message before the HTTP roundtrip. The guard lives as `false_attestation_guard` in `typescript/src/index.ts`.

```ts
const sig = await agent.sign({
  actionType: "mcp:tool_call",
  context: { server: "filesystem", tool: "read" },
  receiptType: "protectmcp:observation",
  captureTopology: "passive_telemetry",
  issuerId: "legal:Acme GmbH",
});
```

`captureTopology` is stamped on the audit-pack manifest entry but never on the signed payload. The other accepted topologies are `in_process_sdk`, `network_proxy`, `browser_extension`, `ebpf_observer`, and `mcp_proxy`. Only `passive_telemetry` triggers the false-attestation guard. The full topology semantics live in the cloud's `docs/capture-topology.md`, and the wire vocabulary is published live at `https://api.asqav.com/.well-known/governance.json` for discovery.

### Configuration change receipts, rule 9

A `receiptType: "protectmcp:lifecycle:configuration_change"` receipt declares the agent's runtime configuration was mutated. The SDK pre-flights the Asqav cloud's rule 9 cross-field gate for NSA CSI U/OO/6030316-26 alignment: the receipt MUST carry `configManifestDigest`. Omitting it throws `AsqavError` with the verbatim `false_attestation_guard: receipt_type=protectmcp:lifecycle:configuration_change requires config_manifest_digest (rule 9)` message before the HTTP roundtrip.

```ts
const sig = await agent.sign({
  actionType: "mcp:config_update",
  context: { server: "filesystem", delta: "tool_added" },
  receiptType: "protectmcp:lifecycle:configuration_change",
  policyDecision: "none",
  configManifestDigest: "sha256:<hex of manifest>",
  cveInventoryDigest: "sha256:<hex of cve snapshot>",
});
```

### Expiry precedence, rule 10

The legacy `validSeconds`, where the server computes `valid_until = signed_at + valid_seconds`, and the caller-supplied horizon `expiresAt` are mutually exclusive. Pass exactly one of the two: `validSeconds: 3600` for "expire one hour after signing", or `expiresAt: "2026-06-01T00:00:00Z"` for an explicit horizon. Passing both throws `AsqavError` with the verbatim `expiry_collision_guard: pass either valid_seconds or expires_at, not both (rule 10)` message before the HTTP roundtrip. Passing neither falls back to the server-side default of `valid_seconds=86400`.

### Digest format, rule 11

Every caller-supplied self-describing digest field, namely `configManifestDigest`, `cveInventoryDigest`, `executableHash`, and `sbomDigest`, MUST match the regex `^sha256:[a-f0-9]{64}$`. `toolFingerprint` is the exception: it uses the cloud wire form of 32 bare lowercase hex chars, SHA-256[:32], matching `^[0-9a-f]{32}$` with no `sha256:` prefix. Anything else throws `AsqavError` with the verbatim `tool_fingerprint_not_32_hex_chars: must be 32 lowercase hex chars (SHA-256[:32]).` message before the HTTP roundtrip. To avoid wire drift, use the SDK's deterministic helpers, each byte-deterministic under JCS:

```ts
import {
  computeToolFingerprint,
  computeConfigManifestDigest,
  computeCveInventoryDigest,
} from "@asqav/sdk";

const fp = computeToolFingerprint("search", { args: { q: "string" } });
const cfg = computeConfigManifestDigest({ server: "filesystem", tools: ["read"] });
const cve = computeCveInventoryDigest([{ id: "CVE-2026-0001", severity: "high" }]);
```

## NSA-aligned receipt fields

Six wire fields on `agent.sign(...)` carry the NSA CSI U/OO/6030316-26 alignment for MCP server lifecycle and tool output binding:

- `resultDigest` - `sha256:<hex>` of the tool output, binds the receipt to a specific result. See <https://www.asqav.com/docs/result-digest>.
- `expiresAt` - explicit ISO-8601 or POSIX validity horizon. Converted client-side to `validSeconds` and sent as a duration, since the cloud owns absolute time-binding. Mutually exclusive with `validSeconds`. See <https://www.asqav.com/docs/expires-at>.
- `nonce` - 12 random bytes auto-generated when omitted. Cloud rejects duplicates inside the validity window. See <https://www.asqav.com/docs/nonce>.
- `toolFingerprint` - 32 bare lowercase hex chars, SHA-256[:32], over `{tool_name, schema}`, auto-derived when `toolName` + `toolSchema` are present. See <https://www.asqav.com/docs/tool-fingerprint>.
- `configManifestDigest` - `sha256:<hex>` of the agent's runtime configuration snapshot. Required on configuration_change receipts. See <https://www.asqav.com/docs/config-manifest-digest>.
- `cveInventoryDigest` - `sha256:<hex>` over the CVE snapshot at sign time. See <https://www.asqav.com/docs/cve-inventory-digest>.

The `protectmcp:observation:result_bound` `receiptType` variant carries `resultDigest` and lets observation receipts bind to a specific tool result without claiming the policy gated the call.

### Build-provenance 4-tuple

Four optional wire fields bind build-side provenance into the signed receipt, subsuming standalone SBOM or SLSA verifiers in a single envelope. All four are independent and may be set together or individually:

- `executableHash` - `sha256:<hex>` of the executable that invoked the action. MUST match `sha256:<64-hex>`.
- `sbomDigest` - `sha256:<hex>` of the canonical CycloneDX or SPDX SBOM document.
- `slsaProvenancePointer` - https URL to the SLSA attestation envelope for the executing build.
- `supplyChainPointer` - https URL to the in-toto, Sigstore, or Rekor entry covering the build.

```ts
import { Agent } from "@asqav/sdk";

const agent = await Agent.create({ apiKey: process.env.ASQAV_API_KEY! });
const receipt = await agent.sign({
  actionType: "tool.invoke",
  toolName: "exec_query",
  executableHash: "sha256:" + "a".repeat(64),
  sbomDigest: "sha256:" + "b".repeat(64),
  slsaProvenancePointer: "https://example.com/slsa/build-123.json",
  supplyChainPointer: "https://rekor.sigstore.dev/api/v1/log/entries/abc",
});
```

See <https://www.asqav.com/docs/executable-hash-and-sbom-provenance> for the field semantics and a worked policy example.

## Optional threat-framework mappings

Seven optional wire fields let a caller pin the receipt to industry threat-and-control taxonomies. Each list is caller-supplied and Asqav-preserved verbatim. The cloud sets `framework_mappings_self_declared=true` on the receipt whenever any of the six list fields is populated, so verifiers can tell self-declared classifications apart from cloud-verified ones.

- `mitreTechniques` - list of MITRE ATT&CK technique ids, for example `["T1059", "T1078"]`.
- `mitreAtlas` - list of MITRE ATLAS ids for AI-system threats, for example `["AML.T0051"]`.
- `owaspLlmTop10` - list of OWASP Top 10 for LLM ids, for example `["LLM01", "LLM02"]`.
- `nistAiRmf` - list of NIST AI RMF function ids, for example `["GOVERN-1.1", "MEASURE-2.7"]`.
- `iso42001` - list of ISO/IEC 42001 control ids, for example `["A.6.2.6"]`.
- `euAiActArticles` - list of EU AI Act article ids, for example `["Article-12", "Article-15"]`.
- `rfc3161Timestamp` - caller-supplied base64-encoded RFC 3161 TimeStampResp in DER.

```ts
const receipt = await agent.sign({
  actionType: "api:call",
  context: { user: "..." },
  complianceMode: true,
  mitreTechniques: ["T1059", "T1078"],
  owaspLlmTop10: ["LLM01"],
  nistAiRmf: ["GOVERN-1.1", "MEASURE-2.7"],
  euAiActArticles: ["Article-12"],
});
```

Each list must be a non-empty array of strings, each entry up to 128 characters. Empty arrays, non-string entries, or oversize entries throw `AsqavError` with the verbatim guard messages `<field>_must_be_non_empty_list` or `<field>_entry_invalid` and skip the HTTP roundtrip. The `rfc3161Timestamp` must be valid base64 or throws `rfc3161_timestamp_not_base64`. Camel-case props project onto snake_case wire keys via the SDK's `IETF_OPTIONAL_FIELD_MAP`.

See <https://www.asqav.com/docs/threat-framework-mapping>.

## Optional witness policy

`witnessPolicy` lets a caller declare an N-of-M durable-anchoring quorum on the receipt. The receipt reaches `witness_quorum_met` only when `required` witnesses hold a real inclusion proof. It projects onto the snake_case `witness_policy` wire key via the SDK's `IETF_OPTIONAL_FIELD_MAP`.

- Shape: `{ required: number, witnesses: Array<"rfc3161" | "opentimestamps"> }`.
- `required` must be an integer in `[1, witnesses.length]`.
- `witnesses` must be a non-empty subset of the two shipped witnesses: `rfc3161` and `opentimestamps`.
- `rekor` is rejected. It is not a shipped witness.
- `rfc3161` confirmation is available on the Enterprise tier only. On the Free tier, `opentimestamps` is the witness that produces an inclusion proof; a policy requiring only `rfc3161` will not reach `witness_quorum_met` on Free.

```typescript
const sig = await agent.sign({
  actionType: "deploy:release",
  context: { build: "..." },
  complianceMode: true,
  witnessPolicy: { required: 1, witnesses: ["rfc3161", "opentimestamps"] },
});
```

Bad input throws `AsqavError` with a verbatim guard message before the HTTP roundtrip: `witness_policy_unknown_witness` for an entry like `rekor`, `witness_policy_required_out_of_range`, `witness_policy_witnesses_must_be_non_empty_list`, `witness_policy_required_must_be_int`, or `witness_policy_duplicate_witness`. Omit the field for today's behaviour.

### Audit Pack export

The SDK exposes the public audit-trail endpoint as `exportAuditJson` for filtered windows of receipts:

```ts
import { exportAuditJson } from "@asqav/sdk";

const bundle = await exportAuditJson({
  startDate: "2026-05-01T00:00Z",
  endDate: "2026-06-01T00:00Z",
  agentId: "agt_abc",
});
console.log(bundle.records.length, bundle.bundleDigest);
```

For offline chain verification, `verifyChain(records)` walks an ordered list of signed envelopes for one agent and re-derives each `previous_receipt_hash`. The first record's seed is `"0".repeat(64)`. The cloud is the authoritative verifier. This helper is a convenience.

Algorithm agility is exposed via `SUPPORTED_ALGORITHMS`. `Agent.create(...)` sends the algorithm to the Asqav cloud, which accepts `ml-dsa-44`, `ml-dsa-65` (default), and `ml-dsa-87`. `ed25519` and `es256` are for local keypair generation only (`generateKeypair("ed25519")`); passing them to `Agent.create(...)` returns a 400 from the cloud. ES256 signatures emit in IEEE-P1363 raw r||s form so they match the cloud verifier byte-for-byte.

## Offline / air-gapped verification

Receipts can be verified without calling the Asqav API once you have a JWKS snapshot.
Full guide: [`docs/offline-verification.md`](../docs/offline-verification.md).

`@noble/post-quantum` ships as a runtime dependency and covers ML-DSA-65 in-process.
Ed25519 and ES256 run via Node's built-in `crypto` module.

Snapshot the JWKS while you have network access, then verify offline:

```ts
import { fetchJwks, verifyReceiptOffline } from "@asqav/sdk";
import { readFileSync, writeFileSync } from "node:fs";

// online: snapshot the keys
const jwks = await fetchJwks();
writeFileSync("jwks.json", JSON.stringify(jwks));

// offline: verify any receipt
const receipt = JSON.parse(readFileSync("receipt.json", "utf8"));
const jwksSnap = JSON.parse(readFileSync("jwks.json", "utf8"));
const result = await verifyReceiptOffline(receipt, jwksSnap);
if (result.verdict !== "PASS") throw new Error(JSON.stringify(result.axes));
```

Supply a predecessor receipt to check the hash-chain link:

```ts
const result = await verifyReceiptOffline(receipt, jwksSnap, { predecessor: prevReceipt });
```

## Integrations

The SDK ships native callbacks and adapters for common agent frameworks:

- **Vercel AI SDK** - import `createAsqavExporter` from `@asqav/sdk/extras/vercel-ai`, pass to `experimental_telemetry: { tracer }`. Every span becomes a signed Asqav action, whether text generation, a tool call, or an embedding.
- **LangChain.js** - import `AsqavCallbackHandler` from `@asqav/sdk/extras/langchain`, construct via `await AsqavCallbackHandler.create({ agent })` then pass to `callbacks: [handler]`. Requires `@langchain/core` as a peer dep.
- **Mastra** - import `AsqavMastraHook` from `@asqav/sdk/extras/mastra`, attach via `await new AsqavMastraHook({ agent }).attach(mastraAgent)`. Requires `@mastra/core` as a peer dep.
- **OpenAI Agents JS** - import `AsqavOpenAIAgentsAdapter` from `@asqav/sdk/extras/openai-agents`, wrap individual tools via `new AsqavOpenAIAgentsAdapter({ agent }).wrapTool(tool)`. Each invocation signs `tool:start` / `tool:end` / `tool:error`. Requires `@openai/agents` as a peer dep.
- **CrewAI**, **LiteLLM**, **LlamaIndex**, **Haystack** - same pattern via the Hooks API. The Python SDK has the richer ecosystem, see <https://asqav.com/docs/integrations> for parity status.

Span names are mapped to Asqav action types: `ai.generateText` -> `ai:completion`, `ai.streamText` -> `ai:completion_stream`, `ai.toolCall` -> `ai:tool_call`, errors -> `ai:error`. Signing is fire-and-forget and never blocks generation. Failures log a warning instead of throwing.

## Errors

All thrown errors extend `AsqavError`. `AuthenticationError` for 401, `RateLimitError` for 429, and `APIError` carrying `statusCode` are exported for fine-grained handling:

```ts
import { AsqavError, AuthenticationError, RateLimitError, APIError } from "@asqav/sdk";

try {
  await agent.sign({ actionType: "api:call", context: { model: "gpt-4" } });
} catch (err) {
  if (err instanceof AuthenticationError) throw err;     // rotate ASQAV_API_KEY
  if (err instanceof RateLimitError) throw err;          // err.retryAfter holds the cooldown
  if (err instanceof APIError) throw err;                // err.statusCode holds the HTTP status
  throw err;
}
```

`TypeError` is thrown before the HTTP roundtrip for the verbatim rule 8 / 9 / 10 / 11 guards listed above. Catch `TypeError` separately so guard violations surface as developer errors, not API errors.

## Neutral verifier

A standalone offline verifier for agent receipts across formats ships as a subpath export:

```ts
import { verify, ADAPTERS } from "@asqav/sdk/verifier";

const result = verify(receipt, ADAPTERS, keyProvider);
// result.verdict is "PASS", "FAIL", or "INCOMPLETE"
```

It verifies the issuer signature over the canonical bytes, the hash-chain link, and structural presence across the asqav-native, AERF, ACTA, agent-receipts, and Authproof formats. It never attests the behaviour of the recorded action, and no account is required. Ed25519 and ES256 verify in-process via `node:crypto`. ML-DSA-65 downgrades to `INCOMPLETE` rather than ever returning a false `PASS`. This is the TypeScript port of the Python `asqav.verifier.oracle`, held to verdict parity by a shared conformance corpus.

## Structured receipts (optional schema)

Pass a `contextSchema` to `agent.sign()` to validate and normalise the
`context` object before it is signed. This keeps every receipt in your audit
trail in a consistent, queryable shape.

```typescript
import { Agent, ValidationError, type ContextSchema, init } from "@asqav/sdk";

const PAYMENT_SCHEMA: ContextSchema = {
  userId:   { type: "string",  required: true },
  amount:   { type: "number",  required: true },
  currency: { type: "string",  required: true },
};

init({ apiKey: "sk_..." });
const agent = await Agent.create({ name: "my-agent" });

// Valid context: keys are sorted before signing (consistent hash order).
const sig = await agent.sign({
  actionType: "payment:initiate",
  context: { userId: "u1", currency: "EUR", amount: 49.95 },
  contextSchema: PAYMENT_SCHEMA,
});

// Missing required field: throws ValidationError BEFORE the network call.
try {
  await agent.sign({
    actionType: "payment:initiate",
    context: { amount: 100 },       // userId missing, currency missing
    contextSchema: PAYMENT_SCHEMA,
  });
} catch (err) {
  if (err instanceof ValidationError) {
    console.error(err.message);   // context_schema_error: field 'userId' is required but missing
    console.error(err.docsUrl);   // https://asqav.com/docs/structured-receipts
  }
}
```

Field descriptor keys:
- `type` (required) - one of `"string"`, `"number"`, `"boolean"`, `"object"`, `"array"`.
- `required` (optional, default `false`) - whether the field must be present.

`"number"` rejects `boolean` values; `"array"` rejects plain objects.
No schema means today's exact behaviour (zero behaviour change).

You can also pass a callable for full JSON Schema (no new mandatory
dependencies in the core package):

```typescript
import Ajv from "ajv";
const ajv = new Ajv();
const validate = ajv.compile(MY_FULL_SCHEMA);

await agent.sign({
  actionType: "action",
  context: { ... },
  contextSchema: (ctx) => {
    if (!validate(ctx)) throw new Error(ajv.errorsText(validate.errors));
  },
});
```

See `examples/schema-receipts.ts` for a runnable demo.

## Bind your agent's declared configuration into the receipt

A signed action is more useful when the receipt also proves *which declared
configuration* the agent was running when it acted. You can bind that today
with the sign context, no extra API surface:

1. Hash whatever declares the agent: an agent manifest, an `AGENTS.md`, a
   container image digest, or an SBOM. Any bytes on disk work.
2. Put the digest in the `context` under your own keys (for example
   `config_digest` plus `config_kind`).
3. Optionally pin those keys with a `contextSchema` so a receipt that claims a
   config always carries both fields.

```typescript
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { Agent, init } from "@asqav/sdk";

function configDigest(path: string): string {
  return "sha256:" + createHash("sha256").update(readFileSync(path)).digest("hex");
}

const CONFIG_BINDING_SCHEMA = {
  config_digest: { type: "string", required: true },
  config_kind:   { type: "string", required: true },
} as const;

init({ apiKey: "sk_..." });
const agent = await Agent.create({ name: "my-agent" });

const digest = configDigest("AGENTS.md");

const sig = await agent.sign({
  actionType: "payment:refund",
  context: {
    amount: 49.95,
    currency: "EUR",
    config_digest: digest,        // sha256:<hex> of your declaration
    config_kind: "agents-md",     // name the artifact you hashed
  },
  contextSchema: CONFIG_BINDING_SCHEMA,
});
console.log(sig.verificationUrl); // digest is inside the signed receipt
```

The digest is part of the signed body, so anyone can fetch the receipt at the
public verify endpoint and confirm the action ran under that exact declaration.
The key names and the artifact kind are yours to choose, which keeps the pattern
generic across any declaration format.

For a couple of common cases there are dedicated wire fields instead of context
keys (`config_manifest_digest` for a runtime config snapshot, `sbom_digest` for
an SBOM). The context recipe above is the general form for any other declaration
you want to bind.

See `examples/bind-config-digest.mjs` for a runnable demo.

## Pluggable detectors (bring-your-own DLP/policy)

asqav owns the enforce + signed-proof spine; detection is pluggable. Register a
detector and it runs inside every `agent.sign()`, after the context is normalised
and before the network call. Its verdict is recorded in the signed receipt (under
`_detectors`), so the audit trail proves which detector saw the action and why. A
deny throws `DetectorBlockedError` and no signature is created.

```typescript
import { Agent, DetectorBlockedError, init, registerDetector } from "@asqav/sdk";

const ssnDetector = {
  name: "ssn-regex",
  inspect(_actionType: string, context: Record<string, unknown>) {
    if (/\b\d{3}-\d{2}-\d{4}\b/.test(JSON.stringify(context))) {
      return { allow: false, confidence: 1, labels: ["US_SSN"], detector: "", reason: "SSN" };
    }
    return { allow: true, confidence: 1, labels: [], detector: "", reason: "" };
  },
};

init({ apiKey: "sk_..." });
const agent = await Agent.create({ name: "my-agent" });
registerDetector(ssnDetector);            // failOpen defaults to false

try {
  await agent.sign({ actionType: "email:send", context: { body: "ssn 123-45-6789" } });
} catch (err) {
  if (err instanceof DetectorBlockedError) console.error(err.detectorName, err.result.labels);
}
```

- **Fail-closed by default**: if `inspect()` throws, sign blocks. Opt out per
  detector with `registerDetector(d, { failOpen: true })`.
- **Additive**: with no detector registered, the signed body is byte-identical
  to today (zero behaviour change).
- **Reference detectors** ship in `@asqav/sdk/extras`: `PresidioDetector` (PII,
  via the Presidio REST analyzer) and `OpaDetector` (Open Policy Agent
  policy-as-code). Any classifier or policy engine can feed the signed preflight
  decision this way.

See `examples/detector-plugin.ts` for a runnable demo.

## Requirements

Node 20 or newer. Uses the built-in `fetch`. Zero native dependencies. ML-DSA cryptography runs server-side.

## Standards

Asqav's compliance receipts are profiled in IETF Internet-Draft [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/), an Independent Submission that profiles [`draft-farley-acta-signed-receipts`](https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/) for EU AI Act Articles 12 and 26, and DORA Article 17 bindings. The SDK aligns with NIST FIPS 204 (ML-DSA), JCS canonicalization, and the NSA Cybersecurity Information Sheet U/OO/6030316-26 on MCP server lifecycle telemetry.

## Roadmap

What ships on Asqav today. Each item is available on `main`:

- Hash-only mode for cloud - default for `*.asqav.com`.
- Self-hosted signer with split-trust - compose file in the Asqav backend repo.
- Bring-your-own KMS for AWS KMS or GCP KMS - Enterprise tier.
- Customer-owned storage - self-hosted, with the relay payload allowlist enforced in code.
- SCITT / COSE_Sign1 receipt export - public `GET /api/v1/signatures/{id}/cose` returns `application/cose`.
- Air-gapped / on-prem mode - offline license + zero-egress. See the backend repo `docs/airgapped-mode.md`.

See <https://asqav.com/docs> for the live feature set.

## Documentation

- Repository: <https://github.com/jagmarques/asqav-sdk>
- Full docs: <https://asqav.com/docs>
- SDK guide: <https://asqav.com/docs/sdk>
- IETF profile: <https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/>
- Discovery descriptor: <https://asqav.com/.well-known/governance.json>

## License

Elastic License 2.0. Get an API key at [asqav.com](https://asqav.com).
