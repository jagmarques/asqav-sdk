# asqav

TypeScript SDK for [asqav.com](https://asqav.com), the evidence layer for AI agents. Sign every agent action with ML-DSA-65 (FIPS 204), enforce policies before execution, and produce regulator-ready audit trails. All cryptography runs server-side; the package has zero native dependencies.

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
  actionType: "payment.wire_transfer",
  context: { amountEur: 850000, beneficiaryIban: "DE89370400440532013000" },
  receiptType: "protectmcp:decision",
  riskClass: "high",
  issuerId: "legal:Acme GmbH",
  iterationId: "task-2026-Q2-4821",
});

console.log(sig.complianceMode);        // true (default; pass complianceMode: false to opt out)
console.log(sig.actionRef);             // "sha256:..." over the JCS-canonical action
console.log(sig.previousReceiptHash);   // 64 hex; "0".repeat(64) on the first record per agent
console.log(sig.verificationUrl);
```

Each signed action lands on a Compliance Receipt under IETF Internet-Draft [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/) by default: ML-DSA-65 (FIPS 204) signature, chain hash, retained `policy_digest`, fail-closed anchoring, and a public verification URL. Pass `complianceMode: false` if you want a non-Compliance receipt.

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
asqav replay <agent_id> <session_id> [--json]              # Pro
asqav replay-verify <agent_id> <session_id> [--strict]     # Pro: IETF chain
asqav preflight <agent_id> <action_type> [--json]          # Pro
asqav budget check / record                                # Pro
asqav approve <session_id> <entity_id>                     # Pro
asqav compliance frameworks / export                       # Business
asqav audit-pack export --start ISO --end ISO --output-file bundle.json
asqav audit-pack policy <sha256:hex>
asqav payloads erase <signature_id> --yes                  # P4 right-to-erasure
asqav org set-compliance-strict <org_id> --enable|--disable
asqav keys generate --algorithm ed25519|es256 [--out priv.bin]
asqav migrate run v3-20|v3-21|v3-22                        # X-Maintenance-Key required
```

Pro and Business commands are gated client-side via `GET /account` so a free-tier key gets a clean upgrade message instead of a mid-pipeline 402. The server is the source of truth; self-hosted deployments without `/account` skip the gate.

## Data handling modes

The SDK auto-detects whether you're pointing at the Asqav cloud or a self-hosted deployment and selects the safer default for each:

- **Cloud (`*.asqav.com`)**: hash-only by default. The SDK builds a fingerprint of your action context, computes a SHA-256 hash locally, and sends only the hash plus a small metadata bag (`action_type`, `agent_id`, `session_id`, `model_name`, `tool_name`). Raw prompts and tool arguments stay on your side.
- **Self-hosted**: full-payload by default. The server can run policy checks, PII redaction, and richer audit. Recommended when you control the deployment.

Override anytime:

```ts
await init({ apiKey: "...", baseUrl: "https://api.asqav.com", mode: "hash-only" });
```

The fingerprint format is sorted JSON with no whitespace per JCS, hashed with SHA-256. See `docs/fingerprint-spec.md` and `conformance/vectors.json` for the spec and cross-language test vectors.

## Compliance receipts (IETF profile)

Compliance Receipts are the SDK default. Each `agent.sign(...)` call produces a receipt that conforms to [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/): ML-DSA-65 signature, JCS canonicalization, retained `policy_digest`, hash-chained `previous_receipt_hash`, OpenTimestamps anchoring. Opt out with `complianceMode: false` if you want the older shape.

The envelope extensions most callers reach for (camelCase on the SDK, snake_case on the JSON wire):

- `receiptType` -> `receipt_type` - `protectmcp:decision`, `protectmcp:restraint`, `protectmcp:lifecycle`, `protectmcp:lifecycle:configuration_change`, `protectmcp:acknowledgment`, `protectmcp:observation`, or `protectmcp:observation:result_bound` (observation receipts that bind tool output via `resultDigest`).
- `riskClass` -> `risk_class` - controlled vocabulary: `unacceptable | high | limited | minimal | gpai | low | medium | unknown`.
- `iterationId` -> `iteration_id` - logical task id, distinct from session.
- `sandboxState` -> `sandbox_state` - `enabled | disabled | unavailable` for high-risk gating.
- `incidentClass` -> `incident_class` - DORA / NYDFS / CIRCIA token (or array of tokens).
- `issuerId` -> `issuer_id` - LEI (ISO 17442), EIN, CIK, or a W3C DID for non-LEI deployers.

### Shadow AI capture (passive_telemetry)

Two `receiptType` values cover the gating axis: `protectmcp:decision` records that a policy ran and gated the action; `protectmcp:observation` records that a passive monitor saw the event without gating it. Pick `observation` when the producer never had the option to block (SIEM forwarder, browser extension in observe-only mode, NetFlow-style proxy with no enforcement hook).

Set `captureTopology: "passive_telemetry"` to declare the producer is observing after the fact. The SDK client-side check pre-flights the Asqav cloud's full rule 8 gate: a `captureTopology: "passive_telemetry"` receipt MUST use `receiptType: "protectmcp:observation"`. Any other receiptType paired with `passive_telemetry` (`:decision`, `:restraint`, `:lifecycle`, `:lifecycle:configuration_change`, `:acknowledgment`) throws `AsqavError` with the verbatim `false_attestation_guard: capture_topology=passive_telemetry receipts must use receipt_type=protectmcp:observation, not :<offending> (rule 8)` message before the HTTP roundtrip (see `false_attestation_guard` in `typescript/src/index.ts`).

```ts
const sig = await agent.sign({
  actionType: "mcp:tool_call",
  context: { server: "filesystem", tool: "read" },
  receiptType: "protectmcp:observation",
  captureTopology: "passive_telemetry",
  issuerId: "legal:Acme GmbH",
});
```

`captureTopology` is stamped on the audit-pack manifest entry but never on the signed payload. The other accepted topologies are `in_process_sdk`, `network_proxy`, `browser_extension`, `ebpf_observer`, and `mcp_proxy`; only `passive_telemetry` triggers the false-attestation guard. The full topology semantics live in the cloud's `docs/capture-topology.md`, and the wire vocabulary is published live at `https://api.asqav.com/.well-known/governance.json` for discovery.

### Configuration change receipts (rule 9)

A `receiptType: "protectmcp:lifecycle:configuration_change"` receipt declares the agent's runtime configuration was mutated. The SDK pre-flights the Asqav cloud's rule 9 cross-field gate (NSA CSI U/OO/6030316-26 alignment): the receipt MUST carry `configManifestDigest`. Omitting it throws `AsqavError` with the verbatim `false_attestation_guard: receipt_type=protectmcp:lifecycle:configuration_change requires config_manifest_digest (rule 9)` message before the HTTP roundtrip.

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

### Expiry precedence (rule 10)

`validSeconds` (legacy, server computes `valid_until = signed_at + valid_seconds`) and `expiresAt` (caller-supplied horizon) are mutually exclusive. Pass exactly one of the two: `validSeconds: 3600` for "expire one hour after signing", or `expiresAt: "2026-06-01T00:00:00Z"` for an explicit horizon. Passing both throws `AsqavError` with the verbatim `expiry_collision_guard: pass either valid_seconds or expires_at, not both (rule 10)` message before the HTTP roundtrip. Passing neither falls back to the server-side default (`valid_seconds=86400`).

### Digest format (rule 11)

Every caller-supplied digest field (`toolFingerprint`, `configManifestDigest`, `cveInventoryDigest`) MUST match the regex `^sha256:[a-f0-9]{64}$`. Anything else throws `AsqavError` with the verbatim `digest_format_guard: <field> must match sha256:<64-hex> (rule 11)` message before the HTTP roundtrip. To avoid wire drift, use the SDK's deterministic helpers (each is byte-deterministic under JCS):

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
- `expiresAt` - explicit ISO-8601 validity horizon. Mutually exclusive with `validSeconds`. See <https://www.asqav.com/docs/expires-at>.
- `nonce` - 12 random bytes auto-generated when omitted; cloud rejects duplicates inside the validity window. See <https://www.asqav.com/docs/nonce>.
- `toolFingerprint` - `sha256:<hex>` over `{tool_name, schema}`, auto-derived when `toolName` + `toolSchema` are present. See <https://www.asqav.com/docs/tool-fingerprint>.
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
  action: "tool.invoke",
  toolName: "exec_query",
  executableHash: "sha256:" + "a".repeat(64),
  sbomDigest: "sha256:" + "b".repeat(64),
  slsaProvenancePointer: "https://example.com/slsa/build-123.json",
  supplyChainPointer: "https://rekor.sigstore.dev/api/v1/log/entries/abc",
});
```

See <https://www.asqav.com/docs/executable-hash-and-sbom-provenance> for the field semantics and a worked policy example.

## Threat-framework mappings (optional)

Seven optional wire fields let a caller pin the receipt to industry threat-and-control taxonomies. Each list is caller-supplied and Asqav-preserved verbatim; the cloud sets `framework_mappings_self_declared=true` on the receipt whenever any of the six list fields is populated, so verifiers can tell self-declared classifications apart from cloud-verified ones.

- `mitreTechniques` - list of MITRE ATT&CK technique ids (e.g. `["T1059", "T1078"]`).
- `mitreAtlas` - list of MITRE ATLAS ids for AI-system threats (e.g. `["AML.T0051"]`).
- `owaspLlmTop10` - list of OWASP Top 10 for LLM ids (e.g. `["LLM01", "LLM02"]`).
- `nistAiRmf` - list of NIST AI RMF function ids (e.g. `["GOVERN-1.1", "MEASURE-2.7"]`).
- `iso42001` - list of ISO/IEC 42001 control ids (e.g. `["A.6.2.6"]`).
- `euAiActArticles` - list of EU AI Act article ids (e.g. `["Article-12", "Article-15"]`).
- `rfc3161Timestamp` - caller-supplied base64-encoded RFC 3161 TimeStampResp (DER).

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

Each list must be a non-empty array of strings, each entry up to 128 characters; empty arrays, non-string entries, or oversize entries throw `AsqavError` with verbatim guard messages (`<field>_must_be_non_empty_list`, `<field>_entry_invalid`) and skip the HTTP roundtrip. The `rfc3161Timestamp` must be valid base64 or throws `rfc3161_timestamp_not_base64`. Camel-case props project onto snake_case wire keys via the SDK's `IETF_OPTIONAL_FIELD_MAP`.

See <https://www.asqav.com/docs/threat-framework-mapping>.

### Audit Pack export

The SDK exposes the public audit-trail endpoint as `exportAuditJson` for filtered windows of receipts (Pro tier and above):

```ts
import { exportAuditJson } from "@asqav/sdk";

const bundle = await exportAuditJson({
  startDate: "2026-05-01T00:00Z",
  endDate: "2026-06-01T00:00Z",
  agentId: "agt_abc",
});
console.log(bundle.records.length, bundle.bundleDigest);
```

For offline chain verification, `verifyChain(records)` walks an ordered list of signed envelopes for one agent and re-derives each `previous_receipt_hash`. The first record's seed is `"0".repeat(64)`. The cloud is the authoritative verifier; this helper is a convenience.

Algorithm agility is exposed via `SUPPORTED_ALGORITHMS`. Pass `algorithm: "ed25519"` or `"es256"` to `Agent.create(...)` for non-post-quantum identities, or `generateKeypair("ed25519")` for offline scenarios. ES256 signatures emit in IEEE-P1363 (raw r||s) form so they match the cloud verifier byte-for-byte.

## Integrations

The SDK ships native callbacks and adapters for common agent frameworks:

- **Vercel AI SDK** - import `createAsqavExporter` from `@asqav/sdk/extras/vercel-ai`, pass to `experimental_telemetry: { tracer }`. Every span (text generation, tool calls, embeddings) becomes a signed Asqav action.
- **LangChain.js** - import `AsqavCallbackHandler` from `@asqav/sdk/extras/langchain`, construct via `await AsqavCallbackHandler.create({ agent })` then pass to `callbacks: [handler]`. Requires `@langchain/core` as a peer dep.
- **Mastra** - import `AsqavMastraHook` from `@asqav/sdk/extras/mastra`, attach via `await new AsqavMastraHook({ agent }).attach(mastraAgent)`. Requires `@mastra/core` as a peer dep.
- **OpenAI Agents JS** - import `AsqavOpenAIAgentsAdapter` from `@asqav/sdk/extras/openai-agents`, wrap individual tools via `new AsqavOpenAIAgentsAdapter({ agent }).wrapTool(tool)`. Each invocation signs `tool:start` / `tool:end` / `tool:error`. Requires `@openai/agents` as a peer dep.
- **CrewAI**, **LiteLLM**, **LlamaIndex**, **Haystack** - same pattern via the Hooks API (Python SDK has the richer ecosystem; see <https://asqav.com/docs/integrations> for parity status).

Span names are mapped to Asqav action types: `ai.generateText` -> `ai:completion`, `ai.streamText` -> `ai:completion_stream`, `ai.toolCall` -> `ai:tool_call`, errors -> `ai:error`. Signing is fire-and-forget and never blocks generation; failures log a warning instead of throwing.

## Errors

All thrown errors extend `AsqavError`. `AuthenticationError` (401), `RateLimitError` (429), and `APIError` (with `statusCode`) are exported for fine-grained handling:

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

## Requirements

Node 20 or newer. Uses the built-in `fetch`. Zero native dependencies; ML-DSA cryptography runs server-side.

## Standards

Asqav's compliance receipts are profiled in IETF Internet-Draft [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/), an Independent Submission that profiles [`draft-farley-acta-signed-receipts`](https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/) for EU AI Act Articles 12 and 26, and DORA Article 17 bindings. The SDK aligns with NIST FIPS 204 (ML-DSA), JCS canonicalization, and the NSA Cybersecurity Information Sheet U/OO/6030316-26 on MCP server lifecycle telemetry.

## Roadmap

What ships on Asqav today. Each item is available on `main`:

- Hash-only mode for cloud - default for `*.asqav.com`.
- Self-hosted signer (split-trust) - compose file in the Asqav backend repo.
- Bring-your-own KMS (AWS KMS / GCP KMS) - Enterprise tier.
- Customer-owned storage - self-hosted; relay payload allowlist enforced in code.
- SCITT / COSE_Sign1 receipt export - public `GET /api/v1/signatures/{id}/cose` returns `application/cose`.
- Air-gapped / on-prem mode - offline license + zero-egress; see the backend repo `docs/airgapped-mode.md`.

See <https://asqav.com/docs> for the live feature set.

## Documentation

- Repository: <https://github.com/jagmarques/asqav-sdk>
- Full docs: <https://asqav.com/docs>
- SDK guide: <https://asqav.com/docs/sdk>
- IETF profile: <https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/>
- Discovery descriptor: <https://asqav.com/.well-known/governance.json>

## License

Apache-2.0. Get an API key at [asqav.com](https://asqav.com).
