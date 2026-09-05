# asqav

[![GitHub stars](https://img.shields.io/github/stars/jagmarques/asqav-sdk?style=social)](https://github.com/jagmarques/asqav-sdk)

TypeScript SDK for [asqav.com](https://asqav.com), the evidence layer for AI agents.

Every agent action gets a signed, hash-chained **compliance receipt**: ML-DSA-65 (FIPS 204,
post-quantum), timestamped against independent witnesses, and verifiable by anyone —
auditor, counterparty, regulator — **without an Asqav account and without trusting us**.

Zero native dependencies. Cryptography runs server-side.

## Install

```bash
npm install @asqav/sdk
```

## Quick start

```bash
npm install -g @asqav/sdk
asqav login        # validates your key, saves it to ~/.asqav/credentials
```

```ts
import { govern } from "@asqav/sdk";

// govern() = init() + Agent.create() in one call
const agent = await govern({ apiKey: process.env.ASQAV_API_KEY, agentName: "my-agent" });

const sig = await agent.sign({ actionType: "api:openai:chat", context: { model: "gpt-4o" } });

console.log(sig.actionRef);            // "sha256:..." over the JCS-canonical action
console.log(sig.previousReceiptHash);  // 64 hex; "0".repeat(64) on this agent's first receipt
console.log(sig.verificationUrl);      // anyone can open this
```

One install, one `govern`, one `sign`. `init({ apiKey })` + `Agent.create({ name })` remain
available when you want control over `algorithm`, `capabilities` and other agent options.

## Verify it without an account

This is the point of the whole thing — the receipt stands on its own:

Run this right now, with no key and no signup:

```ts
import { verify } from "@asqav/sdk";

const result = await verify("sig_example_regulator_cold_verify_2026");
console.log(result.verified);    // false -- and that is the point, see below
console.log(result.chainHash);   // recomputed on your machine from the canonical bytes
```

That id is a **shape example**: its signature bytes are placeholders and its `kid` resolves
to no key, so the verifier returns `verified: false` instead of waving it through. Swap in a
`signature_id` of your own for a receipt that passes. A verifier that says no when the
evidence is absent is the only kind worth having.

From the shell (`npm install -g @asqav/sdk` provides the `asqav` command):

```bash
asqav verify <your_signature_id>
```

Offline or air-gapped, snapshot the keys once and verify with no network at all. This
re-derives the signature itself from the RFC 8785 canonical bytes and reports a verdict per
axis, so it never asks Asqav to vouch for anything:

```ts
import { fetchJwks, verifyReceiptOffline } from "@asqav/sdk";
import { readFileSync, writeFileSync } from "node:fs";

const jwks = await fetchJwks();                        // online, once
writeFileSync("jwks.json", JSON.stringify(jwks));

const receipt = JSON.parse(readFileSync("receipt.json", "utf8"));   // offline from here
const result = await verifyReceiptOffline(receipt, JSON.parse(readFileSync("jwks.json", "utf8")));
if (result.verdict === "unverified") throw new Error(JSON.stringify(result.axes));
```

`verdict` is `"verified" | "verified_keyed" | "unverified"`. `verified_keyed` is a pass over a
keyed digest: internally consistent, but not third-party re-derivable. Pass
`{ predecessor: prevReceipt }` to check the hash-chain link too.
ML-DSA-65 verifies in-process via `@noble/post-quantum`; Ed25519 and ES256 via `node:crypto`.
Guide: [offline and air-gapped verification](https://asqav.com/docs/offline-air-gapped-verification).

## Neutral verifier

A standalone verifier for agent receipts **across formats** ships as a subpath export:

```ts
import { verify, ADAPTERS } from "@asqav/sdk/verifier";

const result = verify(receipt, ADAPTERS, keyProvider);
```

It checks the issuer signature over the canonical bytes, the chain link and structural
presence for the asqav-native, AERF, ACTA, agent-receipts and Authproof formats. No account
required. It is the port of the Python `asqav.verifier.oracle`, held to verdict parity by a
shared conformance corpus.

## Data handling modes

The SDK picks the safer default for where you point it:

- **Cloud (`*.asqav.com`)** — hash-only. A SHA-256 fingerprint is computed locally and only
  the hash plus `actionType`, `agentId`, `sessionId`, `modelName`, `toolName` is sent.
  Prompts and tool arguments never leave your process.
- **Self-hosted** — full payload, so the server can run policy checks and richer audit.

```ts
await init({ apiKey: "...", baseUrl: "https://api.asqav.com", mode: "hash-only" });
```

There is no local offline queue in TypeScript; use the Python SDK's `local_sign` for that.

## High-value actions

For regulated or high-risk actions, pass envelope fields that an auditor will look for:

```ts
const sig = await agent.sign({
  actionType: "payment.wire_transfer",
  context: { amountEur: 850000, beneficiaryIban: "DE89370400440532013000" },
  receiptType: "protectmcp:decision",
  riskClass: "high",                 // low | medium | high | unknown
  issuerId: "legal:Acme GmbH",       // LEI, EIN, CIK or W3C DID
  iterationId: "task-2026-Q2-4821",  // logical task, distinct from session
});
```

## CLI

```bash
asqav whoami                       # active key source, validated
asqav init                         # print a ready-to-paste snippet
asqav verify <signature_id>        # verify a receipt (no key needed)
asqav sign --agent-id ID --action-type T --action-json action.json
asqav agents list | create | revoke
asqav replay-verify <agent_id> <session_id> [--strict]
asqav audit-pack export --start ISO --end ISO --output-file bundle.json
asqav payloads erase <signature_id>          # right-to-erasure
```

Full command reference: [asqav.com/docs/cli](https://asqav.com/docs/cli).

## Framework integrations

Native adapters under `@asqav/sdk/extras/*`:

- **Vercel AI SDK** — `createAsqavExporter`, passed to `experimental_telemetry`. Every span
  becomes a signed action.
- **LangChain.js** — `AsqavCallbackHandler`, passed to `callbacks: [handler]`.
- **Mastra** — `AsqavMastraHook`, attached to a Mastra agent.
- **OpenAI Agents JS** — `AsqavOpenAIAgentsAdapter`, wrapping a tool.

Each needs its framework as a peer dependency. CrewAI, LiteLLM, LlamaIndex and Haystack
follow the same Hooks API; the Python SDK has the richer ecosystem. Parity status:
[integrations](https://asqav.com/docs/integrations).

## What a receipt does not prove

Stated plainly, because an auditor will ask:

- **Not that the action ran, or ran once.** A receipt records a decision and the bytes that
  passed through, never the effect. A retry produces a second receipt.
- **Not that the policy was correct.** `policy_digest` proves which policy artefact existed,
  not that it was the right one.
- **Not that the environment was intact.** No field here carries a remote attestation result.
- **Tamper-evident, not tamper-proof.** Modification is detectable, not prevented.

The full list is in the IETF profile under "What a Compliance Receipt Does Not Prove".

## Reference

| Topic | Docs |
|---|---|
| Threat-framework mappings (MITRE, OWASP, NIST AI RMF, ISO 42001, EU AI Act) | [threat-framework-mapping](https://asqav.com/docs/threat-framework-mapping) |
| NSA CSI U/OO/6030316-26 receipt fields | [nsa-mcp-csi-alignment](https://asqav.com/docs/nsa-mcp-csi-alignment) |
| Build provenance (`executable_hash`, `sbom_digest`, SLSA) | [executable-hash-and-sbom-provenance](https://asqav.com/docs/executable-hash-and-sbom-provenance) |
| Witness policy and multi-witness anchoring | [multi-witness-anchoring](https://asqav.com/docs/multi-witness-anchoring) |
| Binding tool output (`result_digest`) | [result-digest](https://asqav.com/docs/result-digest) |
| Configuration-change receipts | [configuration-change-receipts](https://asqav.com/docs/configuration-change-receipts) |
| Code-authorship receipts | [code-authorship-receipts](https://asqav.com/docs/code-authorship-receipts) |
| Structured receipts (optional schema) | [structured-receipts](https://asqav.com/docs/structured-receipts) |
| Bring-your-own DLP / policy detectors | [scanning](https://asqav.com/docs/scanning) |
| Audit Pack export | [compliance](https://asqav.com/docs/compliance) |
| Independent verification protocol | [independent-verification](https://asqav.com/docs/independent-verification) |

## Requirements

Node 20+. Uses the built-in `fetch`. Zero native dependencies.

## Standards

Profiled in the IETF Internet-Draft
[`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/),
an Independent Submission profiling
[`draft-farley-acta-signed-receipts`](https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/).
Aligns with NIST FIPS 204 (ML-DSA), RFC 8785 (JCS) and NSA CSI U/OO/6030316-26.

## Links

[Docs](https://asqav.com/docs) · [SDK guide](https://asqav.com/docs/sdk) ·
[Repository](https://github.com/jagmarques/asqav-sdk) ·
[Discovery descriptor](https://asqav.com/.well-known/governance.json)

## License

Elastic License 2.0. Get an API key at [asqav.com](https://asqav.com).
