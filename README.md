<p align="center">
  <a href="https://asqav.com">
    <img src="https://asqav.com/logo-text-white.png" alt="Asqav" width="200">
  </a>
</p>
<p align="center">
  The evidence layer for AI agents. Audit trails, approvals, policy gates, and compliance reports for every action.
</p>
<p align="center">
  <a href="https://pypi.org/project/asqav/"><img src="https://img.shields.io/pypi/v/asqav?style=flat-square&logo=pypi&logoColor=white&label=pypi" alt="PyPI version"></a>
  <a href="https://www.npmjs.com/package/@asqav/sdk"><img src="https://img.shields.io/npm/v/%40asqav%2Fsdk?style=flat-square&logo=npm&logoColor=white&label=npm" alt="npm version"></a>
  <a href="https://github.com/jagmarques/asqav-sdk/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/jagmarques/asqav-sdk/ci.yml?style=flat-square&logo=github&logoColor=white" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square&logo=opensourceinitiative&logoColor=white" alt="License: MIT"></a>
  <a href="https://github.com/jagmarques/asqav-sdk"><img src="https://img.shields.io/github/stars/jagmarques/asqav-sdk?style=social" alt="GitHub stars"></a>
</p>
<p align="center">
  <a href="https://asqav.com">Website</a> |
  <a href="https://asqav.com/docs">Docs</a> |
  <a href="https://asqav.com/docs/sdk">SDK Guide</a> |
  <a href="https://www.asqav.com/docs">Compliance</a>
</p>

# Asqav SDK - Python + TypeScript

The official client SDKs for [Asqav](https://asqav.com), the evidence layer for AI agents. Sign every action with ML-DSA-65 (FIPS 204), enforce policies before execution, and produce regulator-ready audit trails.

This repository ships two SDKs from a single monorepo:

- `python/` - the original Python SDK, published to PyPI as [`asqav`](https://pypi.org/project/asqav/)
- `typescript/` - the TypeScript SDK, published to npm as [`@asqav/sdk`](https://www.npmjs.com/package/@asqav/sdk)

Both wrap the same Asqav API. Pick the one that matches your stack.

## Install

Python:

```bash
pip install asqav
```

TypeScript / Node.js:

```bash
npm install @asqav/sdk
```

Both packages have zero native dependencies. All ML-DSA cryptography runs server-side.

## 30-second example

Python:

```python
import asqav

asqav.init(api_key="sk_...")

agent = asqav.Agent.create("my-agent")
sig = agent.sign("api:call", {"model": "gpt-4"})

print(sig.signature_id, sig.verify_url)
```

TypeScript:

```ts
import { init, Agent } from "@asqav/sdk";

init({ apiKey: "sk_..." });

const agent = await Agent.create({ name: "my-agent" });
const sig = await agent.sign({
  actionType: "api:call",
  context: { model: "gpt-4" },
});

console.log(sig.signatureId, sig.verificationUrl);
```

Each signed action returns a receipt like:

```json
{
  "signature_id": "sig_a1b2c3",
  "agent_id": "agt_x7y8z9",
  "action": "api:call",
  "algorithm": "ML-DSA-65",
  "timestamp": "2026-04-27T18:30:00Z",
  "chain_hash": "b94d27b9934d3e08...",
  "verify_url": "https://asqav.com/verify/sig_a1b2c3"
}
```

The agent now has a cryptographic identity, a signed audit trail, and a verifiable action record.

## Data handling modes

The SDK auto-detects whether you're pointing at the Asqav cloud or a self-hosted deployment, and selects the safer default for each:

- **Cloud (`*.asqav.com`)**: hash-only by default. The SDK builds a fingerprint of your action context, computes a SHA-256 hash locally, and sends only the hash plus a small metadata bag (action_type, agent_id, session_id, model_name, tool_name). Raw prompts and tool arguments stay on your side.
- **Self-hosted**: full-payload by default. The server can run policy checks, PII redaction, and richer audit. Recommended when you control the deployment.

Override anytime:

```python
# Python
asqav.init(api_key="...", base_url="https://api.asqav.com", mode="hash-only")
```

```ts
// TypeScript
await init({ apiKey: "...", baseUrl: "https://api.asqav.com", mode: "hash-only" });
```

The fingerprint format is RFC 8785-style sorted JSON with no whitespace, hashed with SHA-256. See `docs/fingerprint-spec.md` and `conformance/vectors.json` for the spec and cross-language test vectors.

For self-hosted and split-trust deployments, bring-your-own KMS, SCITT and COSE receipt export, and air-gapped mode, see [asqav.com/docs](https://asqav.com/docs).

## Standards

Asqav's compliance receipts are profiled in IETF Internet-Draft [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/), an Independent Submission that profiles [`draft-farley-acta-signed-receipts`](https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/) for EU AI Act Articles 12 and 26, and DORA Article 17 bindings.

## Coding agents

Autonomous coding agents open and merge their own pull requests, and a git author string is mutable. The `protectmcp:lifecycle:code_authorship` receipt binds a change to the agent that authored it: the repository, the commit and base SHA, a digest of the diff, the change class, and a producer-asserted `authored_by` block, signed with ML-DSA-65 and chained per agent. Asqav records that the change existed, was key-authored, and was chained at a point in time. It does not clone the repository, re-run the diff, or judge whether the code is correct. A ready-to-use GitHub Action lives in [`github-action/`](github-action/) and can also emit the receipt as an in-toto attestation. See the [code-authorship receipts docs](https://asqav.com/docs/code-authorship-receipts).

## Why governance

| Without governance | With Asqav |
|---|---|
| No record of what agents did | Every action signed with ML-DSA (FIPS 204) |
| Any agent can do anything | Policies block dangerous actions in real-time |
| One person approves everything | Multi-party authorization for critical actions |
| Manual compliance reports | Automated EU AI Act and DORA reports |
| Reasoning lost after the run | Prompt, trace, and output signed and replayable |

## Per-language docs

The root README only covers the cross-language story. For language-specific guides - decorators, async clients, framework integrations, CLI commands, local mode, batched signing, three-phase signing, scope tokens, replay, attestations, and more - see:

- Python: [`python/README.md`](python/README.md)
- TypeScript: [`typescript/README.md`](typescript/README.md)

The full reference lives at [asqav.com/docs](https://asqav.com/docs).

### What's in each SDK

Both SDKs cover the same core surface: agent identity, signed actions, batched signing, policy enforcement, scope tokens, replay, attestations, three-phase signing, and the `user_intent` envelope on `agent.sign(...)` (Ed25519 / ECDSA P-256 / WebAuthn).

The Python SDK additionally ships:

- The `asqav` CLI: `quickstart`, `demo`, `verify`, `doctor`, `agents`, `sync`, plus `replay`, `preflight`, `approve`, `budget` (`check`/`record`), and `compliance` (`frameworks`/`export`). See [asqav.com/docs/cli](https://asqav.com/docs/cli).
- A pytest plugin (`pytest --asqav`) that signs every test result and emits a Merkle-rooted compliance bundle on session finish. See [asqav.com/docs/pytest-plugin](https://asqav.com/docs/pytest-plugin).
- Native callbacks for LangChain, CrewAI, LiteLLM, Haystack, OpenAI Agents SDK, LlamaIndex, smolagents, DSPy, PydanticAI, Letta, Strands, and Instructor (via the Hooks API).
- Cookbooks for Streamlit dashboards and Dify workflows under `python/examples/`.
- Local-mode offline signing with deferred sync.

The TypeScript SDK focuses on the core API and `Agent` surface for Node 20+ runtimes. It additionally ships:

- The `user_intent` envelope on `agent.sign(...)`.
- A Vercel AI SDK adapter at `@asqav/sdk/extras/vercel-ai` that plugs into `experimental_telemetry: { tracer }` and signs every span the AI SDK opens.
- A LangChain.js callback handler `AsqavCallbackHandler` at `@asqav/sdk/extras/langchain` that signs each chain step.
- A Mastra hook `AsqavMastraHook` at `@asqav/sdk/extras/mastra` that signs each step the Mastra agent emits.
- An OpenAI Agents JS adapter `AsqavOpenAIAgentsAdapter` at `@asqav/sdk/extras/openai-agents` that signs tool calls on the agent runner.

If a feature exists in one SDK but not the other, the language README will mark it explicitly.

## Repository layout

```
asqav-sdk/
  python/           Python SDK (pip install asqav)
    src/
    tests/
    pyproject.toml
    README.md
  typescript/       TypeScript SDK (npm install @asqav/sdk)
    src/
    tests/
    package.json
    README.md
  conformance/      Cross-language conformance fixtures both SDKs run against
  .github/workflows/
    ci.yml          Path-filtered CI for both languages
    publish.yml     Tag-based publish (py-v* to PyPI, ts-v* to npm)
  CHANGELOG.md
  README.md         (this file)
```

The two SDKs are versioned and released independently.

## Verifying a receipt

Receipts are portable. Anyone with a `signature_id` can verify it without an API key:

Python:

```python
import asqav

result = asqav.verify("sig_a1b2c3")
assert result["verified"]
print(result["agent_id"], result["chain_hash"])
```

TypeScript:

```ts
import { verify } from "@asqav/sdk";

const result = await verify("sig_a1b2c3");
console.assert(result.verified);
console.log(result.agentId, result.chainHash);
```

Or open the receipt's `verify_url` in a browser. Hashes are reproducible offline from the RFC 8785 (the JSON format spec) payload, so auditors do not need to trust Asqav's servers - the signature speaks for itself.

## Counterparty acknowledgment

When two agents hand off a workflow (A signs an action, B acts on it), the SDK lets B emit a `protectmcp:acknowledgment` receipt that cryptographically binds B's bytes to A's. The bundle carries A's full envelope, B's acknowledgment, and a SHA-256 binding the verifier recomputes offline. A tampered intermediary cannot mutate the bytes without breaking the equality, so a regulator can verify the handoff with one digest comparison.

Python:

```python
from asqav import compute_counterparty_binding, verify_counterparty_binding

binding = compute_counterparty_binding(originating_envelope)
b_payload["counterparty_binding"] = binding.to_wire()
# ... B signs b_payload normally ...

outcome = verify_counterparty_binding(b_envelope, originating_envelope)
assert outcome.valid, outcome.label
```

TypeScript:

```ts
import { computeCounterpartyBinding, verifyCounterpartyBinding } from "@asqav/sdk";

const binding = computeCounterpartyBinding(originatingEnvelope);
bPayload.counterparty_binding = binding;
// ... B signs bPayload normally ...

const outcome = verifyCounterpartyBinding(bEnvelope, originatingEnvelope);
if (!outcome.valid) throw new Error(outcome.label);
```

The cloud `/verify` endpoint reports the same outcome on the verification response under `counterparty_binding_verified`, and `/audit-pack/export` pulls the originating receipt into the bundle automatically when an acknowledgment in the export window references one outside it.

## Conformance

The `conformance/` directory contains shared test fixtures both SDKs run against. Adding a feature means adding a fixture there first, then making both SDKs pass it. CI reruns both matrices whenever `conformance/` changes, even if neither `python/` nor `typescript/` was touched, so cross-language drift is caught at PR time.

## CI

CI runs on every push to `main` and every pull request. It tests only the language whose source changed: pytest on Python 3.10, 3.11, and 3.12, and npm test on Node 20 and 22. A change to `conformance/` runs both. The aggregator job `ci-ok` is the single required status check for branch protection.

### Run governance checks in your own CI

`asqav doctor` validates configuration and connectivity in any environment with `ASQAV_API_KEY` set, returning non-zero on failure so it gates PRs cleanly. Pair it with `asqav.compliance.export_bundle` to package signed receipts from your test runs into a self-contained, Merkle-rooted artifact for auditors. See [`docs/github-actions.md`](docs/github-actions.md) for a copy-paste workflow.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide. The short version:

1. Fork the repo and create a feature branch.
2. Run the relevant test suite locally.
   - Python: `cd python && pip install -e ".[all]" pytest && pytest tests -v`
   - TypeScript: `cd typescript && npm ci && npm test`
3. Open a pull request against `main`. CI must go green for the PR to land.

Direct pushes to `main` are blocked. Every change lands through a PR.

### Releases

Both SDKs ship to different registries on independent cadences, driven by prefixed git tags: `py-v*` publishes the Python half to PyPI, `ts-v*` publishes the TypeScript half to npm. Bump the version in the relevant manifest, update [`CHANGELOG.md`](CHANGELOG.md), then tag and push. The `publish.yml` workflow inspects the tag prefix and runs only the matching publish job. Python uses PyPI's OIDC trusted publisher; TypeScript publishes via the `NPM_TOKEN` repo secret.

## Ecosystem

| Package | What it does |
|---|---|
| [`asqav` (PyPI)](https://pypi.org/project/asqav/) | Python SDK |
| [`@asqav/sdk` (npm)](https://www.npmjs.com/package/@asqav/sdk) | TypeScript SDK |
| [asqav-mcp](https://github.com/jagmarques/asqav-mcp) | MCP server for Claude Desktop, Claude Code, Cursor |
| [asqav-compliance](https://github.com/jagmarques/asqav-compliance) | CI/CD compliance scanner |

## Free tier

Get started at no cost. Free includes 50K signatures/month, 20 agents, 10 policies, 3 team members, OpenTimestamps, MCP server, audit export, and framework integrations. Content scanning, OpenTelemetry, and Replay API on Pro. Quarantine, incident management, multi-party signing, compliance reports, and RFC 3161 timestamps on Business. Managed KMS, SSO, IP allowlist, and bring-your-own KMS on Enterprise. See [asqav.com/pricing](https://asqav.com/pricing.html) for the full breakdown.

## Discovery

Machine-readable service descriptor at [`https://asqav.com/.well-known/governance.json`](https://asqav.com/.well-known/governance.json) for external tools that want to auto-discover Asqav's endpoints, algorithms, capabilities, and integrations.

## Links

- Docs: <https://asqav.com/docs>
- Repo: <https://github.com/jagmarques/asqav-sdk>
- Blog: <https://dev.to/jagmarques>
- Dashboard: <https://asqav.com/dashboard>

## License

MIT - see [LICENSE](LICENSE) for details.

---

If Asqav helps you, consider giving the repo a star. It helps others find the project.
