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

## Roadmap

What ships on Asqav today. Each item is available on `main`.

### Today

**1. Hash-only mode for cloud**
- WHAT: Hash-only mode on cloud. The SDK hashes `{action_type, context}` locally with RFC 8785 + SHA-256 and sends only the hash plus a small whitelisted metadata bag.
- WHY NOW: Default for any client pointed at api.asqav.com. Raw prompts and tool arguments never leave your process. Spec is open in `docs/fingerprint-spec.md` and cross-language vectors ship in `conformance/vectors.json`.

**2. Self-hosted signer (split-trust)**
- WHAT: Self-hosted signer container. Run the Asqav signing path inside your VPC; ML-DSA-65 private keys, raw prompts, and reasoning traces never cross the container boundary.
- WHY NOW: Compose file and env contract documented in the Asqav backend repo at `docker-compose.signer.yml` and `docs/self-hosted-signer.md`. Optional digest-only relay back to `api.asqav.com` is gated by a fixed allowlist enforced in code.

**3. Bring-your-own KMS (AWS KMS / GCP KMS)** - Enterprise tier
- WHAT: Bring-your-own KMS. Provision your agents' ML-DSA-65 keys in your own AWS KMS or GCP KMS so signing material lives in your HSM, not ours.
- WHY NOW: AWS KMS uses the `ML_DSA_65` key spec on FIPS 140-3 Level 3 HSMs; GCP KMS uses `PQ_SIGN_ML_DSA_65` (software preview today, HSM coming). Toggle with `KMS_PROVIDER=aws|gcp`.

**4. Customer-owned storage**
- WHAT: Customer-owned storage on self-hosted. Postgres, Redis, raw payloads, and ML-DSA private keys all sit in your container. Optional upstream relay only ever sees `{hash, signature, timestamp, algorithm, agent_id, signature_id}`.
- WHY NOW: The allowlist is enforced in code at `src/asqav_cloud/core/signer_relay.py` (Asqav backend repo) and asserted by `tests/test_self_hosted_signer.py`. Auditors can read the forbidden-keys frozenset and confirm what cannot leak.

**5. SCITT / COSE_Sign1 receipt export**
- WHAT: SCITT-compatible receipt export. Public `GET /api/v1/signatures/{id}/cose` returns `application/cose` (CBOR tag 18). The same ML-DSA-65 key signs both the JCS form and the COSE_Sign1 form, so SCITT transparency services can ingest receipts directly.
- WHY NOW: Receipt format and canonical record are deterministic. The CBOR encoder and registration policy are live; both forms share the same key material so existing JCS verifiers keep working unchanged.

**6. Air-gapped / on-prem mode**
- WHAT: Self-hosted signer with offline license validation and zero outbound HTTP.
- WHY NOW: Egress gate, license issuer, and operator guide live in the Asqav backend repo (`src/asqav_cloud/core/airgap.py`, `tools/issue_license.py`, `docs/airgapped-mode.md`). Targeted at regulated EU institutions with no-egress requirements.

See the docs at <https://asqav.com/docs> for the current feature set.

## Standards

Asqav's compliance receipts are profiled in IETF Internet-Draft [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/), an Independent Submission that profiles [`draft-farley-acta-signed-receipts`](https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/) for EU AI Act Articles 12 and 26, and DORA Article 17 bindings.

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

Other framework integrations are tracked on the roadmap.

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

CI runs on every push to `main` and every pull request. It uses `dorny/paths-filter@v3` to only execute the matrix jobs for the language whose source actually changed:

- Changes under `python/` run pytest on Python 3.10, 3.11, and 3.12.
- Changes under `typescript/` run npm test on Node 20 and 22.
- Changes under `conformance/` or to the workflow itself run both.

A final aggregator job named `ci-ok` is the single required status check for branch protection. It passes when each matrix job either succeeded or was skipped (no relevant changes), which avoids the GitHub gotcha where a skipped required check stays pending forever.

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

### Releases - dual-version, prefixed tags

Because both SDKs live in one repo but ship to different registries on independent cadences, releases are driven by **prefixed git tags**:

| Language | Tag pattern | Registry |
|---|---|---|
| Python | `py-vMAJOR.MINOR.PATCH` (e.g. `py-v1.2.3`) | PyPI |
| TypeScript | `ts-vMAJOR.MINOR.PATCH` (e.g. `ts-v1.2.3`) | npm |

To cut a release:

1. Bump the version in the relevant package manifest (`python/pyproject.toml` or `typescript/package.json`).
2. Update [`CHANGELOG.md`](CHANGELOG.md) with the new entry.
3. Tag and push:

   ```bash
   # Python release
   git tag py-v0.2.22
   git push origin py-v0.2.22

   # TypeScript release
   git tag ts-v0.1.1
   git push origin ts-v0.1.1
   ```

The `publish.yml` workflow inspects the tag prefix and only runs the matching publish job. Python uses PyPI's OIDC trusted publisher (no API token needed). TypeScript publishes via the `NPM_TOKEN` repo secret.

Tags must match `py-v*.*.*` or `ts-v*.*.*` exactly - GitHub Actions tag filters are globs, not regex, so the three dot-segments are required to gate the publish.

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
