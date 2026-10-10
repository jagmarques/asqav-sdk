# Asqav SDK — Python + TypeScript

[![GitHub stars](https://img.shields.io/github/stars/jagmarques/asqav-sdk?style=social)](https://github.com/jagmarques/asqav-sdk)
[![PyPI](https://img.shields.io/pypi/v/asqav)](https://pypi.org/project/asqav/)
[![npm](https://img.shields.io/npm/v/@asqav/sdk)](https://www.npmjs.com/package/@asqav/sdk)

Client SDKs for [asqav.com](https://asqav.com), the evidence layer for AI agents.

Every agent action gets a signed, hash-chained **compliance receipt**: ML-DSA-65 (FIPS 204,
post-quantum), timestamped against independent witnesses, and verifiable by anyone —
auditor, counterparty, regulator — **without an Asqav account and without trusting us**.

Source-available client SDK under Elastic License 2.0; open verifier and conformance vectors under Apache-2.0; commercial signing service.

Zero native dependencies in either SDK. Cryptography runs server-side.

## Install

```bash
pip install asqav            # Python 3.10+
npm install @asqav/sdk       # Node 20.19 through 20.x, or 22.12 and later
```

## Sign an action

```python
import asqav

agent = asqav.govern(api_key="sk_...", agent_name="my-agent")
sig = agent.sign("api:openai:chat", {"model": "gpt-4o"})
print(sig.verification_url)
```

```ts
import { govern } from "@asqav/sdk";

const agent = await govern({ apiKey: process.env.ASQAV_API_KEY, agentName: "my-agent" });
const sig = await agent.sign({ actionType: "api:openai:chat", context: { model: "gpt-4o" } });
console.log(sig.verificationUrl);
```

## Verify one, with no account

Receipts are portable. Anyone holding a `signature_id` can verify it without an API key.
`verify()` returns the public verdict and recomputes `chain_hash` locally, as the SHA-256
over the RFC 8785 canonical payload, so the chain link is reproducible on your machine.

```python
import asqav

result = asqav.verify("sig_example_regulator_cold_verify_2026")
print(result["verified"])     # False -- deliberately, see below
print(result["chain_hash"])
```

```ts
import { verify } from "@asqav/sdk";

const result = await verify("sig_example_regulator_cold_verify_2026");
console.log(result.verified, result.chainHash);
```

That id is a **shape example**: its signature bytes are placeholders and its `kid` resolves to
no key, so the verifier reports `verified: false` rather than waving it through. Swap in a
`signature_id` of your own for a receipt that passes. A verifier that says no when the
evidence is absent is the only kind worth having.

For a fully offline, zero-trust check that reproduces the signature itself, use
`asqav.verify_receipt_offline(receipt, jwks)` in Python, `verifyReceiptOffline()` in
TypeScript, or the standalone `python -m asqav.verifier.verify_receipt --offline`. The
algorithm is per-receipt from `signature.alg`: ML-DSA-65 for cloud-issued receipts,
Ed25519/ES256 for locally signed ones.

## Per-language guides

The package READMEs are the reference for each language — quick start, CLI, data-handling
modes, framework integrations, offline verification, and what a receipt does not prove:

- **[python/README.md](python/README.md)** — also rendered on [PyPI](https://pypi.org/project/asqav/)
- **[typescript/README.md](typescript/README.md)** — also rendered on [npm](https://www.npmjs.com/package/@asqav/sdk)

## Capture coverage

How each shipped adapter observes the interaction it signs. `harness_enforced`
fires from the harness itself; `framework_callback` from framework seams;
`in_path_proxy` from calls flowing through our wrapper; `passive` scans
without enforcing. `model_invoked` (the governance surface is itself a tool the model may
choose to skip) is the class we never build, so no row below uses it.

| adapter | coverage class | citation (URL) | accessed | version |
|---|---|---|---|---|
| openai (python) | in_path_proxy | https://github.com/openai/openai-python | 2026-09-21 | 3.16.2 |
| anthropic (python) | in_path_proxy | https://github.com/anthropics/anthropic-sdk-python | 2026-09-21 | 1.7.0 |
| letta (python) | in_path_proxy | https://docs.letta.com/ | 2026-10-10 | 1.12.1 |
| haystack (python) | in_path_proxy | https://docs.haystack.deepset.ai/docs/pipelines | 2026-09-21 | 3.1.1 |
| smolagents (python) | in_path_proxy | https://huggingface.co/docs/smolagents/index | 2026-09-21 | 1.26.0 |
| langchain (python) | framework_callback | https://reference.langchain.com/python/langchain-core/callbacks | 2026-09-21 | 1.4.2 |
| litellm (python) | framework_callback | https://docs.litellm.ai/docs/observability/custom_callback | 2026-09-21 | 1.102.0 |
| llamaindex (python) | framework_callback | https://docs.llamaindex.ai/en/stable/module_guides/observability/callbacks/ | 2026-09-21 | 0.14.24 |
| dspy (python) | framework_callback | https://dspy.ai/current/tutorials/observability/ | 2026-09-21 | 3.3.1 |
| instructor (python) | framework_callback | https://python.useinstructor.com/concepts/hooks/ | 2026-09-21 | 1.17.0 |
| crewai (python) | framework_callback | https://docs.crewai.com/en/concepts/tasks | 2026-09-21 | 1.15.22 |
| openai_agents (python) | framework_callback | https://openai.github.io/openai-agents-python/guardrails/ | 2026-09-21 | 0.22.3 |
| strands (python) | framework_callback | https://strandsagents.com/docs/learning/control-your-agent-with-hooks/ | 2026-10-10 | 1.56.0 |
| opa_detector (python) | passive | https://www.openpolicyagent.org/docs/latest/ | 2026-09-21 | v1.20.2 |
| presidio_detector (python) | passive | https://data-privacy-stack.github.io/presidio/ | 2026-09-21 | 2.2.364 |
| cli_hook (harness hook) | harness_enforced | https://code.claude.com/docs/en/hooks | 2026-09-21 | 2.1.278 |
| langchain (typescript) | framework_callback | https://reference.langchain.com/javascript/langchain-core/callbacks/base/BaseCallbackHandler | 2026-09-21 | 1.2.12 |
| openai-agents (typescript) | framework_callback | https://openai.github.io/openai-agents-js/ | 2026-09-21 | 0.18.0 |
| mastra (typescript) | framework_callback | https://mastra.ai/docs/agents/using-tools | 2026-09-21 | 1.67.0 |
| vercel-ai (typescript) | framework_callback | https://ai-sdk.dev/docs/ai-sdk-core/telemetry | 2026-09-21 | 7.0.107 |
| opa (typescript) | passive | https://www.openpolicyagent.org/docs/latest/ | 2026-09-21 | v1.20.2 |
| presidio (typescript) | passive | https://data-privacy-stack.github.io/presidio/ | 2026-09-21 | 2.2.364 |

No `model_invoked` surface is ever built; an MCP-server request is answered
with the harness hook plus the SDK. smolagents is `in_path_proxy` with the local
executor (the default): `wrap_tool` replaces the tool's `forward`, so every call
runs through our wrapper. Remote executors (docker, e2b, modal, blaxel) rebuild
the tool from its class source and drop the wrapper, so nothing is signed there. Hook handlers run under per-hook-family
timeouts (600 seconds for command, HTTP and MCP-tool hooks; 30 for prompt
hooks; 60 for agent hooks, see the hooks reference above).

## Repository layout

```
asqav-sdk/
  python/           Python SDK (pip install asqav)
  typescript/       TypeScript SDK (npm install @asqav/sdk)
  conformance/      Cross-language fixtures both SDKs run against
  verifier/         Neutral multi-format verifier + its conformance corpus
  .github/workflows/
    ci.yml          Path-filtered CI for both languages
    publish.yml     Tag-based publish (py-v* to PyPI, ts-v* to npm)
```

The two SDKs are versioned and released independently.

## Conformance

`conformance/` holds shared fixtures both SDKs run against. Adding a feature means adding a
fixture first, then making both SDKs pass it. CI reruns both matrices whenever `conformance/`
changes even if neither language directory was touched, so cross-language drift is caught at
PR time. The TypeScript verifier is held to verdict parity with the Python oracle by that
same corpus.

## CI

Every push to `main` and every pull request. Only the language whose source changed is
tested: pytest on Python 3.10/3.11/3.12, npm test on Node 20/22; a `conformance/` change runs
both. The aggregator job `ci-ok` is the single required status check.

`asqav doctor` validates configuration and connectivity in any environment with
`ASQAV_API_KEY` set and returns non-zero on failure, so it gates your own PRs cleanly. See
[docs/github-actions.md](docs/github-actions.md) for a copy-paste workflow.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version:

1. Fork and branch.
2. Run the relevant suite:
   - Python: `cd python && pip install -e ".[all,dev]" && pytest tests -v`
   - TypeScript: `cd typescript && npm ci && npm test`
3. Open a PR against `main`. CI must be green to land. Direct pushes to `main` are blocked.

**Releases** are driven by prefixed tags: `py-v*` publishes to PyPI via OIDC, `ts-v*`
publishes to npm via the `NPM_TOKEN` secret. Bump the manifest, update
[CHANGELOG.md](CHANGELOG.md), then tag and push.

## Plans

Asqav has three plans: Free, Pro and Enterprise. Free needs no credit card, Pro is a paid
monthly or annual subscription, and Enterprise is custom and volume-priced.

Quotas, prices and the per-plan feature list live on one page so they cannot drift out of
step with each other: see [asqav.com/pricing](https://asqav.com/pricing.html). Where a
specific capability is plan-gated, the section describing that capability says so.

## Standards

Profiled in the IETF Internet-Draft
[`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/),
an Independent Submission profiling
[`draft-farley-acta-signed-receipts`](https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/).
Aligns with NIST FIPS 204 (ML-DSA), RFC 8785 (JCS) and NSA CSI U/OO/6030316-26.

## Ecosystem

- [`asqav` on PyPI](https://pypi.org/project/asqav/) — the Python SDK
- [`@asqav/sdk` on npm](https://www.npmjs.com/package/@asqav/sdk) — the TypeScript SDK
- [asqav-compliance](https://github.com/jagmarques/asqav-compliance) — CI/CD compliance scanner

## Links

[Docs](https://asqav.com/docs) · [SDK guide](https://asqav.com/docs/sdk) ·
[Discovery descriptor](https://asqav.com/.well-known/governance.json)

## License

Source-available client SDK under Elastic License 2.0; open verifier and conformance vectors under Apache-2.0; commercial signing service. Get an API key at [asqav.com](https://asqav.com).
