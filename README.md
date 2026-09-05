# Asqav SDK — Python + TypeScript

[![GitHub stars](https://img.shields.io/github/stars/jagmarques/asqav-sdk?style=social)](https://github.com/jagmarques/asqav-sdk)
[![PyPI](https://img.shields.io/pypi/v/asqav)](https://pypi.org/project/asqav/)
[![npm](https://img.shields.io/npm/v/@asqav/sdk)](https://www.npmjs.com/package/@asqav/sdk)

Client SDKs for [asqav.com](https://asqav.com), the evidence layer for AI agents.

Every agent action gets a signed, hash-chained **compliance receipt**: ML-DSA-65 (FIPS 204,
post-quantum), timestamped against independent witnesses, and verifiable by anyone —
auditor, counterparty, regulator — **without an Asqav account and without trusting us**.

Zero native dependencies in either SDK. Cryptography runs server-side.

## Install

```bash
pip install asqav            # Python 3.10+
npm install @asqav/sdk       # Node 20+
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

Elastic License 2.0. Get an API key at [asqav.com](https://asqav.com).
