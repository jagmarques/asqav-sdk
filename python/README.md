# asqav

[![GitHub stars](https://img.shields.io/github/stars/jagmarques/asqav-sdk?style=social)](https://github.com/jagmarques/asqav-sdk)

Python SDK for [asqav.com](https://asqav.com), the evidence layer for AI agents.

Every agent action gets a signed, hash-chained **compliance receipt**: ML-DSA-65 (FIPS 204,
post-quantum), timestamped against independent witnesses, and verifiable by anyone —
auditor, counterparty, regulator — **without an Asqav account and without trusting us**.

Zero native dependencies. Cryptography runs server-side.

## Install

```bash
pip install asqav
```

## Quick start

```bash
pip install "asqav[cli]"
asqav login        # validates your key, saves it to ~/.asqav/credentials
```

```python
import asqav

# govern() = init() + Agent.create() in one call
agent = asqav.govern(api_key="sk_...", agent_name="my-agent")

sig = agent.sign("api:openai:chat", {"model": "gpt-4o"})

print(sig.action_ref)             # "sha256:..." over the JCS-canonical action
print(sig.previous_receipt_hash)  # 64 hex; "0"*64 on this agent's first receipt
print(sig.verification_url)       # anyone can open this
```

One install, one `govern`, one `sign`. `asqav.init()` + `asqav.Agent.create()` remain
available when you want control over `algorithm`, `capabilities` and other agent options.

## Verify it without an account

This is the point of the whole thing — the receipt stands on its own:

Run this right now, with no key and no signup:

```python
import asqav

result = asqav.verify("sig_example_regulator_cold_verify_2026")
print(result["verified"])     # False -- and that is the point, see below
print(result["chain_hash"])   # recomputed on your machine from the canonical bytes
```

That id is a **shape example**: its signature bytes are placeholders and its `kid` resolves
to no key, so the verifier returns `verified: false` instead of waving it through. Swap in a
`signature_id` of your own for a receipt that passes. A verifier that says no when the
evidence is absent is the only kind worth having.

From the shell (the `cli` extra provides the `asqav` command):

```bash
pip install "asqav[cli]"
asqav verify <your_signature_id>
```

Offline or air-gapped, snapshot the keys once and verify with no network at all. This
re-derives the signature itself, so it needs the `verify` extra
(`dilithium-py` for ML-DSA-65, `cryptography` for the Ed25519 and ES256 axes — without it
those signatures report INCOMPLETE rather than verifying):

```bash
pip install "asqav[verify]"
```

```python
import asqav, json

jwks = asqav.fetch_jwks()                     # online, once
json.dump(jwks, open("jwks.json", "w"))

receipt = json.load(open("receipt.json"))     # offline from here
result  = asqav.verify_receipt_offline(receipt, json.load(open("jwks.json")))
assert result["verdict"] == "PASS", result["axes"]
```

Pass `predecessor=prev_receipt` to check the hash-chain link too.
Guide: [offline and air-gapped verification](https://asqav.com/docs/offline-air-gapped-verification).

## No account? Queue locally

```python
from asqav.local import local_sign, LocalQueue

local_sign("my-agent", "api:openai:chat", {"model": "gpt-4o"})  # -> ~/.asqav/queue/

import asqav
asqav.init(api_key="sk_...")
LocalQueue().sync()            # {"synced": N, "failed": M}
```

## Data handling modes

The SDK picks the safer default for where you point it:

- **Cloud (`*.asqav.com`)** — hash-only. A SHA-256 fingerprint is computed locally and only
  the hash plus `action_type`, `agent_id`, `session_id`, `model_name`, `tool_name` is sent.
  Prompts and tool arguments never leave your side.
- **Self-hosted** — full payload, so the server can run policy checks and richer audit.

```python
asqav.init(api_key="...", base_url="https://api.asqav.com", mode="hash-only")
```

## High-value actions

For regulated or high-risk actions, pass envelope fields that an auditor will look for:

```python
sig = agent.sign(
    "payment.wire_transfer",
    {"amount_eur": 850000, "beneficiary_iban": "DE89370400440532013000"},
    receipt_type="protectmcp:decision",
    risk_class="high",                 # low | medium | high | unknown
    issuer_id="legal:Acme GmbH",       # LEI, EIN, CIK or W3C DID
    iteration_id="task-2026-Q2-4821",  # logical task, distinct from session
)
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

Native callbacks under `asqav.extras.*` for **LangChain**, **CrewAI**, **LiteLLM**,
**OpenAI Agents** and others, plus a pytest plugin. Adapters install behind documented
extras (`asqav[langchain]`, `asqav[litellm]`, `asqav[openai-agents]`).
See [integrations](https://asqav.com/docs/integrations) and
[pytest plugin](https://asqav.com/docs/pytest-plugin).

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

Python 3.10+. Uses `httpx`. Zero native dependencies.

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
