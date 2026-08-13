# LiteLLM integration (cloud-signed receipts)

LiteLLM ships a built-in `asqav` callback that writes a local, tamper-evident
audit ledger — no setup, no account. That is the on-ramp. This page covers the
next step: [asqav](https://pypi.org/project/asqav/)'s cloud signing, which
turns each LiteLLM call into a third-party-verifiable receipt.

## The two layers

LiteLLM's built-in callback (class `AsqavLogger`, registered as the string
`asqav`) appends one record per call to a local JSONL file, each record
carrying a SHA-256 chain hash over its own fields plus the previous record's
hash. The sequence is tamper-evident and verifiable offline with stdlib tools.

It is deliberately local-first:

- no network, no asqav account, no keys;
- the chain proves a record was not altered after the fact;
- it does not produce a third-party-verifiable receipt — anyone with write
  access to the file could have authored the whole chain.

asqav's cloud signing adds the missing layer. For each call it signs the
call's mapped fields and content digests with the agent's key through the
asqav cloud, producing an ML-DSA receipt that any third party can verify
against the agent's published key — without trusting the machine that logged
the call.

## Enable the local ledger (no account needed)

```python
import litellm

litellm.callbacks = ["asqav"]
```

That is all. Every successful call appends a chained record to the local
ledger. Use this when you want tamper-evidence with zero external dependency.

## Enable cloud signing (asqav account)

Install the extra:

```bash
pip install asqav[litellm]
```

Turn it on:

```python
import asqav
import litellm
from asqav.extras.litellm import AsqavSigningLogger

asqav.init("sk_live_...")
litellm.callbacks = [AsqavSigningLogger(agent_name="my-llm-agent")]

response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "hello"}],
)
```

Each successful call now signs a receipt via the asqav cloud and appends one
local index line carrying the `signature_id` and `verification_url`, stitched
to the call's content digests. The local index and the cloud receipt are
verifiable together later.

The two layers can run side by side:

```python
litellm.callbacks = ["asqav", AsqavSigningLogger(agent_name="my-llm-agent")]
```

### Guardrail seam

`AsqavGuardrail` signs the call lifecycle instead of (or as well as) the
completed call — pre-call, post-call, error, and moderation events:

```python
from asqav.extras.litellm import AsqavGuardrail

litellm.callbacks = [AsqavGuardrail(agent_name="my-llm-agent")]
```

### Verify the local index

```bash
asqav verify-litellm-log
```

Checks the local JSONL index and reports the signed receipts it can verify.

## Configuration

| Env var | Meaning | Default |
| --- | --- | --- |
| `ASQAV_API_KEY` | asqav key (or `asqav.init(...)` / `api_key=`) | required for cloud signing |
| `ASQAV_LOG_PATH` | local index path | `~/.litellm_asqav_audit.jsonl` |
| `ASQAV_REDACT_CONTENT` | `"false"` stores raw prompt/response in the index | digest-only (`true`) |

`AsqavSigningLogger` also accepts `api_key`, `agent_name`, `agent_id`,
`log_path`, and `redact_content` overrides.

## Notes

- Signing is fail-soft: a signing or network error never raises into the
  LiteLLM call. Signing is evidence, not enforcement.
- Redaction is on by default: the signed context carries content digests, not
  raw prompt/response, so a receipt proves a payload was present without
  reconstructing it.
- Streamed calls expose no aggregated response object, so
  `response_content_digest` is `None` for them (the request side still signs).
- With no resolvable key the logger warns once and no-ops — it never silently
  pretends to sign.
- LiteLLM is an optional extra; the module raises a clear install error only
  when used without it.
