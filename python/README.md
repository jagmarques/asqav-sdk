# asqav

Python SDK for [asqav.com](https://asqav.com). All ML-DSA cryptography runs server-side. Drop-in for AI agent governance: audit trails, policy enforcement, compliance.

## Install

```bash
pip install asqav
```

## Quick start

```python
import asqav

asqav.init(api_key="sk_...")
agent = asqav.Agent.create("my-agent")
sig = agent.sign("api:call", {"model": "gpt-4"})

print(sig.verification_url)
```

Each signed action is recorded server-side with an ML-DSA-65 (FIPS 204) signature, a chain hash, and a public verification URL.

## CLI

The package ships an `asqav` CLI mirroring the Python API. Set `ASQAV_API_KEY` and run:

```bash
asqav verify <signature_id>
asqav agents list / create / revoke
asqav sessions list / end
asqav replay <agent_id> <session_id>          # Pro
asqav preflight <agent_id> <action_type>      # Pro
asqav budget check / record                   # Pro
asqav approve <session_id> <entity_id>        # Pro
asqav compliance frameworks / export          # Business
asqav policies / webhooks list / create / delete   # Pro
```

Pro and Business commands are gated client-side via `GET /account` so a free-tier key gets a clean upgrade message instead of a mid-pipeline 402.

## Roadmap

Six-line view of what is shipped on Asqav:

- Hash-only mode for cloud - Today (default for `*.asqav.com`).
- Self-hosted signer (split-trust) - Today.
- Bring-your-own KMS (AWS KMS / GCP KMS) - Today, Enterprise tier.
- Customer-owned storage - Today (self-hosted; relay payload allowlist enforced in code).
- SCITT / COSE_Sign1 receipt export - Today (public `GET /api/v1/signatures/{id}/cose` returns `application/cose`).
- Air-gapped / on-prem mode - Today (offline license + zero-egress, see `docs/airgapped-mode.md` in the backend repo).

See the docs at <https://asqav.com/docs> for the current feature set.

## Documentation

- Repository: <https://github.com/jagmarques/asqav-sdk>
- Full docs: <https://asqav.com/docs>
- Roadmap: <https://asqav.com/roadmap>

## License

MIT. Get an API key at [asqav.com](https://asqav.com).
