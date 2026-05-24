# Asqav shadow-AI shim

Drop-in egress proxy that signs every outbound LLM request through the
co-located asqav-signer and forwards the request upstream unchanged.
Sign-only, fail-open: if the signer is unreachable the request still
goes out.

## Quickstart

```bash
asqav shadow-ai init --dir ./shadow-ai
cd shadow-ai
cp .env.template .env
# fill in ASQAV_API_KEY, ASQAV_AGENT_ID, POSTGRES_PASSWORD
asqav shadow-ai up --dir .
asqav shadow-ai status --dir .
```

Point your application at `http://localhost:8080` instead of
`https://api.openai.com` (or `https://api.anthropic.com`) and every
request will be signed before it leaves the host.

## What the shim does

1. Receives the upstream-shaped request on `:8080`.
2. Calls the local `asqav-signer` to mint a Compliance Receipt.
3. Forwards the original request to the configured upstream.
4. Returns the upstream response unchanged.

The signer writes receipts to its embedded Postgres so the audit trail
survives shim restarts.

## Files in this directory

- `docker-compose.yml` - the two-container stack (shim + signer).
- `.env.template` - required environment variables.
- `secrets/license.json` - your Asqav signer license, mounted read-only.

## CLI commands

| Command | What it does |
| --- | --- |
| `asqav shadow-ai init` | Scaffold this directory. |
| `asqav shadow-ai up` | `docker compose up -d --build`. |
| `asqav shadow-ai down` | `docker compose down`. |
| `asqav shadow-ai status` | Probe shim + signer health endpoints. |
| `asqav shadow-ai logs` | `docker compose logs`. |

Full setup notes live at https://asqav.com/docs/shadow-ai.
