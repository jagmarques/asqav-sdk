# Self-Hosted Signer

Run the Asqav signing path on your own infrastructure. ML-DSA-65
private keys, raw prompt text, and raw reasoning traces stay inside
the container boundary. Only signature digests, signature bytes,
timestamps, and structural metadata ever leave your network. This is
the split-trust deployment model.

For the full reference (architecture diagram, environment contract,
upstream relay semantics, QTSP / RFC 3161 wiring), see the cloud-side
doc: https://www.asqav.com/docs/self-hosted-signer

## When to pick this mode

- EU operators who must keep prompt / trace content in-region.
- Regulated institutions where ML-DSA private keys cannot cross an
  organizational trust boundary.
- Air-gapped operators who must refuse all outbound HTTP. See
  https://www.asqav.com/docs/airgapped-mode for the air-gapped
  variant.

## Stand up the signer

The reference docker-compose for the signer plus its Postgres and
Redis dependencies is published with the cloud repo:

```bash
curl -O https://raw.githubusercontent.com/jagmarques/asqav/main/docker-compose.signer.yml
```

Create a `.env` next to it with at minimum:

```bash
POSTGRES_PASSWORD=<a strong random value, 32+ chars>
BUILD_SHA=unknown
```

Bring it up:

```bash
docker compose -f docker-compose.signer.yml up -d
```

The signer exposes the Asqav API surface on port `8000` inside the
compose network. Front it with your own ingress (Traefik, nginx,
Caddy, ALB) and terminate TLS there. The signer container does not
ship a TLS terminator.

## Point the SDK at your signer

The SDK reads `ASQAV_API_URL` at import time, falling back to
`https://api.asqav.com/api/v1`. Override it to your signer's URL:

```bash
export ASQAV_API_URL=https://signer.internal.example.com/api/v1
export ASQAV_API_KEY=sk_self_hosted_xxx
```

```python
import asqav

asqav.init()  # uses ASQAV_API_URL + ASQAV_API_KEY from environment

agent = asqav.Agent.create(name="my-agent")
signed = agent.sign_action(
    "tool:call",
    {"tool": "search", "query": "EU AI Act Article 12"},
)
print(signed.signature_id, signed.signed_at)
```

You can also pass `base_url` explicitly when constructing the client
for tests or multi-tenant setups:

```python
from asqav import client

client.configure(
    api_key="sk_self_hosted_xxx",
    base_url="https://signer.internal.example.com/api/v1",
)
```

## What flows where

| Boundary | What enters | What leaves |
| --- | --- | --- |
| Agent host | `action_type`, full context (prompt, tool args) | nothing |
| Signer container | the above | `signature_id`, signature bytes, hash chain, timestamp, structural metadata |
| Upstream relay (optional) | digest + signature + timestamp + metadata | none of the raw context |

The agent's prompt and tool arguments never cross the signer
boundary on the wire. If you also enable the optional upstream relay
for aggregation, the cloud receives only the digest, signature, and
metadata, not the raw context.

## Hash-only mode

Hash-only mode is the cloud default, and it remains the default when
you point the SDK at a self-hosted signer. The SDK computes RFC 8785
JCS + SHA-256 locally and only the hash plus a small whitelisted
metadata bag leaves the agent host. See
https://www.asqav.com/docs/fingerprint-spec for the canonical form.

## Air-gapped variant

For deployments that must refuse all outbound HTTP, set
`ASQAV_AIRGAPPED=1` and provide an offline license per the
`docker-compose.signer.yml` comments. Bitcoin anchoring, the upstream
relay, and update checks are disabled in this mode; signing, replay,
and compliance bundle export still work locally. See
https://www.asqav.com/docs/airgapped-mode.

## Verifying a signed receipt

Independent verification does not require any Asqav infrastructure.
The audit-pack export carries the public ML-DSA-65 key, the hash
chain, and the timestamp tokens. Anyone with the exported pack can
verify offline:

```bash
asqav verify --pack ./compliance-bundle.json
```

The same audit-pack also feeds the IETF Compliance Receipts profile
(`draft-marques-asqav-compliance-receipts`), so third-party
verifiers can consume it without an Asqav SDK.

## Troubleshooting

- **`asqav doctor` reports "signer unreachable"**: confirm your
  ingress is fronting the compose `asqav-signer` service on the host
  you set in `ASQAV_API_URL` and that TLS is terminated correctly.
- **`POSTGRES_PASSWORD must be set in .env`**: the compose file
  refuses to start without an explicit password, by design. Generate
  one with `openssl rand -base64 36` and write it to `.env`.
- **Hash chain breaks after restore from backup**: restore must
  include the `signature_records` table in full; the hash chain is
  per-org and rebuilds deterministically from the last anchored
  receipt. Contact info@asqav.com if you need help.
