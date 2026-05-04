# `asqav` CLI reference

The `asqav` CLI ships in the `asqav[cli]` extra. Every command wraps a public Python API; the same arguments accepted on the command line work as kwargs on the corresponding SDK call. Pro and Business commands fail with a clean upgrade message when the calling org is below the required tier (gated client-side via `GET /account`; the cloud is the source of truth).

```bash
pip install asqav[cli]
export ASQAV_API_KEY=sk_...
```

## Public commands

### `asqav verify <signature_id> [--output text|json]`

Public, no API key required. Prints the verification outcome plus the IETF Compliance Receipts profile sub-axes when the cloud emitted them: `chain_valid`, `anchor_status_ots`, `anchor_status_rfc3161`, `signed_at_skew_seconds`, `missing_fields`. With `--output json`, returns the full `VerificationResponse` dataclass as JSON.

```bash
asqav verify sig_abc123 --output json
```

## Compliance Receipts (IETF profile)

### `asqav sign`

Issues a Compliance Receipt under [`draft-marques-asqav-compliance-receipts-00`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/). Mirrors `Agent.sign(...)` end-to-end.

```bash
asqav sign \
  --agent-id agt_x7y8z9 \
  --action-type payment.wire_transfer \
  --action-json action.json \
  --compliance-mode \
  --receipt-type protectmcp:decision \
  --risk-class high \
  --sandbox-state enabled \
  --iteration-id task-2026-Q2-4821 \
  --issuer-id "legal:Acme GmbH" \
  --policy-decision permit
```

Flags:

| Flag | SDK kwarg | Notes |
| --- | --- | --- |
| `--action-id` | `_action_id` (in context) | Stable client-side action id; cloud allocates one when empty. |
| `--action-type` | `action_type` | Required. |
| `--compliance-mode/--no-compliance-mode` | `compliance_mode` | Default false. |
| `--action-ref` | `action_ref` | `sha256:<hex>`; auto-derived from `--action-json` when missing. |
| `--action-json <path|->` | `context` | File path or `-` for stdin. |
| `--sandbox-state` | `sandbox_state` | `enabled|disabled|unavailable`. |
| `--iteration-id` | `iteration_id` | Distinct from session_id. |
| `--risk-class` | `risk_class` | `low|medium|high|unknown`. |
| `--incident-class` | `incident_class` | DORA ITS vocabulary. |
| `--issuer-id` | `issuer_id` | Overrides server resolution. |
| `--receipt-type` | `receipt_type` | `protectmcp:decision|protectmcp:restraint|protectmcp:lifecycle`. Default `protectmcp:decision`. |
| `--reason` | `reason` | Required when `--policy-decision` is `deny|rate_limit`. |
| `--policy-decision` | `policy_decision` | Default `permit`. |
| `--algorithm` | `algorithm` | Cloud uses the agent's algorithm; CLI warns on mismatch. |
| `--session-id` | `agent._session_id` | Bind the record to an existing session. |
| `--valid-seconds` | `valid_seconds` | Validity window. |
| `--policy-artefact <path|->` | `_policy_artefact` (in context) | Cloud stores; receipt's `policy_digest` is its content hash. |
| `--output` | n/a | `text` (default) or `json`. |

### `asqav audit-pack export`

Exports a signed Audit Pack bundle for receipts in `[start, end)`. The cloud signs the JCS-canonical bundle bytes with the org's most-recent active agent key so verifiers can rederive the digest offline.

```bash
asqav audit-pack export \
  --start 2026-04-01T00:00:00Z \
  --end   2026-05-01T00:00:00Z \
  --only-compliance \
  --output-file bundle.json
```

### `asqav audit-pack policy <digest>`

Resolves a `policy_digest` to the retained policy artefact JSON. Accepts either `sha256:<64-hex>` or a bare 64-hex string (the prefix is added).

```bash
asqav audit-pack policy a3f6...c91d --output text
```

### `asqav replay-verify <agent_id> <session_id> [--strict]`

Re-derives the IETF chain over `sha256(canonical_json(signed_envelope))`. With `--strict` rejects any step that lacks a cloud-emitted `signed_envelope` (legacy synthetic shape cannot prove byte-level integrity).

```bash
asqav replay-verify agt_x7y8z9 sess_abc --strict --output json
```

### `asqav payloads erase <signature_id>`

Right-to-erasure (P4). Calls `DELETE /signatures/payloads/{signature_id}`. Idempotent; the signature, chain hash, and anchors stay intact so the receipt remains verifiable.

```bash
asqav payloads erase sig_abc123 --yes
```

### `asqav org set-compliance-strict <org_id> --enable|--disable`

Toggles per-org `compliance_mode_strict` (M-NEW-1). When enabled, the cloud rejects any sign request that is not flagged `compliance_mode=true` with HTTP 412.

```bash
asqav org set-compliance-strict org_abc --enable
```

### `asqav keys generate --algorithm <ml-dsa-65|ed25519|es256>`

Generates a local keypair for offline / air-gapped flows. Ed25519 and ES256 emit PKCS#8 PEM. ML-DSA-65 raises a clear error pointing at `asqav agents create` (server-side keygen).

```bash
asqav keys generate --algorithm ed25519 --out priv.pem
```

## Operator commands

### `asqav migrate run <v3-20|v3-21|v3-22>`

Triggers a maintenance migration via `POST /maintenance/run-migration-{name}`. Reads the maintenance key from `ASQAV_MAINTENANCE_KEY` and refuses to run when missing.

```bash
ASQAV_MAINTENANCE_KEY=mk_... asqav migrate run v3-20
```

| Migration | Purpose |
| --- | --- |
| v3-20 | IETF profile schema additions on `signature_records` + `policy_artefacts` table. |
| v3-21 | Partial unique index defending the IETF receipt chain against forks. |
| v3-22 | `organizations.compliance_mode_strict` boolean (M-NEW-1). |
