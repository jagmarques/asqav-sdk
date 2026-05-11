# asqav

Python SDK for [asqav.com](https://asqav.com), the evidence layer for AI agents. All ML-DSA cryptography runs server-side. Drop-in audit trails, approvals, policy gates, and compliance reports.

## Install

```bash
pip install asqav
```

## Quick start

```python
import asqav

asqav.init(api_key="sk_...")
agent = asqav.Agent.create("my-agent")

sig = agent.sign(
    "payment.wire_transfer",
    {"amount_eur": 850000, "beneficiary_iban": "DE89370400440532013000"},
    receipt_type="protectmcp:decision",
    risk_class="high",
    issuer_id="legal:Acme GmbH",
    iteration_id="task-2026-Q2-4821",
)

print(sig.compliance_mode)        # True (default; pass compliance_mode=False to opt out)
print(sig.action_ref)             # "sha256:..." over the JCS-canonical action
print(sig.previous_receipt_hash)  # 64 hex; "0"*64 on the first record per agent
print(sig.verification_url)
```

Each signed action lands on a Compliance Receipt under IETF Internet-Draft [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/) by default: ML-DSA-65 (FIPS 204) signature, chain hash, retained `policy_digest`, fail-closed anchoring, and a public verification URL. Pass `compliance_mode=False` if you want a non-Compliance receipt.

## CLI

The package ships an `asqav` CLI mirroring the Python API. Set `ASQAV_API_KEY` and run:

```bash
asqav verify <signature_id> [--output json]   # IETF axes when present
asqav sign --agent-id ID --action-type T --action-json action.json \
           --compliance-mode --receipt-type protectmcp:decision \
           --risk-class high --issuer-id legal:Acme
asqav agents list / create / revoke
asqav sessions list / end
asqav replay <agent_id> <session_id>          # Pro
asqav replay-verify <agent_id> <session_id> [--strict]   # IETF chain
asqav preflight <agent_id> <action_type>      # Pro
asqav budget check / record                   # Pro
asqav approve <session_id> <entity_id>        # Pro
asqav compliance frameworks / export          # Business
asqav audit-pack export --start ISO --end ISO --output-file bundle.json
asqav audit-pack policy <sha256:hex>
asqav payloads erase <signature_id>           # P4: GDPR right-to-erasure
asqav org set-compliance-strict <org_id> --enable|--disable
asqav keys generate --algorithm ed25519|es256 [--out priv.pem]
asqav migrate run v3-20|v3-21|v3-22           # X-Maintenance-Key required
asqav policies / webhooks list / create / delete   # Pro
```

Pro and Business commands are gated client-side via `GET /account` so a free-tier key gets a clean upgrade message instead of a mid-pipeline 402.

The IETF Compliance Receipts profile commands (`sign --compliance-mode`, `audit-pack export`, `audit-pack policy`, `payloads erase`, `replay-verify --strict`, `org set-compliance-strict`) match the SDK kwargs on `Agent.sign(...)` and `verify_compliance_receipt(...)`. See `docs/CLI.md` for full flag reference.

## Roadmap

Six-line view of what is shipped on Asqav:

- Hash-only mode for cloud - Today (default for `*.asqav.com`).
- Self-hosted signer (split-trust) - Today.
- Bring-your-own KMS (AWS KMS / GCP KMS) - Today, Enterprise tier.
- Customer-owned storage - Today (self-hosted; relay payload allowlist enforced in code).
- SCITT / COSE_Sign1 receipt export - Today (public `GET /api/v1/signatures/{id}/cose` returns `application/cose`).
- Air-gapped / on-prem mode - Today (offline license + zero-egress, see `docs/airgapped-mode.md` in the backend repo).

See the docs at <https://asqav.com/docs> for the current feature set.

## Standards

Asqav's compliance receipts are profiled in IETF Internet-Draft [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/), profiling the upstream [`draft-farley-acta-signed-receipts`](https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/) for EU AI Act Articles 12 and 26, and DORA Article 17 bindings.

## Compliance receipts (IETF profile)

Compliance Receipts are the SDK default. Each `agent.sign(...)` call produces a receipt that conforms to [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/): ML-DSA-65 signature, RFC 3161 + OpenTimestamps anchors, retained `policy_digest`, hash-chained `previousReceiptHash`. Opt out with `compliance_mode=False` if you want the older shape.

The four envelope extensions most callers reach for:

- `receipt_type` - `protectmcp:decision`, `protectmcp:restraint`, or `protectmcp:lifecycle`.
- `risk_class` - controlled vocabulary: `unacceptable | high | limited | minimal | gpai | low | medium | unknown`.
- `iteration_id` - logical task id, distinct from session.
- `sandbox_state` - `enabled | disabled | unavailable` for high-risk gating.
- `incident_class` - DORA / NYDFS / CIRCIA token (or array of tokens).
- `issuer_id` - LEI (ISO 17442), EIN, CIK, or a W3C DID for non-LEI deployers.

### Audit Pack export

The cloud signs a Compliance Audit Pack (per IETF -03 Section 7) over a window of receipts. The SDK wraps the endpoint:

```python
pack = asqav.fetch_audit_pack(start="2026-05-01T00:00Z", end="2026-06-01T00:00Z")
print(pack["bundle_digest"])              # sha256:<hex>
print(pack["bundle_signature"])           # base64 ML-DSA-65 sig over the bundle
print(pack["regime_mapping"])             # {regime_token: [record_id, ...]}
print(pack["algorithm_registry_version"]) # registry version pinned at issuance
```

`asqav.export_bundle(signatures, framework="dora")` is the offline alternative for air-gapped flows: it computes a Merkle root over an in-memory list of receipts without calling the cloud. Use `fetch_audit_pack` whenever the cloud is reachable, since only the cloud signature gives the auditor a tamper-evident manifest.

Local-side sanity checks (presence of REQUIRED fields, namespace, 300s skew bound, predecessor rederivation) are available as `asqav.verify_compliance_receipt(envelope, predecessor_envelope=...)`. The cloud is the authoritative verifier; this helper is a convenience.

Algorithm agility per profile section 10.8 is exposed via `asqav.SUPPORTED_ALGORITHMS`. Pass `algorithm="ed25519"` or `"es256"` to `Agent.create(...)` for non-post-quantum identities, or `asqav.generate_local_keypair("ed25519")` for offline scenarios.

## Documentation

- Repository: <https://github.com/jagmarques/asqav-sdk>
- Full docs: <https://asqav.com/docs>

## License

MIT. Get an API key at [asqav.com](https://asqav.com).
