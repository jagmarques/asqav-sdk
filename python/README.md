# asqav

Python SDK for [asqav.com](https://asqav.com), the evidence layer for AI agents. Sign every agent action with ML-DSA-65 (FIPS 204), enforce policies before execution, and produce regulator-ready audit trails. All cryptography runs server-side; the package has zero native dependencies.

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
asqav sessions list / end <session_id>
asqav replay <agent_id> <session_id>                       # Pro
asqav replay-verify <agent_id> <session_id> [--strict]     # Pro: IETF chain
asqav preflight <agent_id> <action_type>                   # Pro
asqav budget check / record                                # Pro
asqav approve <session_id> <entity_id>                     # Pro
asqav compliance frameworks / export                       # Business
asqav audit-pack export --start ISO --end ISO --output-file bundle.json
asqav audit-pack policy <sha256:hex>
asqav payloads erase <signature_id>                        # P4 right-to-erasure
asqav org set-compliance-strict <org_id> --enable|--disable
asqav keys generate --algorithm ed25519|es256 [--out priv.pem]
asqav migrate run v3-20|v3-21|v3-22                        # X-Maintenance-Key required
asqav policies / webhooks list / create / delete           # Pro
```

Pro and Business commands are gated client-side via `GET /account` so a free-tier key gets a clean upgrade message instead of a mid-pipeline 402. The server is the source of truth; self-hosted deployments without `/account` skip the gate.

## Data handling modes

The SDK auto-detects whether you're pointing at the Asqav cloud or a self-hosted deployment and selects the safer default for each:

- **Cloud (`*.asqav.com`)**: hash-only by default. The SDK builds a fingerprint of your action context, computes a SHA-256 hash locally, and sends only the hash plus a small metadata bag (`action_type`, `agent_id`, `session_id`, `model_name`, `tool_name`). Raw prompts and tool arguments stay on your side.
- **Self-hosted**: full-payload by default. The server can run policy checks, PII redaction, and richer audit. Recommended when you control the deployment.

Override anytime:

```python
asqav.init(api_key="...", base_url="https://api.asqav.com", mode="hash-only")
```

The fingerprint format is sorted JSON with no whitespace per JCS, hashed with SHA-256. See `docs/fingerprint-spec.md` and `conformance/vectors.json` for the spec and cross-language test vectors.

## Compliance receipts (IETF profile)

Compliance Receipts are the SDK default. Each `agent.sign(...)` call produces a receipt that conforms to [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/): ML-DSA-65 signature, JCS canonicalization, retained `policy_digest`, hash-chained `previous_receipt_hash`, OpenTimestamps anchoring. Opt out with `compliance_mode=False` if you want the older shape.

The envelope extensions most callers reach for:

- `receipt_type` - `protectmcp:decision`, `protectmcp:restraint`, `protectmcp:lifecycle`, `protectmcp:lifecycle:configuration_change`, `protectmcp:acknowledgment`, `protectmcp:observation`, or `protectmcp:observation:result_bound` (observation receipts that bind tool output via `result_digest`).
- `risk_class` - controlled vocabulary: `unacceptable | high | limited | minimal | gpai | low | medium | unknown`.
- `iteration_id` - logical task id, distinct from session.
- `sandbox_state` - `enabled | disabled | unavailable` for high-risk gating.
- `incident_class` - DORA / NYDFS / CIRCIA token (or array of tokens).
- `issuer_id` - LEI (ISO 17442), EIN, CIK, or a W3C DID for non-LEI deployers.

### Shadow AI capture (passive_telemetry)

Two `receipt_type` values cover the gating axis: `protectmcp:decision` records that a policy ran and gated the action; `protectmcp:observation` records that a passive monitor saw the event without gating it. Pick `observation` when the producer never had the option to block (SIEM forwarder, browser extension in observe-only mode, NetFlow-style proxy with no enforcement hook).

Set `capture_topology='passive_telemetry'` to declare the producer is observing after the fact. The SDK client-side check pre-flights the Asqav cloud's full rule 8 gate: a `capture_topology='passive_telemetry'` receipt MUST use `receipt_type='protectmcp:observation'`. Any other receipt_type paired with `passive_telemetry` (`:decision`, `:restraint`, `:lifecycle`, `:lifecycle:configuration_change`, `:acknowledgment`) raises `ValueError` with the verbatim `false_attestation_guard: capture_topology=passive_telemetry receipts must use receipt_type=protectmcp:observation, not :<offending> (rule 8)` message before the HTTP roundtrip (see `false_attestation_guard` in `python/src/asqav/client.py`).

```python
sig = agent.sign(
    "mcp:tool_call",
    {"server": "filesystem", "tool": "read"},
    receipt_type="protectmcp:observation",
    capture_topology="passive_telemetry",
    issuer_id="legal:Acme GmbH",
)
```

`capture_topology` is stamped on the audit-pack manifest entry but never on the signed payload. The other accepted topologies are `in_process_sdk`, `network_proxy`, `browser_extension`, `ebpf_observer`, and `mcp_proxy`; only `passive_telemetry` triggers the false-attestation guard. The full topology semantics live in the cloud's `docs/capture-topology.md`, and the wire vocabulary is published live at `https://api.asqav.com/.well-known/governance.json` for discovery.

### Configuration change receipts (rule 9)

A `receipt_type='protectmcp:lifecycle:configuration_change'` receipt declares the agent's runtime configuration was mutated. The SDK pre-flights the Asqav cloud's rule 9 cross-field gate (NSA CSI U/OO/6030316-26 alignment): the receipt MUST carry `config_manifest_digest`. Omitting it raises `ValueError` with the verbatim `false_attestation_guard: receipt_type=protectmcp:lifecycle:configuration_change requires config_manifest_digest (rule 9)` message before the HTTP roundtrip.

```python
sig = agent.sign(
    "mcp:config_update",
    {"server": "filesystem", "delta": "tool_added"},
    receipt_type="protectmcp:lifecycle:configuration_change",
    policy_decision="none",
    config_manifest_digest="sha256:<hex of manifest>",
    cve_inventory_digest="sha256:<hex of cve snapshot>",
)
```

### Expiry precedence (rule 10)

`valid_seconds` (legacy, server computes `valid_until = signed_at + valid_seconds`) and `expires_at` (caller-supplied horizon) are mutually exclusive. Pass exactly one of the two: `valid_seconds=3600` for "expire one hour after signing", or `expires_at="2026-06-01T00:00:00Z"` for an explicit horizon. Passing both raises `ValueError` with the verbatim `expiry_collision_guard: pass either valid_seconds or expires_at, not both (rule 10)` message before the HTTP roundtrip. Passing neither falls back to the server-side default (`valid_seconds=86400`).

### Digest format (rule 11)

Every caller-supplied digest field (`tool_fingerprint`, `config_manifest_digest`, `cve_inventory_digest`, `executable_hash`, `sbom_digest`) MUST match the regex `^sha256:[a-f0-9]{64}$`. The two URL pointer fields (`slsa_provenance_pointer`, `supply_chain_pointer`) MUST start with `http://` or `https://`. Anything else raises `ValueError` with a verbatim guard message before the HTTP roundtrip. To avoid wire drift, use the SDK's deterministic helpers (each is byte-deterministic under JCS):

```python
from asqav.client import (
    _compute_tool_fingerprint,
    _compute_config_manifest_digest,
    _compute_cve_inventory_digest,
)

fp = _compute_tool_fingerprint("search", {"args": {"q": "string"}})
cfg = _compute_config_manifest_digest({"server": "filesystem", "tools": ["read"]})
cve = _compute_cve_inventory_digest([{"id": "CVE-2026-0001", "severity": "high"}])
```

## NSA-aligned receipt fields

Six wire fields on `agent.sign(...)` carry the NSA CSI U/OO/6030316-26 alignment for MCP server lifecycle and tool output binding:

- `result_digest` - `sha256:<hex>` of the tool output, binds the receipt to a specific result. See <https://www.asqav.com/docs/result-digest>.
- `expires_at` - explicit ISO-8601 validity horizon. Mutually exclusive with `valid_seconds`. See <https://www.asqav.com/docs/expires-at>.
- `nonce` - 12 random bytes auto-generated when omitted; cloud rejects duplicates inside the validity window. See <https://www.asqav.com/docs/nonce>.
- `tool_fingerprint` - `sha256:<hex>` over `{tool_name, schema}`, auto-derived when `tool_name` + `tool_schema` are present. See <https://www.asqav.com/docs/tool-fingerprint>.
- `config_manifest_digest` - `sha256:<hex>` of the agent's runtime configuration snapshot. Required on configuration_change receipts. See <https://www.asqav.com/docs/config-manifest-digest>.
- `cve_inventory_digest` - `sha256:<hex>` over the CVE snapshot at sign time. See <https://www.asqav.com/docs/cve-inventory-digest>.

The `protectmcp:observation:result_bound` `receipt_type` variant carries `result_digest` and lets observation receipts bind to a specific tool result without claiming the policy gated the call.

### Build-provenance 4-tuple

Four optional wire fields bind build-side provenance into the signed receipt:

- `executable_hash` - `sha256:<hex>` of the executable that invoked the action.
- `sbom_digest` - `sha256:<hex>` of the canonical CycloneDX or SPDX SBOM document.
- `slsa_provenance_pointer` - https URL to the SLSA attestation envelope.
- `supply_chain_pointer` - https URL to the in-toto, Sigstore, or Rekor entry.

```python
sig = agent.sign(
    "build:provenance",
    {"image": "asqav/cloud:0.5.1"},
    compliance_mode=True,
    executable_hash="sha256:<hex of executable>",
    sbom_digest="sha256:<hex of SBOM>",
    slsa_provenance_pointer="https://attestations.example.com/slsa/build-1.intoto.jsonl",
    supply_chain_pointer="https://rekor.sigstore.dev/api/v1/log/entries/abc",
)
```

See <https://www.asqav.com/docs/executable-hash-and-sbom-provenance>.

### Audit Pack export

The cloud signs a Compliance Audit Pack over a window of receipts. The SDK wraps the endpoint:

```python
pack = asqav.fetch_audit_pack(start="2026-05-01T00:00Z", end="2026-06-01T00:00Z")
print(pack["bundle_digest"])              # sha256:<hex>
print(pack["bundle_signature"])           # base64 ML-DSA-65 sig over the bundle
print(pack["regime_mapping"])             # {regime_token: [record_id, ...]}
print(pack["algorithm_registry_version"]) # registry version pinned at issuance
```

`asqav.export_bundle(signatures, framework="dora")` is the offline alternative for air-gapped flows: it computes a Merkle root over an in-memory list of receipts without calling the cloud. Use `fetch_audit_pack` whenever the cloud is reachable, since only the cloud signature gives the auditor a tamper-evident manifest.

Local-side sanity checks (presence of REQUIRED fields, namespace, 300s skew bound, predecessor rederivation) are available as `asqav.verify_compliance_receipt(envelope, predecessor_envelope=...)`. The cloud is the authoritative verifier; this helper is a convenience.

Algorithm agility is exposed via `asqav.SUPPORTED_ALGORITHMS`. Pass `algorithm="ed25519"` or `"es256"` to `Agent.create(...)` for non-post-quantum identities, or `asqav.generate_local_keypair("ed25519")` for offline scenarios.

## Integrations

The SDK ships native callbacks and adapters for common agent frameworks under `asqav.extras.*`:

- **LangChain** - `from asqav.extras.langchain import AsqavCallbackHandler`, pass to a chain's `callbacks=[handler]`. Signs each chain, tool, and LLM lifecycle event.
- **CrewAI** - `from asqav.extras.crewai import AsqavCrewHook, AsqavGuardrailProvider`, attach the hook to the Crew so every task and agent step signs through Asqav.
- **LiteLLM** - `from asqav.extras.litellm import AsqavGuardrail`, register on `litellm.callbacks`. Signs every completion across providers.
- **OpenAI Agents SDK** - `from asqav.extras.openai_agents import AsqavGuardrail, AsqavTracingProcessor`, attaches to the runner and signs tool calls plus tracer spans.
- **LlamaIndex**, **Haystack**, **smolagents**, **DSPy**, **Letta**, **Strands**, **Instructor** - same pattern via `asqav.extras.<framework>`; each exposes one `AsqavAdapter` subclass.
- **pytest** - run `pytest --asqav --asqav-agent=test-runner` with `ASQAV_API_KEY` set; each test result signs and a Merkle-rooted compliance bundle emits on session finish.

Cookbooks for FastAPI middleware, AutoGen, and reasoning-trace capture live under `python/examples/`. See <https://asqav.com/docs/integrations> for the per-framework guide.

## Errors

All raised exceptions extend `AsqavError`. `AuthenticationError` (401), `RateLimitError` (429), and `APIError` (with `status_code`) are exported for fine-grained handling:

```python
from asqav import AsqavError, AuthenticationError, RateLimitError, APIError

try:
    agent.sign("api:call", {"model": "gpt-4"})
except AuthenticationError:
    # rotate ASQAV_API_KEY
    raise
except RateLimitError as exc:
    # exc.retry_after holds the cooldown in seconds
    raise
except APIError as exc:
    # exc.status_code holds the HTTP status
    raise
```

`ValueError` is raised before the HTTP roundtrip for the verbatim rule 8 / 9 / 10 / 11 guards listed above. Catch `ValueError` separately so guard violations surface as developer errors, not API errors.

## Requirements

Python 3.10 or newer. Uses `httpx` for the API client. Zero native dependencies; ML-DSA cryptography runs server-side.

## Standards

Asqav's compliance receipts are profiled in IETF Internet-Draft [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/), an Independent Submission that profiles [`draft-farley-acta-signed-receipts`](https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/) for EU AI Act Articles 12 and 26, and DORA Article 17 bindings. The SDK aligns with NIST FIPS 204 (ML-DSA), JCS canonicalization, and the NSA Cybersecurity Information Sheet U/OO/6030316-26 on MCP server lifecycle telemetry.

## Roadmap

What ships on Asqav today. Each item is available on `main`:

- Hash-only mode for cloud - default for `*.asqav.com`.
- Self-hosted signer (split-trust) - compose file in the Asqav backend repo.
- Bring-your-own KMS (AWS KMS / GCP KMS) - Enterprise tier.
- Customer-owned storage - self-hosted; relay payload allowlist enforced in code.
- SCITT / COSE_Sign1 receipt export - public `GET /api/v1/signatures/{id}/cose` returns `application/cose`.
- Air-gapped / on-prem mode - offline license + zero-egress; see the backend repo `docs/airgapped-mode.md`.

See <https://asqav.com/docs> for the live feature set.

## Documentation

- Repository: <https://github.com/jagmarques/asqav-sdk>
- Full docs: <https://asqav.com/docs>
- SDK guide: <https://asqav.com/docs/sdk>
- IETF profile: <https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/>
- Discovery descriptor: <https://asqav.com/.well-known/governance.json>

## License

Apache-2.0. Get an API key at [asqav.com](https://asqav.com).
