# asqav

[![GitHub stars](https://img.shields.io/github/stars/jagmarques/asqav-sdk?style=social)](https://github.com/jagmarques/asqav-sdk)

Python SDK for [asqav.com](https://asqav.com), the evidence layer for AI agents. Sign every agent action with ML-DSA-65 (FIPS 204), enforce policies before execution, and produce regulator-ready audit trails. All cryptography runs server-side, and the package has zero native dependencies.

## Install

```bash
pip install asqav
```

## Quick start

Get an API key at [asqav.com](https://asqav.com), then sign your first agent action:

```python
import asqav

asqav.init(api_key="sk_...")
agent = asqav.Agent.create("my-agent")

sig = agent.sign("api:openai:chat", {"model": "gpt-4o"})

print(sig.compliance_mode)        # True (default; pass compliance_mode=False to opt out)
print(sig.action_ref)             # "sha256:..." over the JCS-canonical action
print(sig.previous_receipt_hash)  # 64 hex; "0"*64 on the first record per agent
print(sig.verification_url)
```

That's it. One install, one init, one sign call. The receipt lands on the Asqav cloud under [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/): ML-DSA-65 (FIPS 204) signature, chain hash, retained `policy_digest`, fail-closed anchoring, and a public verification URL.

### No account? Offline path

If you need to record actions without a network call or an API key, use `local_sign`. Actions queue to disk and can be synced later:

```python
from asqav.local import local_sign, LocalQueue

item_id = local_sign("my-agent", "api:openai:chat", {"model": "gpt-4o"})
# Writes a pending JSON item to ~/.asqav/queue/

# When back online, init with your key then sync:
import asqav
asqav.init(api_key="sk_...")   # or set ASQAV_API_KEY in the environment
result = LocalQueue().sync()   # uploads queued items, returns {"synced": N, "failed": M}
```

Pass `compliance_mode=False` on any `agent.sign(...)` call if you want a non-Compliance receipt.

## CLI

The package ships an `asqav` CLI mirroring the Python API. Set `ASQAV_API_KEY` and run:

```bash
asqav verify <signature_id> [--output json]   # IETF axes when present
asqav sign --agent-id ID --action-type T --action-json action.json \
           --compliance-mode --receipt-type protectmcp:decision \
           --risk-class high --issuer-id legal:Acme
asqav agents list / create / revoke
asqav sessions list / end <session_id>
asqav replay <agent_id> <session_id>
asqav replay-verify <agent_id> <session_id> [--strict]     # IETF chain
asqav preflight <agent_id> <action_type>
asqav budget check / record
asqav approve <session_id> <entity_id>
asqav compliance frameworks / export
asqav audit-pack export --start ISO --end ISO --output-file bundle.json
asqav audit-pack policy <sha256:hex>
asqav payloads erase <signature_id>                        # P4 right-to-erasure
asqav org set-compliance-strict <org_id> --enable|--disable
asqav keys generate --algorithm ed25519|es256 [--out priv.pem]
asqav migrate run v3-20|v3-21|v3-22                        # X-Maintenance-Key required
asqav policies / webhooks list / create / delete
```

Every command listed here works on the free tier. The Asqav cloud is the source of truth for what your key may do.

## Data handling modes

The SDK auto-detects whether you're pointing at the Asqav cloud or a self-hosted deployment and selects the safer default for each:

- **Cloud (`*.asqav.com`)**: hash-only by default. The SDK builds a fingerprint of your action context, computes a SHA-256 hash locally, and sends only the hash plus a small metadata bag of `action_type`, `agent_id`, `session_id`, `model_name`, and `tool_name`. Raw prompts and tool arguments stay on your side.
- **Self-hosted**: full-payload by default. The server can run policy checks, PII redaction, and richer audit. Recommended when you control the deployment.

Override anytime:

```python
asqav.init(api_key="...", base_url="https://api.asqav.com", mode="hash-only")
```

The fingerprint format is sorted JSON with no whitespace per JCS, hashed with SHA-256. See `docs/fingerprint-spec.md` and `conformance/vectors.json` for the spec and cross-language test vectors.

## Advanced / high-value actions

Once you have the basics working, you can pass compliance envelope fields for regulated or high-risk actions. The full parameter set is useful for things like large financial transfers, healthcare decisions, or anything that needs an auditor-facing trail.

```python
sig = agent.sign(
    "payment.wire_transfer",
    {"amount_eur": 850000, "beneficiary_iban": "DE89370400440532013000"},
    receipt_type="protectmcp:decision",
    risk_class="high",
    issuer_id="legal:Acme GmbH",
    iteration_id="task-2026-Q2-4821",
)
```

The envelope extensions most callers reach for:

- `receipt_type` - `protectmcp:decision`, `protectmcp:restraint`, `protectmcp:lifecycle`, `protectmcp:lifecycle:configuration_change`, `protectmcp:acknowledgment`, `protectmcp:observation`, or `protectmcp:observation:result_bound`, which is an observation receipt that binds tool output via `result_digest`.
- `risk_class` - controlled vocabulary: `low | medium | high | unknown`.
- `iteration_id` - logical task id, distinct from session.
- `sandbox_state` - `enabled | disabled | unavailable` for high-risk gating.
- `incident_class` - DORA / NYDFS / CIRCIA token, or an array of tokens.
- `issuer_id` - LEI per ISO 17442, EIN, CIK, or a W3C DID for non-LEI deployers.

## Compliance receipts: the IETF profile

Compliance Receipts are the SDK default. Each `agent.sign(...)` call produces a receipt that conforms to [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/): ML-DSA-65 signature, JCS canonicalization, retained `policy_digest`, hash-chained `previous_receipt_hash`, OpenTimestamps anchoring. Opt out with `compliance_mode=False` if you want the older shape.

### Shadow AI capture with passive_telemetry

Two `receipt_type` values cover the gating axis: `protectmcp:decision` records that a policy ran and gated the action. `protectmcp:observation` records that a passive monitor saw the event without gating it. Pick `observation` when the producer never had the option to block, such as a SIEM forwarder, a browser extension in observe-only mode, or a NetFlow-style proxy with no enforcement hook.

Set `capture_topology='passive_telemetry'` to declare the producer is observing after the fact. The SDK client-side check pre-flights the Asqav cloud's full rule 8 gate: a `capture_topology='passive_telemetry'` receipt MUST use `receipt_type='protectmcp:observation'`. Any other receipt_type paired with `passive_telemetry`, namely `:decision`, `:restraint`, `:lifecycle`, `:lifecycle:configuration_change`, or `:acknowledgment`, raises `ValueError` with the verbatim `false_attestation_guard: capture_topology=passive_telemetry receipts must use receipt_type=protectmcp:observation[:result_bound], not :<offending> (rule 8)` message before the HTTP roundtrip. The guard lives as `false_attestation_guard` in `python/src/asqav/client.py`.

```python
sig = agent.sign(
    "mcp:tool_call",
    {"server": "filesystem", "tool": "read"},
    receipt_type="protectmcp:observation",
    capture_topology="passive_telemetry",
    issuer_id="legal:Acme GmbH",
)
```

`capture_topology` is stamped on the audit-pack manifest entry but never on the signed payload. The other accepted topologies are `in_process_sdk`, `network_proxy`, `browser_extension`, `ebpf_observer`, and `mcp_proxy`. Only `passive_telemetry` triggers the false-attestation guard. The full topology semantics live in the cloud's `docs/capture-topology.md`, and the wire vocabulary is published live at `https://api.asqav.com/.well-known/governance.json` for discovery.

### Configuration change receipts, rule 9

A `receipt_type='protectmcp:lifecycle:configuration_change'` receipt declares the agent's runtime configuration was mutated. The SDK pre-flights the Asqav cloud's rule 9 cross-field gate for NSA CSI U/OO/6030316-26 alignment: the receipt MUST carry `config_manifest_digest`. Omitting it raises `ValueError` with the verbatim `false_attestation_guard: receipt_type=protectmcp:lifecycle:configuration_change requires config_manifest_digest (rule 9)` message before the HTTP roundtrip.

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

### Expiry precedence, rule 10

The legacy `valid_seconds`, where the server computes `valid_until = signed_at + valid_seconds`, and the caller-supplied horizon `expires_at` are mutually exclusive. Pass exactly one of the two: `valid_seconds=3600` for "expire one hour after signing", or `expires_at="2026-06-01T00:00:00Z"` for an explicit horizon. Passing both raises `ValueError` with the verbatim `expiry_collision_guard: pass either valid_seconds or expires_at, not both (rule 10)` message before the HTTP roundtrip. Passing neither falls back to the server-side default of `valid_seconds=86400`.

### Digest format, rule 11

Every caller-supplied self-describing digest field, namely `config_manifest_digest`, `cve_inventory_digest`, `executable_hash`, and `sbom_digest`, MUST match the regex `^sha256:[a-f0-9]{64}$`. `tool_fingerprint` is the exception: it uses the cloud wire form of 32 bare lowercase hex chars, SHA-256[:32], matching `^[0-9a-f]{32}$` with no `sha256:` prefix. Anything else raises `ValueError` with the verbatim `tool_fingerprint_not_32_hex_chars: must be 32 lowercase hex chars (SHA-256[:32]).` guard. The two URL pointer fields `slsa_provenance_pointer` and `supply_chain_pointer` MUST start with `http://` or `https://`. Anything else raises `ValueError` with a verbatim guard message before the HTTP roundtrip. To avoid wire drift, use the SDK's deterministic helpers, each byte-deterministic under JCS:

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
- `expires_at` - explicit ISO-8601 or POSIX validity horizon. Converted client-side to `valid_seconds` and sent as a duration, since the cloud owns absolute time-binding. Mutually exclusive with `valid_seconds`. See <https://www.asqav.com/docs/expires-at>.
- `nonce` - 12 random bytes auto-generated when omitted. Cloud rejects duplicates inside the validity window. See <https://www.asqav.com/docs/nonce>.
- `tool_fingerprint` - 32 bare lowercase hex chars, SHA-256[:32], over `{tool_name, schema}`, auto-derived when `tool_name` + `tool_schema` are present. See <https://www.asqav.com/docs/tool-fingerprint>.
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

## Optional threat-framework mappings

Seven optional wire fields let a caller pin the receipt to industry threat-and-control taxonomies. Each list is caller-supplied and Asqav-preserved verbatim. The cloud sets `framework_mappings_self_declared=true` on the receipt whenever any of the six list fields is populated, so verifiers can tell self-declared classifications apart from cloud-verified ones.

- `mitre_techniques` - list of MITRE ATT&CK technique ids, for example `["T1059", "T1078"]`.
- `mitre_atlas` - list of MITRE ATLAS ids for AI-system threats, for example `["AML.T0051"]`.
- `owasp_llm_top10` - list of OWASP Top 10 for LLM ids, for example `["LLM01", "LLM02"]`.
- `nist_ai_rmf` - list of NIST AI RMF function ids, for example `["GOVERN-1.1", "MEASURE-2.7"]`.
- `iso_42001` - list of ISO/IEC 42001 control ids, for example `["A.6.2.6"]`.
- `eu_ai_act_articles` - list of EU AI Act article ids, for example `["Article-12", "Article-15"]`.
- `rfc3161_timestamp` - caller-supplied base64-encoded RFC 3161 TimeStampResp in DER.

```python
sig = agent.sign(
    "api:call",
    {"user": "..."},
    compliance_mode=True,
    mitre_techniques=["T1059", "T1078"],
    owasp_llm_top10=["LLM01"],
    nist_ai_rmf=["GOVERN-1.1", "MEASURE-2.7"],
    eu_ai_act_articles=["Article-12"],
)
```

Each list must be a non-empty list of strings, each entry up to 128 characters. Empty lists, non-string entries, or oversize entries raise `ValueError` with the verbatim guard messages `<field>_must_be_non_empty_list` or `<field>_entry_invalid` and skip the HTTP roundtrip. The `rfc3161_timestamp` must be valid base64 or raises `rfc3161_timestamp_not_base64`.

See <https://www.asqav.com/docs/threat-framework-mapping>.

## Optional witness policy

`witness_policy` lets a caller declare an N-of-M durable-anchoring quorum on the receipt. The receipt reaches `witness_quorum_met` only when `required` witnesses hold a real inclusion proof.

- Shape: `{"required": int, "witnesses": [<subset of "rfc3161", "opentimestamps">]}`.
- `required` must be in `[1, len(witnesses)]`.
- `witnesses` must be a non-empty subset of the two shipped witnesses: `rfc3161` and `opentimestamps`.
- `rekor` is rejected. It is not a shipped witness.
- `rfc3161` confirmation is available on the Enterprise tier only. On the Free tier, `opentimestamps` is the witness that produces an inclusion proof; a policy requiring only `rfc3161` will not reach `witness_quorum_met` on Free.

```python
sig = agent.sign(
    "deploy:release",
    {"build": "..."},
    compliance_mode=True,
    witness_policy={"required": 1, "witnesses": ["rfc3161", "opentimestamps"]},
)
```

Bad input raises `ValueError` with a verbatim guard message before the HTTP roundtrip: `witness_policy_unknown_witness` for an entry like `rekor`, `witness_policy_required_out_of_range`, `witness_policy_witnesses_must_be_non_empty_list`, `witness_policy_required_must_be_int`, or `witness_policy_duplicate_witness`. Omit the field for today's behaviour.

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

Local-side sanity checks, covering presence of REQUIRED fields, namespace, the 300s skew bound, and predecessor rederivation, are available as `asqav.verify_compliance_receipt(envelope, predecessor_envelope=...)`. The cloud is the authoritative verifier. This helper is a convenience.

`asqav.SUPPORTED_ALGORITHMS` is the set `asqav.generate_local_keypair(...)` can mint locally: `{ed25519, es256}`. It does not list the cloud agent-signing algorithms. `Agent.create(...)` sends its `algorithm` to the Asqav cloud, which accepts `ml-dsa-44`, `ml-dsa-65` (default), and `ml-dsa-87`. `ed25519` and `es256` are for local keypair generation only, so passing either to `Agent.create(...)` returns a 400 from the cloud.

## Offline / air-gapped verification

Receipts can be verified without calling the Asqav API once you have a JWKS snapshot.
Full guide: [`docs/offline-verification.md`](../docs/offline-verification.md).

Install the `verify` extra, which adds `dilithium-py` (ML-DSA-65) and `cryptography`
(Ed25519, ES256):

```bash
pip install "asqav[verify]"
```

Snapshot the JWKS while you have network access, then verify offline:

```python
import asqav, json

# online: snapshot the keys
jwks = asqav.fetch_jwks()
json.dump(jwks, open("jwks.json", "w"))

# offline: verify any receipt
receipt = json.load(open("receipt.json"))
jwks    = json.load(open("jwks.json"))
result  = asqav.verify_receipt_offline(receipt, jwks)
assert result["verdict"] == "PASS", result["axes"]
```

Supply a predecessor receipt to check the hash-chain link:

```python
result = asqav.verify_receipt_offline(receipt, jwks, predecessor=prev_receipt)
```

## Integrations

The SDK ships native callbacks and adapters for common agent frameworks under `asqav.extras.*`:

- **LangChain** - `from asqav.extras.langchain import AsqavCallbackHandler`, pass to a chain's `callbacks=[handler]`. Signs each chain, tool, and LLM lifecycle event.
- **CrewAI** - `from asqav.extras.crewai import AsqavCrewHook, AsqavGuardrailProvider`, attach the hook to the Crew so every task and agent step signs through Asqav.
- **LiteLLM** - `from asqav.extras.litellm import AsqavGuardrail`, register on `litellm.callbacks`. Signs every completion across providers.
- **OpenAI Agents SDK** - `from asqav.extras.openai_agents import AsqavGuardrail, AsqavTracingProcessor`, attaches to the runner and signs tool calls plus tracer spans.
- **LlamaIndex**, **Haystack**, **smolagents**, **DSPy**, **Letta**, **Strands**, **Instructor** - same pattern via `asqav.extras.<framework>`, each exposing one `AsqavAdapter` subclass.
- **pytest** - run `pytest --asqav --asqav-agent=test-runner` with `ASQAV_API_KEY` set. Each test result signs and a Merkle-rooted compliance bundle emits on session finish.

Cookbooks for FastAPI middleware, AutoGen, and reasoning-trace capture live under `python/examples/`. See <https://asqav.com/docs/integrations> for the per-framework guide.

## Errors

All raised exceptions extend `AsqavError`. `AuthenticationError` for 401, `RateLimitError` for 429, and `APIError` carrying `status_code` are exported for fine-grained handling:

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

## Structured receipts (optional schema)

Pass a `context_schema` to `Agent.sign()` to validate and normalise the
`context` dict before it is signed. This keeps every receipt in your audit
trail in a consistent, queryable shape.

```python
import asqav
from asqav import AsqavValidationError

PAYMENT_SCHEMA = {
    "user_id":  {"type": "string", "required": True},
    "amount":   {"type": "number", "required": True},
    "currency": {"type": "string", "required": True},
}

asqav.init(api_key="sk_...")
agent = asqav.Agent.create("my-agent")

# Valid context: keys are sorted before signing (consistent hash order).
sig = agent.sign(
    "payment:initiate",
    {"user_id": "u1", "currency": "EUR", "amount": 49.95},
    context_schema=PAYMENT_SCHEMA,
)

# Missing required field: raises AsqavValidationError BEFORE the network call.
try:
    agent.sign(
        "payment:initiate",
        {"amount": 100},          # user_id missing, currency missing
        context_schema=PAYMENT_SCHEMA,
    )
except AsqavValidationError as exc:
    print(exc)            # context_schema_error: field 'user_id' is required but missing
    print(exc.docs_url)   # https://asqav.com/docs/structured-receipts
```

Field descriptor keys:
- `type` (required) - one of `"string"`, `"number"`, `"boolean"`, `"object"`, `"array"`.
- `required` (optional, default `False`) - whether the field must be present.

`"number"` rejects `bool` values; `"array"` rejects plain dicts.
No schema means today's exact behaviour (zero behaviour change).

You can also pass a callable for full JSON Schema validation (no new
mandatory dependencies needed in the core package):

```python
def my_validator(ctx: dict) -> None:
    import jsonschema
    jsonschema.validate(ctx, MY_FULL_SCHEMA)

agent.sign("action", context, context_schema=my_validator)
```

See `examples/schema_receipts_example.py` for a runnable demo.

## Pluggable detectors (bring-your-own DLP/policy)

asqav owns the enforce + signed-proof spine; detection is pluggable. Register a
detector and it runs inside every `Agent.sign()`, after the context is normalised
and before the network call. Its verdict is recorded in the signed receipt (under
`_detectors`), so the audit trail proves which detector saw the action and why. A
deny raises `DetectorBlockedError` and no signature is created.

```python
import asqav
from asqav import DetectorBlockedError, DetectorResult, register_detector

class SsnDetector:
    name = "ssn-regex"
    def inspect(self, action_type: str, context: dict) -> DetectorResult:
        if "123-45-6789" in str(context):
            return DetectorResult(allow=False, labels=["US_SSN"], reason="SSN in context")
        return DetectorResult(allow=True)

asqav.init(api_key="sk_...")
agent = asqav.Agent.create("my-agent")
register_detector(SsnDetector())            # fail_open defaults to False

try:
    agent.sign("email:send", {"body": "ssn 123-45-6789"})
except DetectorBlockedError as exc:
    print(exc.detector_name, exc.result.labels)   # ssn-regex ['US_SSN']
```

- **Fail-closed by default**: if `inspect()` raises, sign blocks. Opt out per
  detector with `register_detector(d, fail_open=True)`.
- **Additive**: with no detector registered, the signed body is byte-identical
  to today (zero behaviour change).
- **Reference detectors** ship in `asqav.extras` behind optional extras:
  `PresidioDetector` (PII, `pip install asqav[presidio]`) and `OpaDetector`
  (Open Policy Agent policy-as-code, no extra deps). Any classifier or policy
  engine can feed the signed preflight decision this way.

See `examples/detector_plugin_example.py` for a runnable demo.

## Requirements

Python 3.10 or newer. Uses `httpx` for the API client. Zero native dependencies. ML-DSA cryptography runs server-side.

## Standards

Asqav's compliance receipts are profiled in IETF Internet-Draft [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/), an Independent Submission that profiles [`draft-farley-acta-signed-receipts`](https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/) for EU AI Act Articles 12 and 26, and DORA Article 17 bindings. The SDK aligns with NIST FIPS 204 (ML-DSA), JCS canonicalization, and the NSA Cybersecurity Information Sheet U/OO/6030316-26 on MCP server lifecycle telemetry.

## Roadmap

What ships on Asqav today. Each item is available on `main`:

- Hash-only mode for cloud - default for `*.asqav.com`.
- Self-hosted signer with split-trust - compose file in the Asqav backend repo.
- Bring-your-own KMS for AWS KMS or GCP KMS - Enterprise tier.
- Customer-owned storage - self-hosted, with the relay payload allowlist enforced in code.
- SCITT / COSE_Sign1 receipt export - public `GET /api/v1/signatures/{id}/cose` returns `application/cose`.
- Air-gapped / on-prem mode - offline license + zero-egress. See the backend repo `docs/airgapped-mode.md`.

See <https://asqav.com/docs> for the live feature set.

## Documentation

- Repository: <https://github.com/jagmarques/asqav-sdk>
- Full docs: <https://asqav.com/docs>
- SDK guide: <https://asqav.com/docs/sdk>
- IETF profile: <https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/>
- Discovery descriptor: <https://asqav.com/.well-known/governance.json>

## License

MIT. Get an API key at [asqav.com](https://asqav.com).
