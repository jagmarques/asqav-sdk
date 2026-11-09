# Agent.sign arguments

`Agent.sign` signs one action. The cloud creates the signature with ML-DSA and the call returns a
`SignatureResponse`. The SDK checks the arguments before the request and the cloud checks them again.

- `action_type`: Type of action (e.g., "read:data", "api:call"). Accepts semantic pattern names like "sql-read" which auto-expand to their glob equivalents.

- `context`: Additional context for the action.

- `context_schema`: Optional schema for the `context` dict. When supplied, the context is validated and then key-sorted before signing so audit trails are consistent and queryable. Two forms are accepted:
  Built-in descriptor (dependency-free):

  ```python
  {"user_id": {"type": "string", "required": True},
   "score":   {"type": "number"}}
  ```

  Field types: string | number | boolean | object | array. `required` defaults to False. Raises `AsqavValidationError` on mismatch BEFORE the network call.
  Custom callable (bring-your-own validator):

  ```python
  def my_validator(ctx: dict) -> None:
      import jsonschema
      jsonschema.validate(ctx, MY_FULL_SCHEMA)

  agent.sign("action", ctx, context_schema=my_validator)
  ```

  A callable receives the (merged, hook-dispatched) context and must raise on invalid input. Omit for today's behavior.

- `system_prompt_hash`: Optional hash of the system prompt for reproducibility tracking.

- `model_params`: Optional dict of model parameters (temperature, top_p, etc.) for reproducibility tracking.

- `tool_inputs_hash`: Optional hash of tool inputs for reproducibility tracking.

- `trace_id`: Optional trace ID for correlating actions across a multi-step workflow. Use generate_trace_id() to create one.

- `parent_id`: Optional parent action ID for linking child actions to their parent in a workflow. Powers the agent graph view.

- `counterparty`: Optional identity of the endpoint the agent is calling. Included in the signed record so an auditor can verify the agent reached a specific endpoint, not an imposter. Recommended shape:

  ```python
  {"kind": "tls_spki_sha256", "value": "<hex>"}
  {"kind": "pubkey_fingerprint", "value": "<hex>"}
  {"kind": "url_sni", "value": "api.example.com"}
  ```

  Any dict is accepted; unknown kinds are recorded verbatim.

- `tool_name`: Optional name of the tool being invoked when this action is a tool call. Surfaced in the dashboard graph. Only the name string travels; tool arguments do not.

- `model_name`: Optional name of the model being called when this action is an LLM call (e.g. "gpt-4", "claude-opus-4-7"). Only the name string travels; the prompt does not.

- `capture_topology`: Producer-side topology. One of `in_process_sdk`, `network_proxy`, `browser_extension`, `mcp_proxy`, `passive_telemetry`, `github_sha_pull`. Stamped on the audit-pack manifest entry; never on the signed payload. Client-supplied capture_topology is ADVISORY: the server stamps the authoritative value from the ingress route. `passive_telemetry` requires `receipt_type='protectmcp:observation'` (false-attestation guard).

- `result_digest`: NSA CSI U/OO/6030316-26 alignment. `sha256:<hex>` of the tool output. Caller-supplied and forwarded verbatim to the cloud. When omitted and `result` is given, the digest is computed from `result` and sent by default (criterion 787: every action gets an outcome).

- `result`: Optional tool result (bytes, str, or JSON-able value). Its canonical `sha256:<hex>` is computed and sent as `result_digest` unless `result_digest` is given explicitly. Sign-then-run callers that have no result yet omit both and attach the outcome later via `Agent.attach_result`.

- `expires_at`: Receipt validity horizon. ISO-8601 string or POSIX float. The cloud owns absolute time-binding and accepts only a duration, so the SDK converts this absolute horizon to a client-computed `valid_seconds` and sends that instead. Mutually exclusive with `valid_seconds` (rule 10). Pass exactly one of the two: `valid_seconds` for "expire N seconds after signing", or `expires_at` for an explicit horizon (converted client-side). Passing both raises `expiry_collision_guard`. Passing neither falls back to the server-side default (`valid_seconds=86400`).

- `tool_schema`: Optional JSON schema describing the tool surface. When provided alongside `tool_name` and `tool_fingerprint` is omitted, the SDK computes the fingerprint client-side via JCS canonicalization.

- `tool_fingerprint`: Explicit fingerprint over the canonical `{tool_name, schema}` blob. When omitted but `tool_name` + `tool_schema` are present, the SDK derives it client-side. MUST be 32 bare lowercase hex chars (SHA-256[:32], no `sha256:` prefix), matching the cloud wire form; use `_compute_tool_fingerprint` to avoid drift.

- `config_manifest_digest`: NSA CSI alignment. `sha256:<hex>` of the agent's runtime configuration manifest. REQUIRED for `receipt_type=protectmcp:lifecycle:configuration_change` (false-attestation rule 9). MUST match `sha256:<64-hex>` (rule 11); use `_compute_config_manifest_digest` to avoid drift.

- `cve_inventory_digest`: NSA CSI alignment. `sha256:<hex>` of the agent's CVE inventory snapshot at signing time. MUST match `sha256:<64-hex>` (rule 11); use `_compute_cve_inventory_digest` to avoid drift.

- `executable_hash`: Build-provenance 4-tuple. `sha256:<hex>` of the executable that invoked the action. Binds build-side provenance into the signed receipt. MUST match `sha256:<64-hex>` (rule 11).

- `sbom_digest`: Build-provenance 4-tuple. `sha256:<hex>` of the canonical CycloneDX or SPDX SBOM document. MUST match `sha256:<64-hex>` (rule 11).

- `slsa_provenance_pointer`: Build-provenance 4-tuple. https URL to the SLSA attestation envelope for the executing build.

- `supply_chain_pointer`: Build-provenance 4-tuple. https URL to the in-toto, Sigstore, or Rekor entry covering the executing build.

- `mitre_techniques`: Caller-supplied list of MITRE ATT&CK technique ids (e.g. `["T1059", "T1078"]`). Self-declared; never Asqav-verified. Auto-flips `framework_mappings_self_declared` to True on the receipt.

- `mitre_atlas`: Caller-supplied list of MITRE ATLAS ids for AI-system threats (e.g. `["AML.T0051", "AML.T0043"]`). Self-declared; never Asqav-verified.

- `owasp_llm_top10`: Caller-supplied list of OWASP Top 10 for LLM ids (e.g. `["LLM01", "LLM02"]`). Self-declared.

- `owasp_agentic_top10`: Caller-supplied OWASP Agentic Top 10 ids ASI01 through ASI10 without an edition suffix. Self-declared.

- `nist_ai_rmf`: Caller-supplied list of NIST AI RMF function ids (e.g. `["GOVERN-1.1", "MEASURE-2.7"]`). Self-declared.

- `iso_42001`: Caller-supplied list of ISO/IEC 42001 control ids (e.g. `["A.6.2.6"]`). Self-declared.

- `eu_ai_act_articles`: Caller-supplied list of EU AI Act article ids (e.g. `["Article-12", "Article-15", "Article-50"]`). Self-declared.

- `rfc3161_timestamp`: Caller-supplied base64-encoded RFC 3161 TimeStampResp (DER). Preserved on the receipt for offline TSA chain verification independent of cloud-issued anchors.

- `witness_policy`: Caller-declared N-of-M durable-anchoring quorum. Shape `{"required": int, "witnesses": [<subset of "rfc3161", "opentimestamps">]}`. `required` must be in `[1, len(witnesses)]`; `witnesses` must be a non-empty subset of the two shipped witnesses. `rekor` is rejected (not a shipped witness). The receipt reaches `witness_quorum_met` only when `required` witnesses hold a real inclusion proof. Optional; omit for today's behaviour.
- `nonce`: per-call token. The cloud answers 409 while a live receipt for this agent in this org holds
  it. When omitted the SDK generates one.

- `approver_id`, `initiator_id`, `acceptance_reason`, `accepted_at`, `supersedes`, `sarif_digest`,
  `finding_ref`, `approval_ref`, `risk_snapshot`: risk-acceptance (exception) receipt fields. Valid
  only on `receipt_type=protectmcp:lifecycle:risk_acceptance`.

- `invocation_ref`: producer-asserted invocation pointer, valid on any receipt type. It names one
  invocation only and promises nothing about uniqueness or exactly-once delivery.

- `repo_ref`, `commit_sha`, `base_sha`, `change_digest`, `change_ref`, `change_approval_ref`,
  `change_class`, `authored_by`: code-authorship receipt fields. Valid only on
  `receipt_type=protectmcp:lifecycle:code_authorship`.

- `counterparty_binding`: agent-to-agent acknowledgment. Binds this receipt to the peer receipt it
  acknowledges. Valid only on `receipt_type=protectmcp:acknowledgment`.
