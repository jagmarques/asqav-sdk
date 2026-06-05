# Changelog

All notable changes to Asqav (the SDK) will be documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [SemVer](https://semver.org/).

## [Unreleased]

## [TypeScript 0.5.8] - 2026-06-05

### Added
- A standalone neutral verifier under `typescript/src/verifier/` that verifies receipts across five formats (Asqav-native, AERF, ACTA, agent-receipts, Authproof) to parity with the Python oracle. Three canonicalisers (`jcs`, `asqavJcs`, `jcsRfc8785`) byte-match the Python ones, Ed25519 and ES256 verify through `node:crypto`, and ML-DSA-65 reports SKIPPED (Node has no primitive), so a post-quantum receipt is INCOMPLETE rather than a false PASS. A gate test reproduces the Python verdict on all conformance vectors, byte-checks the upstream canonicalization vectors, and verifies a receipt minted by the real Authproof SDK.
- A number-preserving JSON parser so float literals and integers beyond IEEE-754 safe range canonicalise to the exact bytes the producer signed; distinct large integers stay distinct.

## [Python 0.5.8] - 2026-06-05

### Added
- An Authproof adapter in the verifier oracle (the fifth format): ES256 over the insertion-order JSON of the receipt minus its signature, with the signer key read from the embedded P-256 JWK. New `verify_es256` in `verifier/oracle/crypto.py` wraps the `cryptography` library. A conformance vector minted by the real Authproof SDK gives a cross-implementation interop check.
- The neutral verifier now ships inside the `asqav` wheel: `from asqav.verifier.oracle import verify, ADAPTERS`. A real Asqav production hash-mode receipt is pinned as a conformance vector that reports INCOMPLETE without the optional `dilithium-py` dependency, proving the post-quantum path is never a false PASS.

### Fixed
- Malformed signature fields (non-string, non-hex, non-base64) now decode to a verification FAIL across the AERF and Asqav-native compliance paths instead of raising, restoring the never-crash contract.

## [TypeScript 0.5.7] - 2026-06-02

### Added
- Code-authorship receipt support on `agent.sign({...})`. New `receiptType` value `protectmcp:lifecycle:code_authorship` plus eight optional camelCase props projected to snake_case wire keys: `repoRef`, `commitSha`, `baseSha`, `changeDigest`, `changeRef`, `changeApprovalRef`, `changeClass`, and `authoredBy`. The `authoredBy` object names the producer-asserted author (`humanId`, `modelId`, `modelVersion`, `attestationSource`). Every field records that a producer-asserted value existed at signing time; Asqav does not re-run the diff, resolve the refs, or attest the model.
- Client-side validators in lockstep with the cloud SignRequest guards: `changeDigest` must be `sha256:<64 hex>`; `changeClass` must be one of `read|write|delete|execute|deploy`; a populated `authoredBy.modelId` / `modelVersion` requires `authoredBy.attestationSource`; the eight fields are fenced to `receiptType=protectmcp:lifecycle:code_authorship`; the receipt requires `repoRef` + `commitSha`, `policyDecision='none'`, and `complianceMode` (the fields project into the signed receipt only under compliance mode). Each rejection carries a `docsUrl` pointing at the code-authorship docs page.
- New `AuthoredBy` interface, `CODE_AUTHORSHIP_DOCS_URL` and `CODE_AUTHORSHIP_CHANGE_CLASS_NAMESPACE` constants. Tests in `typescript/tests/codeAuthorship.test.ts`.

### Notes
- Requires an Asqav cloud build that advertises `protectmcp:lifecycle:code_authorship` in `/.well-known/governance.json`. Mirrors the cloud guards landed in the cloud build for this receipt type.

## [Python 0.5.7] - 2026-06-02

### Added
- Code-authorship receipt support on `agent.sign(...)`. New `receipt_type` value `protectmcp:lifecycle:code_authorship` plus eight optional kwargs forwarded to the wire: `repo_ref`, `commit_sha`, `base_sha`, `change_digest`, `change_ref`, `change_approval_ref`, `change_class`, and `authored_by`. The `authored_by` dict names the producer-asserted author (`human_id`, `model_id`, `model_version`, `attestation_source`). Every field records that a producer-asserted value existed at signing time; Asqav does not re-run the diff, resolve the refs, or attest the model.
- Client-side validators in lockstep with the cloud SignRequest guards: `change_digest` must be `sha256:<64 hex>`; `change_class` must be one of `read|write|delete|execute|deploy`; a populated `authored_by.model_id` / `model_version` requires `authored_by.attestation_source`; the eight fields are fenced to `receipt_type=protectmcp:lifecycle:code_authorship`; the receipt requires `repo_ref` + `commit_sha`, `policy_decision='none'`, and `compliance_mode` (the fields project into the signed receipt only under compliance mode). Each rejection raises `AsqavValidationError` carrying a `docs_url` pointing at the code-authorship docs page.
- New `AuthoredBy` TypedDict and `CODE_AUTHORSHIP_DOCS_URL` constant, exported from `asqav`. Tests in `python/tests/test_client_code_authorship.py`.

### Fixed
- The zero-dependency verifier (`verifier/verify_receipt.py`) added `protectmcp:lifecycle:code_authorship` to its accepted receipt-type set; without it the structure axis rejected a valid code-authorship receipt on type and downgraded the whole verdict to FAIL. Test in `verifier/test_verify_receipt.py`.

### Notes
- Requires an Asqav cloud build that advertises `protectmcp:lifecycle:code_authorship` in `/.well-known/governance.json`. Mirrors the cloud guards landed in the cloud build for this receipt type.

## [TypeScript 0.5.6] - 2026-06-01

### Added
- Risk-acceptance / exception receipt support on `agent.sign({...})`. New `receiptType` value `protectmcp:lifecycle:risk_acceptance` plus nine optional camelCase props projected to snake_case wire keys: `approverId`, `initiatorId`, `acceptanceReason`, `acceptedAt`, `supersedes`, `sarifDigest`, `findingRef`, `approvalRef`, and `riskSnapshot`. The `riskSnapshot` object carries a producer-asserted point-in-time read of third-party risk signals (`snapshotAt`, `snapshotSource`, `epss`, `cvss`, `cvssVector`, `kevListed`, `cveIds`); numerics travel as strings. Every field records that a producer-asserted value existed at signing time; none is computed, scored, or verified by Asqav.
- Client-side validators in lockstep with the cloud SignRequest guards: `sarifDigest` must be `sha256:<64 hex>`; a populated `riskSnapshot` numeric requires `snapshotSource`; the nine fields are fenced to `receiptType=protectmcp:lifecycle:risk_acceptance`; the receipt requires `approverId` + `acceptanceReason`, `policyDecision='none'`, and `complianceMode` (the fields project into the signed receipt only under compliance mode). Each rejection carries a `docsUrl` pointing at the risk-acceptance docs page.
- New `RiskSnapshot` interface, `RISK_ACCEPTANCE_DOCS_URL` constant, and `AsqavError.docsUrl`. Tests in `typescript/tests/riskAcceptance.test.ts`.

### Notes
- Requires an Asqav cloud build that advertises `protectmcp:lifecycle:risk_acceptance` in `/.well-known/governance.json` (live as of 2026-06-01). Verified against the live build before this release.

## [Python 0.5.6] - 2026-06-01

### Added
- Risk-acceptance / exception receipt support on `agent.sign(...)`. New `receipt_type` value `protectmcp:lifecycle:risk_acceptance` plus nine optional kwargs forwarded to the wire: `approver_id`, `initiator_id`, `acceptance_reason`, `accepted_at`, `supersedes`, `sarif_digest`, `finding_ref`, `approval_ref`, and `risk_snapshot`. The `risk_snapshot` dict carries a producer-asserted point-in-time read of third-party risk signals (`snapshot_at`, `snapshot_source`, `epss`, `cvss`, `cvss_vector`, `kev_listed`, `cve_ids`); numerics travel as strings. Every field records that a producer-asserted value existed at signing time; none is computed, scored, or verified by Asqav.
- Client-side validators in lockstep with the cloud SignRequest guards: `sarif_digest` must be `sha256:<64 hex>`; a populated `risk_snapshot` numeric requires `snapshot_source`; the nine fields are fenced to `receipt_type=protectmcp:lifecycle:risk_acceptance`; the receipt requires `approver_id` + `acceptance_reason`, `policy_decision='none'`, and `compliance_mode` (the fields project into the signed receipt only under compliance mode). Each rejection raises `AsqavValidationError` carrying a `docs_url` pointing at the risk-acceptance docs page.
- New `AsqavValidationError` exception, `RiskSnapshot` TypedDict, and `RISK_ACCEPTANCE_DOCS_URL` constant, exported from `asqav`. Tests in `python/tests/test_client_risk_acceptance.py`.

### Fixed
- The zero-dependency verifier (`verifier/verify_receipt.py`) added `protectmcp:lifecycle:risk_acceptance` to its accepted receipt-type set; without it the structure axis rejected a valid risk-acceptance receipt on type and downgraded the whole verdict to FAIL. Test in `verifier/test_verify_receipt.py`.

### Notes
- Requires an Asqav cloud build that advertises `protectmcp:lifecycle:risk_acceptance` in `/.well-known/governance.json` (live as of 2026-06-01). Verified against the live build before this release.

## [TypeScript 0.5.5] - 2026-05-31

### Added
- `mandateId` on `agent.sign({...})`: one new optional camelCase prop projected to the snake_case `mandate_id` wire key. Names the resolvable `man_<token>` mandate the sign is authorised under. The cloud re-verifies the issuer signature, scope, server-clock window, and revocation, then builds the signed `authorized_under_mandate` attestation server-side. The attestation is never a request input. Requires Asqav cloud 0.5.2 or higher.

### Changed
- CLI version string in `src/cli.ts` bumped to `"0.5.5"` to match the package version.

### Notes
- `controls_evaluated` is a server-built block and never a request field; the TypeScript verify surface delegates full verification to the cloud `GET /verify/{id}` route, so the controls_evaluated structural recognition lands in the Python `verify_compliance_receipt` helper only. The TypeScript half stays at its established projection plus send-side parity level.

## [Python 0.5.5] - 2026-05-31

### Added
- `mandate_id` kwarg on `agent.sign(...)`: one new optional kwarg forwarded to the `mandate_id` wire key. Names the resolvable `man_<token>` mandate the sign is authorised under. The cloud re-verifies the issuer signature, scope, server-clock window, and revocation, then builds the signed `authorized_under_mandate` attestation server-side. The attestation is never a request input. Requires Asqav cloud 0.5.2 or higher.
- `verify_compliance_receipt` recognises `authorized_under_mandate` structurally and rejects a present-but-malformed attestation with the `false_mandate_attestation_guard` error code (lockstep with the cloud guard). A populated attestation must carry `mandate_id`, `issuer_id`, `verified=true`, and a `sha256:<64 hex>` `scope_digest`; an absent attestation is honest silence and never flagged.
- `verify_compliance_receipt` recognises `controls_evaluated` structurally and rejects a present-but-malformed block with the `false_control_attestation_guard` error code (lockstep with the cloud guard). The block must be an object of known control keys; `quorum` requires `fired=true` and a bare 64-hex `attestation_hash`; `policy.evaluated=true` requires `matched_count >= 1`. An unknown control key or a type mismatch is a forged attestation; an absent key is honest silence. `controls_evaluated` is server-built and never a sign input.
- New tests in `python/tests/test_verify_compliance_receipt.py` covering a valid mandate plus controls block, the verified!=true and bad scope_digest mandate rejections, and the controls_evaluated forged-key plus `policy.evaluated=true` with `matched_count=0` rejections.

## [TypeScript 0.5.4] - 2026-05-29

### Fixed
- `tool_fingerprint` now emits and validates the cloud wire form: 32 bare lowercase hex chars (SHA-256[:32], `^[0-9a-f]{32}$`), no `sha256:` prefix. Previously the SDK emitted and validated the self-describing `sha256:<64 hex>` form, which the cloud rejects, so every SDK-derived `tool_fingerprint` failed cloud validation. `computeToolFingerprint` returns the bare 32-hex string and the `toolFingerprint` validator throws the verbatim `tool_fingerprint_not_32_hex_chars: must be 32 lowercase hex chars (SHA-256[:32]).` guard. The other digest fields (`resultDigest`, `configManifestDigest`, `cveInventoryDigest`, `executableHash`, `sbomDigest`) keep the `sha256:<64 hex>` self-describing form unchanged.
- `expiresAt` is now converted client-side to `valid_seconds` and is no longer sent on the wire. The cloud SignRequest no longer carries an `expires_at` field and silently ignores extras, so sending `expires_at` would have been dropped without error and the default window applied. The SDK now parses the absolute horizon (ISO-8601 string or POSIX number), computes `valid_seconds = max(1, ceil(expiresAt - now))`, and sends only the duration. This is clock-safe: the cloud owns absolute time-binding and stamps the moment from its own clock. The rule-10 `expiry_collision_guard` (passing both `validSeconds` and `expiresAt`) still throws.

### Changed
- CLI version string in `src/cli.ts` bumped to `"0.5.4"` to match the package version.

## [Python 0.5.4] - 2026-05-29

### Fixed
- `tool_fingerprint` now emits and validates the cloud wire form: 32 bare lowercase hex chars (SHA-256[:32], `^[0-9a-f]{32}$`), no `sha256:` prefix. Previously the SDK emitted and validated the self-describing `sha256:<64 hex>` form, which the cloud rejects, so every SDK-derived `tool_fingerprint` failed cloud validation. `_compute_tool_fingerprint` returns the bare 32-hex string and the `tool_fingerprint` validator raises the verbatim `tool_fingerprint_not_32_hex_chars: must be 32 lowercase hex chars (SHA-256[:32]).` guard. The other digest fields (`result_digest`, `config_manifest_digest`, `cve_inventory_digest`, `executable_hash`, `sbom_digest`) keep the `sha256:<64 hex>` self-describing form unchanged.
- `expires_at` is now converted client-side to `valid_seconds` and is no longer sent on the wire. The cloud SignRequest no longer carries an `expires_at` field and silently ignores extras, so sending `expires_at` would have been dropped without error and the default window applied. The SDK now parses the absolute horizon (ISO-8601 string or POSIX float) to a UTC datetime, computes `valid_seconds = max(1, ceil(expires_at - now))`, and sends only the duration. This is clock-safe: the cloud owns absolute time-binding and stamps the moment from its own clock. The rule-10 `expiry_collision_guard` (passing both `valid_seconds` and `expires_at`) still raises.

## [TypeScript 0.5.3] - 2026-05-29

### Added
- `witnessPolicy` on `agent.sign({...})`: one new optional camelCase prop mirroring the cloud `witness_policy` extension. Shape `{ required: number, witnesses: Array<"rfc3161" | "opentimestamps"> }`, projected to the snake_case `witness_policy` wire key via `IETF_OPTIONAL_FIELD_MAP`. Declares an N-of-M durable-anchoring quorum; the receipt reaches `witness_quorum_met` only when `required` witnesses hold a real inclusion proof. OPTIONAL; absent keeps today's behaviour. Requires Asqav cloud 0.5.3 or higher.
- Client-side validators emit verbatim guard tokens that round-trip through the conformance vectors: `witness_policy_unknown_witness` rejects any witness outside the two shipped witnesses (`rekor` is rejected because it is not shipped), `witness_policy_required_out_of_range` rejects `required` outside `[1, witnesses.length]`, `witness_policy_witnesses_must_be_non_empty_list` rejects an empty list, `witness_policy_required_must_be_int` rejects a non-integer `required`, and `witness_policy_duplicate_witness` rejects duplicates.
- New tests in `typescript/tests/witnessPolicy.test.ts` covering wire projection, the rekor rejection, the `required` range, and the other shape guards.
- Exported `WITNESS_NAMESPACE`, the `Witness` type, and the `WitnessPolicy` interface.

### Changed
- CLI version string in `src/cli.ts` bumped to `"0.5.3"` to match the package version.

## [Python 0.5.3] - 2026-05-29

### Added
- `witness_policy` kwarg on `agent.sign(...)`: one new optional kwarg mirroring the cloud `witness_policy` extension. Shape `{"required": int, "witnesses": [<subset of "rfc3161", "opentimestamps">]}`, forwarded verbatim to the wire. Declares an N-of-M durable-anchoring quorum; the receipt reaches `witness_quorum_met` only when `required` witnesses hold a real inclusion proof. OPTIONAL; absent keeps today's behaviour. Requires Asqav cloud 0.5.3 or higher.
- Client-side validators emit verbatim guard tokens: `witness_policy_unknown_witness` (`rekor` is rejected because it is not shipped), `witness_policy_required_out_of_range`, `witness_policy_witnesses_must_be_non_empty_list`, `witness_policy_required_must_be_int`, and `witness_policy_duplicate_witness`.
- New tests in `python/tests/test_client_witness_policy.py` covering wire forwarding, the rekor rejection, the `required` range, and the other shape guards.
- Exported `WITNESS_NAMESPACE` from `asqav.client`.

## [TypeScript 0.5.2] - 2026-05-26

### Added
- Threat-framework taxonomy mappings on `agent.sign({...})`: seven new optional camelCase props mirroring the cloud SignRequest cross-field validator. `mitreTechniques` (wire `mitre_techniques`, MITRE ATT&CK technique ids), `mitreAtlas` (`mitre_atlas`, MITRE ATLAS ids for AI-system threats), `owaspLlmTop10` (`owasp_llm_top10`, OWASP Top 10 for LLM ids), `nistAiRmf` (`nist_ai_rmf`, NIST AI RMF function ids and subcategories), `iso42001` (`iso_42001`, ISO/IEC 42001 control ids), `euAiActArticles` (`eu_ai_act_articles`, EU AI Act article ids), and `rfc3161Timestamp` (`rfc3161_timestamp`, base64-encoded TimeStampResp DER). All seven OPTIONAL. The cloud auto-flips `framework_mappings_self_declared=true` whenever any of the six list fields is populated, so verifiers can tell self-declared classifications apart from cloud-verified ones. Requires Asqav cloud 0.5.2 or higher.
- Client-side validators emit verbatim guard tokens that round-trip through the conformance vectors: `<field>_must_be_non_empty_list` rejects an empty array; `<field>_entry_invalid` rejects non-string entries or strings >128 chars; `rfc3161_timestamp_not_base64` rejects malformed base64.
- New tests in `typescript/tests/threatFrameworkMappings.test.ts` covering wire forwarding, validation rejection, omission, and the all-seven-together happy path.

### Changed
- CLI version string in `src/cli.ts` bumped to `"0.5.2"` to match the package version.

## [Python 0.5.2] - 2026-05-26

### Added
- Threat-framework taxonomy mappings on `agent.sign(...)`: seven new optional kwargs mirroring the cloud SignRequest cross-field validator. `mitre_techniques` (MITRE ATT&CK technique ids), `mitre_atlas` (MITRE ATLAS ids for AI-system threats), `owasp_llm_top10` (OWASP Top 10 for LLM ids), `nist_ai_rmf` (NIST AI RMF function ids and subcategories), `iso_42001` (ISO/IEC 42001 control ids), `eu_ai_act_articles` (EU AI Act article ids), and `rfc3161_timestamp` (base64-encoded TimeStampResp DER). All seven are OPTIONAL; the cloud auto-flips `framework_mappings_self_declared=true` whenever any of the six list fields is populated. Requires Asqav cloud 0.5.2 or higher.
- Client-side validators emit verbatim guard tokens that round-trip through the conformance vectors: `<field>_must_be_non_empty_list`, `<field>_entry_invalid`, `rfc3161_timestamp_not_base64`.
- New parametric tests over the seven fields plus validation rejection (`python/tests/test_client_threat_framework_mappings.py`).

## [TypeScript 0.5.1] - 2026-05-25

### Added
- Build-provenance 4-tuple on `agent.sign({...})`: four new optional wire fields mirroring the cloud SignRequest. `executableHash` (wire `executable_hash`, `sha256:<hex>` of the executable that invoked the action), `sbomDigest` (wire `sbom_digest`, `sha256:<hex>` of the CycloneDX or SPDX SBOM), `slsaProvenancePointer` (wire `slsa_provenance_pointer`, https URL to the SLSA attestation envelope), and `supplyChainPointer` (wire `supply_chain_pointer`, https URL to the in-toto, Sigstore, or Rekor entry). All four OPTIONAL. Requires Asqav cloud 0.5.1 or higher.
- `digest_format_guard` extended to cover `executableHash` and `sbomDigest`; SDK throws `AsqavError` with the verbatim guard message before the HTTP roundtrip when either is not `^sha256:[a-f0-9]{64}$`.
- `pointer_url_guard` (new) covers `slsaProvenancePointer` and `supplyChainPointer`; SDK throws when either is not an http(s) URL.
- New tests in `typescript/tests/executableHash4Tuple.test.ts` covering wire forwarding plus shape rejection.

### Changed
- CLI version string in `src/cli.ts` bumped to `"0.5.1"` to match the package version.

## [Python 0.5.1] - 2026-05-25

### Added
- Build-provenance 4-tuple on `agent.sign(...)`: four new optional wire fields mirroring the cloud SignRequest. `executable_hash` (`sha256:<hex>` of the executable that invoked the action), `sbom_digest` (`sha256:<hex>` of the canonical CycloneDX or SPDX SBOM), `slsa_provenance_pointer` (https URL to the SLSA attestation envelope), and `supply_chain_pointer` (https URL to the in-toto, Sigstore, or Rekor entry). All four are OPTIONAL; the cloud accepts each independently. Subsumes third-party signed-build-provenance vendors by binding the same evidence into the signed receipt.
- `digest_format_guard` extended to cover `executable_hash` and `sbom_digest`; SDK raises `ValueError` with the verbatim guard message before the HTTP roundtrip when either field is not `^sha256:[a-f0-9]{64}$`.
- `pointer_url_guard` (new) covers `slsa_provenance_pointer` and `supply_chain_pointer`; SDK raises `ValueError` when either is not an http(s) URL.
- New parametric tests over the four fields plus shape rejection (`python/tests/test_client_executable_hash_4tuple.py`). Requires Asqav cloud 0.5.1 or higher.

## [TypeScript 0.5.0] - 2026-05-25

### Changed
- Version bump from 0.4.0 to 0.5.0 to align the npm @asqav/sdk version with the Python `asqav` package and the Asqav cloud 0.5.0 release. No wire-shape changes versus TypeScript 0.4.0; all NSA CSI U/OO/6030316-26 alignment features (five new optional props on `agent.sign({...})`, rule 9 / rule 10 / rule 11 cross-field guards, `protectmcp:observation:result_bound` receipt type, and the public digest helpers) shipped in 0.4.0 and are unchanged. CLI version string in `src/cli.ts` bumped from `"0.4.0"` to `"0.5.0"` to match the package version. Requires Asqav cloud 0.5.0 or higher.

## [Python 0.5.0] - 2026-05-25

### Added
- NSA CSI U/OO/6030316-26 alignment: five new optional wire fields on `agent.sign(...)` mirroring the cloud 0.5.0 SignRequest. `result_digest` and `expires_at` are caller-supplied passthrough; `tool_fingerprint` is auto-derived from `tool_name` + `tool_schema` (deterministic JCS `sha256:<hex>`) when omitted; `config_manifest_digest` and `cve_inventory_digest` are caller-supplied. `nonce` (24-hex-char) is auto-generated by the SDK when the caller omits one for replay protection.
- Cross-field guard mirroring cloud rule 9: a `receipt_type='protectmcp:lifecycle:configuration_change'` receipt MUST carry `config_manifest_digest`, otherwise the SDK raises `ValueError` with the verbatim `configuration_change_missing_config_manifest_digest: receipt_type=protectmcp:lifecycle:configuration_change requires config_manifest_digest (sha256:<64 hex>).` message before the HTTP roundtrip.
- Cross-field guard mirroring cloud rule 10 (`expiry_collision_guard`): `valid_seconds` and `expires_at` are mutually exclusive. Passing both raises `ValueError` verbatim before the HTTP roundtrip. Passing neither uses the server-side default (`valid_seconds=86400`). Precedence documented in `client.py` docstrings and both READMEs.
- Cross-field guard mirroring cloud rule 11 (`digest_format_guard`): `tool_fingerprint`, `config_manifest_digest`, `cve_inventory_digest` MUST match `^sha256:[a-f0-9]{64}$`. Anything else raises `ValueError` verbatim before the HTTP roundtrip.
- New receipt_type variant `protectmcp:observation:result_bound` for observation receipts that bind tool output via `result_digest`. Added to `RECEIPT_TYPE_NAMESPACE` and round-tripped by the parametrised tests.
- Internal helpers `_compute_tool_fingerprint(tool_name, tool_schema)`, `_compute_config_manifest_digest(manifest)`, `_compute_cve_inventory_digest(cve_list)` and `_generate_nonce()` for callers that want to precompute either field. All digest helpers are byte-deterministic under JCS (RFC 8785) and produce output that matches the rule-11 regex.
- Parametrised tests over the five new fields, the rule-9 / rule-10 / rule-11 happy + sad paths, and full namespace round-trip including `protectmcp:observation:result_bound` (`python/tests/test_client_nsa_aligned_fields.py`).

### Changed
- Minor version bump: five new wire fields = additive surface change, no breaking shape. Requires Asqav cloud 0.5.0 or higher.

## [TypeScript 0.4.0] - 2026-05-25

### Added
- NSA CSI U/OO/6030316-26 alignment: five new optional props on `agent.sign({...})` mirroring the cloud 0.5.0 SignRequest. `resultDigest` and `expiresAt` are caller-supplied passthrough; `toolFingerprint` is auto-derived from `toolName` + `toolSchema` (deterministic JCS `sha256:<hex>`) when omitted; `configManifestDigest` and `cveInventoryDigest` are caller-supplied. `nonce` (24-hex-char) is auto-generated by the SDK when the caller omits one for replay protection.
- Cross-field guard mirroring cloud rule 9: a `receiptType: "protectmcp:lifecycle:configuration_change"` receipt MUST carry `configManifestDigest`, otherwise the SDK throws `AsqavError` with the verbatim `configuration_change_missing_config_manifest_digest: receipt_type=protectmcp:lifecycle:configuration_change requires config_manifest_digest (sha256:<64 hex>).` message before the HTTP roundtrip.
- Cross-field guard mirroring cloud rule 10 (`expiry_collision_guard`): `validSeconds` and `expiresAt` are mutually exclusive. Passing both throws `AsqavError` verbatim before the HTTP roundtrip. Passing neither uses the server-side default (`valid_seconds=86400`). Precedence documented in JSDoc and both READMEs.
- Cross-field guard mirroring cloud rule 11 (`digest_format_guard`): `toolFingerprint`, `configManifestDigest`, `cveInventoryDigest` MUST match `^sha256:[a-f0-9]{64}$`. Anything else throws `AsqavError` verbatim before the HTTP roundtrip.
- New receiptType variant `protectmcp:observation:result_bound` for observation receipts that bind tool output via `resultDigest`. Added to `RECEIPT_TYPE_NAMESPACE` and round-tripped by the parametrised tests.
- Public helpers `computeToolFingerprint(toolName, toolSchema)`, `computeConfigManifestDigest(manifest)`, `computeCveInventoryDigest(cveList)` and `generateNonce()` re-exported from `@asqav/sdk` for callers that want to precompute either field. All digest helpers are byte-deterministic under JCS (RFC 8785) and produce output that matches the rule-11 regex.
- New test file `typescript/tests/nsaAlignedFields.test.ts` mirroring the Python parametrised cases for rule-9 / rule-10 / rule-11 and full namespace round-trip including `protectmcp:observation:result_bound`.

### Changed
- Minor version bump: five new wire fields = additive surface change, no breaking shape. Requires Asqav cloud 0.5.0 or higher.

## [Python 0.4.11] - 2026-05-25

### Changed
- README rule-8 paragraph rewritten to describe the full `false_attestation_guard` scope (rejects passive_telemetry paired with any non-observation receipt_type). The 0.4.10 wheel METADATA documented only the `:decision` case which under-described the runtime check.

## [TypeScript 0.3.8] - 2026-05-25

### Changed
- `false_attestation_guard` widened to mirror the Asqav cloud's full guard: `captureTopology=passive_telemetry` now requires `receiptType=protectmcp:observation`. The SDK client-side check rejects all five other receipt types (`:decision`, `:restraint`, `:lifecycle`, `:lifecycle:configuration_change`, `:acknowledgment`) so callers fail fast instead of receiving HTTP 422 from the cloud. The verbatim error message preserves the `false_attestation_guard:` prefix and the `(rule 8)` suffix; the offending value is interpolated. Requires Asqav cloud 0.4.0 or higher.

## [Python 0.4.10] - 2026-05-25

### Changed
- `false_attestation_guard` widened to mirror the Asqav cloud's full guard: `capture_topology=passive_telemetry` now requires `receipt_type=protectmcp:observation`. The SDK client-side check rejects all five other receipt types (`:decision`, `:restraint`, `:lifecycle`, `:lifecycle:configuration_change`, `:acknowledgment`) so callers fail fast instead of receiving HTTP 422 from the cloud. The verbatim error message preserves the `false_attestation_guard:` prefix and the `(rule 8)` suffix; the offending value is interpolated. Requires Asqav cloud 0.4.0 or higher.

## [Python 0.4.9] - 2026-05-25

### Changed
- Wire-format break: SDK now reads the provider-agnostic anchor field names from cloud responses. `bitcoin.bitcoin_tx` is now `bitcoin.anchor_tx_ref` and `bitcoin.bitcoin_block` is now `bitcoin.anchor_block_height` on `/api/v1/sessions/{id}/signatures` and `/api/v1/agents/{id}/sign` response bodies. Requires Asqav cloud 0.3.8 or higher.
- Public dataclass fields `BitcoinAnchor.bitcoin_tx` and `BitcoinAnchor.bitcoin_block` (and `BitcoinAnchorStatus.bitcoin_tx` / `bitcoin_block`) are renamed to `anchor_tx_ref` and `anchor_block_height` to match the new wire vocabulary. Consumers reading these attributes MUST update.

### Added
- `protectmcp:lifecycle:configuration_change` is now part of `RECEIPT_TYPE_NAMESPACE`. The Asqav cloud accepts this token to record lifecycle conflict receipts; the SDK was rejecting it client-side with `invalid_receipt_type`. The vocabulary on `python/src/asqav/client.py` now mirrors the cloud's pydantic validator and the parametrised test covers the new value.

## [TypeScript 0.3.7] - 2026-05-25

### Changed
- Wire-format break: SDK now reads the provider-agnostic anchor field names from cloud responses. `bitcoin.bitcoin_tx` is now `bitcoin.anchor_tx_ref` and `bitcoin.bitcoin_block` is now `bitcoin.anchor_block_height` on the verify response. Requires Asqav cloud 0.3.8 or higher.
- Public interface `BitcoinAnchorStatus.bitcoinTx` and `bitcoinBlock` are renamed to `anchorTxRef` and `anchorBlockHeight` to match the new wire vocabulary. Consumers reading these properties MUST update.

### Added
- `protectmcp:lifecycle:configuration_change` is now part of `RECEIPT_TYPE_NAMESPACE` so the SDK forwards the value through `agent.sign(...)` instead of throwing `invalid_receipt_type` client-side.

## [Python 0.4.8] - 2026-05-24

### Fixed
- Adds `templates/shadow-ai/.env.template` to the packaged wheel and sdist via a single-file `force-include` scoped to that path. Without it, `asqav shadow-ai init` raised `FileNotFoundError` mid-scaffold and left the target directory half-written.

## [Python 0.4.7] - 2026-05-24

### Fixed
- Package the `templates/shadow-ai/` docker-compose template set under `asqav/templates/shadow-ai/` so `asqav shadow-ai init` resolves its assets from the installed wheel.

## [TypeScript 0.3.6] - 2026-05-24

### Added
- `protectmcp:observation` receipt_type and `passive_telemetry` capture_topology in the wire vocabulary.
- Client-side false-attestation guard mirroring the cloud rule: rejects `capture_topology=passive_telemetry` paired with `receipt_type=protectmcp:decision`, with verbatim message `false_attestation_guard: capture_topology=passive_telemetry receipts must use receipt_type=protectmcp:observation, not :decision (rule 8)`.
- 288 jest tests across 25 files pass against the new vocab.

## [Python 0.4.6] - 2026-05-24

### Added
- `protectmcp:observation` receipt_type and `passive_telemetry` capture_topology in the SDK vocabularies.
- Client-side false-attestation guard mirroring the cloud rule: rejects `capture_topology=passive_telemetry` paired with `receipt_type=protectmcp:decision`, with verbatim message `false_attestation_guard: capture_topology=passive_telemetry receipts must use receipt_type=protectmcp:observation, not :decision (rule 8)`.

## [Python 0.4.5 / TypeScript 0.3.5] - 2026-05-18

### Added (TypeScript)
- Three framework adapters under `@asqav/sdk/extras`: `AsqavCallbackHandler` (LangChain.js, `@langchain/core`), `AsqavMastraHook` (Mastra, `@mastra/core`), `AsqavOpenAIAgentsAdapter` (OpenAI Agents JS, `@openai/agents`). Each is lazy-imported through an optional peer dependency and falls back with a clear ImportError-equivalent message when the peer is not installed.
- `raiseMissingPeer(framework, peer, install, cause?)` is now exported from `@asqav/sdk/extras` so downstream adapter authors share the same Python-parity ImportError contract.
- `python/docs/self-hosted-signer.md`: split-trust deployment guide showing how to point the SDK at a customer-operated signer container via `ASQAV_API_URL` plus the runnable `docker-compose.signer.yml`.

### Added (Python + TypeScript)
- `canonicalize_tool_args` (Python) and `canonicalizeToolArgs` (TypeScript) pre-validate `tool_args` payloads and reject float leaves with a JSON-pointer-locating error before delegating to JCS canonicalization, surfacing the rule that floats are not byte-stable across runtimes.
- Counterparty acknowledgment helpers (`compute_counterparty_binding`, `verify_counterparty_binding`, `CounterpartyBinding`, `ACKNOWLEDGMENT_RECEIPT_TYPE`) in both Python and TypeScript. Mirrors the cloud's `counterparty_binding` extension field; computes base64 SHA-256 of the originating envelope's canonical bytes and recompares at verify time. `protectmcp:acknowledgment` joins the receipt-type namespace. `verify_compliance_receipt` accepts an optional `originating_envelope` argument and surfaces `counterparty_binding_verified`.
- `VerificationResponse` exposes `type`, `bitcoinAnchor`, `signatureEnvelope`, `anchors`, `algorithmRegistryVersion` (Python: snake_case). Match the cloud `/verify` response per IETF -03 §8.1.
- `VerificationDetail` exposes `anchorValidOts`, `anchorValidRfc3161`, `policyDigestResolved`, `duplicateEmissionCandidate`.
- TypeScript validation-label union includes `agent_revoked_before_issuance`.
- `VerificationDetail.regimesSatisfied` (Python: `regimes_satisfied`): regulator tokens the cloud derived for the receipt (e.g. `eu_ai_act`, `dora`).

### Removed
- `VerificationResponse.verifierSignature` (Python: `verifier_signature`). The cloud omits this field on emit; it projects as `null` end-to-end, so the typed surface is dropped rather than left in as dead weight.
- Inner anchor `type` rejects the `"ots"` alias. The cloud emits only `"opentimestamps"` (canonical) and `"rfc3161"`; the SDK type stubs are tightened accordingly.

### Changed
- `generate_local_keypair` rejects `"ml-dsa-65"`. ML-DSA-65 keypair generation is server-side only (cloud KMS); call `asqav.Agent.create(name, algorithm='ml-dsa-65')` to mint one. `SUPPORTED_ALGORITHMS` is `{ed25519, es256}` and the default for `generate_local_keypair()` is `ed25519`. ML-DSA-65 raises a `ValueError` at validation time, with the typed `Algorithm` Literal narrowed to reflect what the SDK mints locally.
- `raiseMissingPeer` (TypeScript) signature is now `(framework, peer, install, cause?)` rather than `(peer, install)`. The optional `cause` preserves the underlying `(import error: ...)` context the inline pattern emitted. The three first-party adapters all call through it, matching the Python ImportError contract.

## [Python 0.4.4 / TypeScript 0.3.4] - 2026-05-10

### Changed
- **`complianceMode` defaults to `True`**. New SDK callers get a Compliance Receipt out of the box; pass `complianceMode: false` to opt out. Both Python and TypeScript.
- **`FRAMEWORKS` list** replaced with the 10 regime tokens the cloud actually emits: `eu_ai_act`, `dora`, `nydfs_500`, `colorado_ai`, `texas_traiga`, `nist_ai_rmf`, `circia`, `hipaa_security`, `sec_17a4a`, `sec_17a4b`. Drops the four placeholder tokens (`eu_ai_act_art12`, `eu_ai_act_art14`, `dora_ict`, `soc2`).

### Added
- `fetch_audit_pack()` (Python): wraps the cloud `/api/v1/audit-pack/export` endpoint and returns the cloud-signed bundle (`bundle_digest`, `bundle_signature`, `regime_mapping`, `revocation_manifest`). Use this to pull a regulator-ready Audit Pack; the offline `export_bundle` aggregator stays for self-hosted-signer use. (TypeScript helper lands in a follow-up release.)

### Internal
- Comment-hygiene + AI-navigation gold standard sweep across both SDK source trees (47 multi-line comment blocks collapsed; 14 forbidden framing tokens reworded; SignatureResponse / ReplayStep / ReplayTimeline class docstrings expanded).
- Stripped draft-version refs (`-00`) and spec-section citations (`§5`, `§5.7`) from SDK source + docs. Public-facing references now use the version-less Datatracker URL `https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/`.

## [Python 0.4.3 / TypeScript 0.3.3] - 2026-05-08

### Changed
- Shared `canonicalize_action(action_type, context)` / `canonicalizeAction(actionType, context)` helper. Both SDKs route `hash_action` / `hashAction`, `_compute_action_ref`, and the hash-only sign body through the same primitive. No wire-format change.

## [Python 0.4.2 / TypeScript 0.3.2] - 2026-05-08

### Changed
- **`payload_digest` size on hash-mode signs**: hash-only mode now forwards `payload_size` (byte length of the canonical fingerprint) on every request. The cloud builds the wire `payload_digest` as `{hash, size}` instead of `{hash}` alone. Both SDKs compute the size from the same canonical bytes used for the hash, so callers see no API change. Previous releases relied on the cloud filling size from server-side context, which left hash-only compliance signs without the `size` axis required by the spec.

## [Python 0.4.0 / TypeScript 0.3.0] - 2026-05-08

### Changed
- **Wire shape**: under `compliance_mode`, the cloud emits the three-key Compliance Receipts envelope `{payload, signature, anchors}` plus pure metadata. The SDKs expose `payload`, `signature` (polymorphic: string in non-compliance, `{alg, kid, sig}` object form under compliance_mode), and `anchors` directly on `SignatureResponse`.
- **Removed `signatureObject`**: the polymorphic `signature` field carries the same value. Use `response.signature_envelope()` (Python) or `signatureEnvelope(response)` (TypeScript) for the dict form.
- **Removed flat-field projection fallbacks**: the projection helpers skip rebuilding `{alg, kid, sig}` from `algorithm` + `signature_b64`. Older receipts return None / undefined from the helpers.
- **Anchor entries carry `commit_hash`** sourced from the cloud's `anchor_commit_hash` column. Field is `None` on rows that lack the column.

## [Python 0.3.13 / TypeScript 0.2.11] - 2026-05-08

### Changed
- Both SDK readers accept the new `signature` object form (cloud 0.2.14+) directly, falling back to `signatureObject` for older clouds. Polymorphic `signature: string | object` typed across the public surface. `kid` is read from `kid`, `issuer_id`, or `key_id` in that order.
- `incident_class` widened to `string | string[] | undefined`. Validator runs vocabulary check element-wise. Array form covers multi-regime cases (e.g. simultaneously DORA + CIRCIA).

## [Python 0.3.12 / TypeScript 0.2.10] - 2026-05-07

### Changed
- Both SDKs accept the cloud's wire-shape-aligned `payload_digest`. The plain `"sha256:<hex>"` string keeps parsing; the new upstream object form `{hash, size?, preview?}` (cloud 0.2.13+) parses too. Type widened on `SignatureResponse.payload_digest` (Python) and `SignActionOptions.payloadDigest` (TypeScript) to `str | dict | None` and `string | { hash; size?; preview? }` respectively.
- No callsite or behaviour change beyond the type widening. Receipts issued by older clouds and receipts issued by 0.2.13+ both verify on the same path.

## [Python 0.3.11 / TypeScript 0.2.9] - 2026-05-06

### Fixed
- DORA `incident_class` vocabulary now matches the canonical six-value list from JC 2024-33 Annex II field 3.23 (the EBA / ESMA / EIOPA Joint Committee Final Report on the draft RTS and ITS on incident reporting under Regulation (EU) 2022/2554, published 17 July 2024). Previous releases forwarded an opaque string with no client-side schema, so callers passing pre-final draft tokens hit a server 422 with no actionable error.

### Added
- `DORA_INCIDENT_CLASS_NAMESPACE` (Python `frozenset`, TypeScript `as const` tuple plus `DoraIncidentClass` union) exposing the six canonical values: `cybersecurity_related`, `process_failure`, `system_failure`, `external_event`, `payment_related`, `other`.
- Cloud accepts the canonical 6-value vocabulary only; SDK validates client-side before the HTTP roundtrip.
- `sign` / `agent.sign` raise `ValueError` (Python) or `AsqavError` (TypeScript) before any HTTP call when `incident_class` is outside the canonical six, matching how `RECEIPT_TYPE_NAMESPACE` is policed.

### Changed
- CLI `--incident-class` help text and `python/docs/CLI.md` flag table replaced the "DORA ITS vocabulary" gloss with the canonical six values and the JC 2024-33 citation.
- TypeScript field reference in `typescript/README.md` likewise points at the canonical list rather than a generic "DORA ITS code".
- Docs sweep aligning READMEs to the corrected -02 spec text.

## [Python 0.3.10 / TypeScript 0.2.8] - 2026-05-04

### Added
- IETF Compliance Receipts profile shipping. Both SDKs implement the wire shape from `draft-marques-asqav-compliance-receipts` (top-level `decision`, `signatureObject`, `anchors[]`), JCS canonicalisation for the signed envelope, hash-chain `previousReceiptHash`, retention floor, receipt type / reason vocabulary, and algorithm agility.

### Changed
- Wire-shape alignment with the -00 spec text: `decision` is the canonical top-level field; `signatureObject` carries the signed bytes plus algorithm metadata; `anchors[]` lists every external anchor (TSA, transparency log) with verification material inline.
- Hash chain field renamed `prev_chain_hash` -> `previousReceiptHash` to match draft naming.

### Notes
- TypeScript 0.2.5, 0.2.6, 0.2.7 were intermediate work-in-progress versions; 0.2.8 is the first ship that fully matches the -00 wire shape and is the recommended upgrade target.
- Python 0.3.7 was yanked on PyPI (shipped with `__version__="0.3.6"` baked in); 0.3.8 fixed that. 0.3.9 added the IETF profile incrementally; 0.3.10 finalises wire-shape alignment.

## [Python 0.3.8] - 2026-05-03

### Fixed
- `asqav.__version__` was drifting at `0.3.6` while the package metadata read `0.3.7`. Bumped to 0.3.8 because PyPI does not allow re-uploading a version. 0.3.7 has been yanked on PyPI.

## [Python 0.3.7] - 2026-05-03

### Added
- Initial round of IETF Compliance Receipts profile work (superseded by 0.3.9 / 0.3.10).

### Notes
- Yanked on PyPI: shipped with `__version__="0.3.6"` baked in. Use 0.3.8 or later.

## [Python 0.3.6] - 2026-05-02

### Added
- Python CLI gap-fill. New commands wrap the REST-only endpoints:
  - `asqav agents revoke <agent_id> [--reason X]` (Pro)
  - `asqav sessions list [--limit N] [--status X] [--agent ID]`
  - `asqav sessions end <session_id> [--status completed]`
  - `asqav policies list / create / delete` (Pro)
  - `asqav webhooks list / create / delete` (Pro)

## [Python 0.3.5] - 2026-05-02

### Added
- Tier gating for the `asqav` CLI. The Pro-only commands (`replay`, `preflight`, `budget`, `approve`) and Business-only `compliance export` now check the calling org's subscription tier via the new `GET /api/v1/account` endpoint and exit with a clean message + upgrade URL when the tier is insufficient. The server is still the source of truth; this only adds a friendly client-side gate. Older self-hosted deployments without `/account` are unaffected (the gate skips when the endpoint is unreachable).
- `__version__` in `asqav/__init__.py` realigned with the package version (was drifting at 0.3.1).

## [TypeScript 0.2.5] - 2026-05-02

### Added
- TypeScript `BudgetTracker` mirrors the Python primitive. `new BudgetTracker({ agent, limit, currency })` exposes `check(estimatedCost)` (fail-closed arithmetic) and `record(actionType, actualCost, context?)` which signs a `budget:*` action via the underlying agent. Re-exported from the package root.
- TypeScript `Agent.preflight(actionType)` mirrors the Python sibling. Combines the `/agents/{id}/status` revoke/suspend check and the `/policies` block-rule check into a single fail-open `PreflightResult` with a human-readable explanation.
- New `asqav` CLI (npm `bin` entry). Mirrors the Python surface: `verify`, `agents list/create/revoke`, `sessions list/end`, `replay`, `preflight`, `budget check/record`, `approve`, `compliance frameworks/export`. Pro/Business commands are tier-gated client-side via `GET /account` with the same fail-open semantics as the Python CLI.

## [Python 0.3.4 / TypeScript 0.2.4] - 2026-05-01

### Added
- User-intent envelope on `agent.sign(... user_intent={...})`. The end-user signs a digest of the action; the backend verifies the signature (ed25519, ecdsa-p256; webauthn advisory) and persists it on the same record. `SignatureResponse.user_intent_verified` surfaces the verdict. TypeScript mirror: `agent.sign({userIntent: {...}})` returns `userIntentVerified`. See `docs/user-intent.md`.
- Python CLI now has full API parity:
  - `asqav replay <agent_id> <session_id>` / `asqav replay --bundle <path>` (wraps `asqav.replay` / `asqav.replay_from_bundle`)
  - `asqav preflight <agent_id> <action_type>` (wraps `Agent.preflight`)
  - `asqav budget check` / `asqav budget record` (wraps `BudgetTracker`)
  - `asqav approve <session_id> <entity_id>` (wraps `approve_action`)
  - `asqav compliance frameworks` / `asqav compliance export --session ... --output ...` (wraps `export_bundle`)
- Pytest plugin: `pytest --asqav --asqav-agent=ci-runner` signs each test result and writes a Merkle-rooted compliance bundle. Configurable via `--asqav-output`, `--asqav-framework`. Registered as `project.entry-points.pytest11`. Programmatic: `asqav.pytest_plugin.make_bundle_from_report(...)`.
- Instructor integration: `from asqav.extras.instructor import AsqavInstructorHook; AsqavInstructorHook(agent_name=...).attach(client)` signs the five instructor hook events. Install with `pip install asqav[instructor]`.

### Examples
- `examples/budget_tracking.py`, `examples/scope_enforcement.py`, `examples/quarterly_audit.py`, `examples/human_approval.py`, `examples/streamlit_dashboard.py`, `examples/dify_workflow.md`. Each shows both the Python API and the matching CLI commands.

### Fixed
- `ReplayTimeline.verify_chain()` now actually verifies the chain. Each step records its predecessor's chain hash; `verify_chain()` recomputes and compares, flagging tampered or reordered entries instead of always returning `True`. Older bundles missing `prev_chain_hash` are flagged unverifiable rather than passing silently.

### Docs
- `docs/github-actions.md`: copy-paste workflow with `asqav doctor` as a CI gate plus a pytest fixture that captures signed actions into JSONL and exports a compliance bundle as a workflow artifact.
- CLI tagline updated to "AI agent governance - audit trails, policy enforcement, compliance" (matches the March 2026 repositioning).

## [Python 0.3.3 / TypeScript 0.2.3] - 2026-04-28

### Added
- Multi-party countersigning. `agent.sign(co_signers=[...])` records a list of peer agent_ids the original signer expects to countersign. Each peer calls `agent.countersign(signature_id)` and the server signs the SAME canonical bytes with that peer's ML-DSA key, then appends to the record's `co_signatures`. Mirror API in TypeScript: `agent.sign({coSigners: [...]})` and `agent.countersign(signatureId)`. No prompts or tool args travel during countersign; only signatures are added. Requires backend with co-signature support.

## [Python 0.3.2 / TypeScript 0.2.2] - 2026-04-30

### Added
- tool_name, model_name, parent_id surfaced into the hash-only metadata bag automatically when present in context (via `_tool_name` / `_model_name` / `_parent_id` keys) or passed as explicit kwargs to sign(). Powers the new agent graph view in the dashboard. No prompt or tool-arg content leaks; only the names.

## [Python 0.3.1 / TypeScript 0.2.1] - 2026-04-28

### Added
- `mode` client config: `auto` (default), `hash-only`, `full-payload`. SDK auto-detects cloud endpoints (`*.asqav.com`) and uses hash-only by default for GDPR data minimization. Self-hosted deployments keep full-payload.
- Public `canonicalize()` and `hash_action()` helpers (Python) and `canonicalize()` / `hashAction()` (TypeScript) that build the action fingerprint. RFC 8785 (the JSON format spec) plus SHA-256, or HMAC-SHA-256 with optional org salt.
- Cross-language conformance: both SDKs verified against the same `conformance/vectors.json`.

### Changed
- When `api_base_url` points at `*.asqav.com`, the SDK hashes context locally before sending. Set `mode="full-payload"` (or `ASQAV_MODE=full-payload`) to keep the previous behavior.

## Unreleased

- repo restructure: split Python and TypeScript SDKs into `python/` and `typescript/` subdirectories. Python users unaffected (`pip install asqav` continues working). TypeScript SDK introduced - `npm install @asqav/sdk`.
- workflows: path-filtered CI, dual-language publish on `py-v*` / `ts-v*` tags.

## [0.2.21] - 2026-04-25

### Added
- `sign_reasoning(prompt, trace, output, agent_id, ...)` helper. Signs hashes of prompt / reasoning trace / output as three-phase bilateral receipts on the existing hash store. Optional raw-store with retention parameter.

## [0.2.20] - 2026-04-20

### Added
- `BitcoinAnchor` dataclass and `SignatureResponse.bitcoin_anchor` field. Status is one of `none`, `pending`, `confirmed`, `failed` so callers can branch before treating a signature as Bitcoin-anchored. Addresses a gap where the ML-DSA `signature` was present but anchoring state was invisible to client code.

### Changed
- README tier clarification: removed the "managed KMS on Pro" claim, added an Enterprise sentence covering Managed KMS and bring-your-own KMS. Reflects the cloud's Managed KMS tier move.

## [0.2.19] - 2026-04-20

### Added
- `SignatureResponse` now exposes `policy_digest`, `policy_decision`, `authorization_ref` so third parties can reconstruct the JCS bytes that were signed and re-verify offline. Populated by both `Agent.sign` and `Agent.sign_batch` (and the async client).

## [0.2.18] - 2026-04-20

### Added
- `asqav demo` CLI command. Opens a local governance dashboard at `http://localhost:3030` with 4 pre-loaded scenarios (rm -rf, fintech wire transfer, kubernetes scale-to-zero, clinical lab order). No signup, no Docker, no API key. Each decision produces an HMAC-signed receipt that the dashboard re-verifies in the browser. Use it to preview the approval-card UX before wiring the real SDK into an agent.

## [0.2.17] - 2026-04-19

### Added
- `sign_batch` method on `Agent` for submitting many signatures in one HTTP round-trip.
- DSPy integration hooks via `asqav.extras.dspy`.

## [0.2.16] - 2026-04-18

### Added
- `SignatureResponse` now exposes `algorithm`, `chain_hash`, `rfc3161_tsa`, `rfc3161_serial`, `scan_result` fields.
- `Agent.decommission`, `Agent.quarantine`, `Agent.unquarantine` helpers.
