# Changelog

All notable changes to Asqav (the SDK) will be documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [SemVer](https://semver.org/).

## [Unreleased]

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
- `VerificationResponse.verifierSignature` (Python: `verifier_signature`). The cloud no longer emits this field; it always projected as `null`, so the typed surface is dropped rather than left in as dead weight.
- Inner anchor `type` rejects the `"ots"` alias. The cloud emits only `"opentimestamps"` (canonical) and `"rfc3161"`; the SDK type stubs are tightened accordingly.

### Changed
- `generate_local_keypair` no longer accepts `"ml-dsa-65"`. ML-DSA-65 keypair generation is server-side only (cloud KMS); call `asqav.Agent.create(name, algorithm='ml-dsa-65')` to mint one. `SUPPORTED_ALGORITHMS` is now `{ed25519, es256}` and the default for `generate_local_keypair()` is `ed25519`. The previous behaviour raised `NotImplementedError` at call time; this is now a `ValueError` at validation time, with the typed `Algorithm` Literal narrowed to reflect what the SDK actually mints locally.
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
- **Removed flat-field projection fallbacks**: the projection helpers no longer rebuild `{alg, kid, sig}` from `algorithm` + `signature_b64`. Older receipts return None / undefined from the helpers.
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
- README tier clarification: removed stale "managed KMS on Pro" claim, added an Enterprise sentence covering Managed KMS and bring-your-own KMS. Reflects cloud v0.2.10 tier move.

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
