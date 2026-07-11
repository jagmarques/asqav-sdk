# Changelog

All notable changes to the Asqav SDK are documented here.
Both language halves version together; tags are independent (`py-v*`, `ts-v*`).

## [Unreleased]

## [0.8.2] - 2026-07-11

Release-hygiene patch that re-ships the 0.8.1 feature set from `main`. The `asqav`
0.8.1 artifacts on PyPI were built from commit 828bb820, which predates the public
no-key `verify()` wrapper and the fail-clean verification surface the 0.8.1 notes
below describe. 0.8.2 builds those notes from `main`, so the published package
matches its changelog. The npm package `@asqav/sdk` had no 0.8.1 release (that
publish run failed at the build step), so npm moves from 0.8.0 straight to 0.8.2.

### Fixed

- Rebuild the 0.8.1 feature set from `main` for both language halves. The `asqav`
  PyPI 0.8.1 build sits at commit 828bb820, which predates the top-level `verify()`
  export and the fail-clean verify surface, so 0.8.2 rebuilds from `main` where both
  are present. `@asqav/sdk` ships 0.8.2 as its first release after 0.8.0.

## [0.8.1] - 2026-07-08

Everything on `main` ships in this patch, so the entries below are consolidated
here from the working set. The date refreshes at publish.

### Added

- **Default-on LangChain governance (Python + TypeScript).**
  `enable_langchain_governance()` / `enableLangchainGovernance()` registers a
  configure hook and binds an `AsqavCallbackHandler` to the run context, so
  every chain, tool, and LLM run signs an Asqav receipt with no per-invoke
  callback config. The manual `AsqavCallbackHandler` path is unchanged. (#346)
- **Default-on crewAI governance (Python).** `enable_crew_governance(crew)`
  wires a crew's `step_callback` and `task_callback` to an `AsqavCrewHook`, so
  every step and task signs a receipt with no manual per-crew callback config.
  Duck-types the crew, so `crewai` need not be installed to import the function. (#347)
- Hardened the default-on LangChain and crewAI adapters: the LCEL/RunnableLambda
  path where `serialized` arrives as None falls back to a stable chain label,
  the LangChain configure hook registers at most once so a repeat call cannot
  double-sign, and `enable_crew_governance` is idempotent. (#361)
- **Public no-key `verify()` (Python + TypeScript).** `asqav.verify(signature_id)`
  and the top-level `verify()` export fetch the public verify verdict with no API
  key, then recompute the receipt's `chain_hash` locally (the SHA-256 over the RFC
  8785 canonical payload, the value a successor carries as `previousReceiptHash`)
  so the chain link is reproducible offline. The flagship README "Verifying a
  receipt" example now runs as written.
- Added `tsx` as a TypeScript devDependency. It is the runner the
  quickstart example already documents (`npx tsx quickstart.ts`).

### Changed

- Relicensed asqav-sdk from MIT to Elastic License 2.0 (ELv2) for versions
  released after 0.8.0. ELv2 permits free use and modification but restricts
  offering the SDK as a hosted or managed service to third parties.
  Version 0.8.0 and all versions published before it remain MIT-licensed
  irrevocably. The conformance test vectors in `conformance/` remain
  Apache-2.0 licensed.

### Removed

- Dropped the in-SDK MCP governance adapter (`enable_mcp_governance` /
  `enableMcpGovernance`) from both language halves, along with its `mcp`
  Python extra, its `@modelcontextprotocol/sdk` optional peer and
  `./extras/mcp` subpath export, its runnable examples, its tests, and
  `docs/integrations-mcp.md`. Governance integration moves to a hook-based
  approach, so the MCP-specific wrapping path leaves the SDK. This does not
  touch the `protectmcp:*` receipt vocabulary or the `mcp_proxy`
  capture-topology token, which stay as part of the wire contract.

### Fixed

- Required-field deserializer guards (Python + TypeScript). `Agent.sign()`,
  `Agent.countersign()`, `verify_signature()` / `verifySignature()`, and
  `Agent.create()` / `Agent.get()` (plus their async, batch, and
  session-listing counterparts) now raise a typed `AsqavResponseError` when
  the cloud response omits a required field, instead of a raw `KeyError`
  (Python) or a silently `undefined` value (TypeScript). Both languages
  route through one shared guard, `require_field()` / `requireField()`.
- The standalone receipt verifier (`python -m asqav.verifier.verify_receipt`)
  now fails clean on missing, empty, or non-object input: it prints one readable
  line and exits nonzero instead of leaking a urllib/json traceback, and it reads
  a receipt from stdin via `--receipt -`. The verifier README worked example
  points at the live public fixture `sig_example_regulator_cold_verify_2026`.
- The standalone verifier and the offline `verify_receipt_offline` API now fail
  clean on every remaining malformed-input shape: a NaN/Infinity receipt, a
  non-object `signature` block, a malformed or `public_key`-less JWKS, a
  non-string `sig`, a binary / non-UTF8 `--receipt` file, and a falsy non-list
  `anchors` value (`{}`, `""`, `0`). Each prints one readable line with the
  input-error exit code, and the offline API returns INCOMPLETE rather than
  raising on a malformed JWKS.
- The `asqav verify` CLI reports a connection or timeout failure as one readable
  line and a nonzero exit, instead of a raw httpx traceback.
- The public no-key `verify()` (TypeScript) targets the base URL configured via
  `init()` and reads `ASQAV_API_URL` when no key is set, so it honors a custom
  endpoint the same way the Python half does.

### Security

- Offline receipt verification blocks backdating on revoked keys. A revoked
  key with a `revoked_at` timestamp returns INCOMPLETE instead of PASS for
  any receipt whose self-attested `issued_at` predates the revocation, unless
  a trusted time anchor corroborates the timing. An attacker holding the
  compromised key cannot re-sign with a backdated timestamp and read PASS
  offline. (#362)

## [0.8.0] - 2026-07-03

### Added

- **Standards-interop doors (Python and TypeScript).** Additive utilities that
  take one Asqav receipt and return it wrapped as the envelopes agent builders
  already consume, with one byte-identical inner receipt inside each. Envelopes:
  W3C Verifiable Credential 2.0, CloudEvents 1.0, an OpenTelemetry GenAI
  attribute map, a C2PA third-party assertion, and an ERC-8004 validation
  request shape. Every door has an inverse, and `extract_receipt` /
  `extractReceipt` recovers the inner receipt from any envelope. Pure and
  offline: no signing, no network, no input mutation. Presentation only; the
  authoritative signature stays inside the embedded receipt.
- Cross-SDK parity holds on the JSON Canonicalization Scheme safe input domain
  (object keys in the Basic Multilingual Plane, integers within the IEEE-754
  safe range), pinned by a shared golden both suites assert against. Outside
  that domain the two halves can diverge, which the doors docs state plainly.
## [0.7.0] - 2026-07-01

### Added

- **One-call `govern()` entrypoint (Python and TypeScript).** Composes
  `init()` + `Agent.create()` (and hook registration) for the common case, so a
  new integration is a single call. Strictly additive; `init`/`Agent.create`/
  `Agent.sign` are unchanged.
- **Schema-driven structured receipts (opt-in).** Pass `context_schema`
  (Python) / `contextSchema` (TypeScript) to `Agent.sign()` to validate and
  normalise the `context` before signing, so audit trails are structured and
  queryable. Invalid context raises a clear error before any network call. No
  schema means today's exact behaviour (byte-identical signed body). A callable
  validator is also accepted for full JSON Schema.
- **Pluggable detectors (bring-your-own DLP/policy).** `register_detector` /
  `registerDetector` runs a detector inside `Agent.sign()`; its verdict is
  recorded into the signed receipt under `_detectors`, and a deny raises
  `DetectorBlockedError` before signing. Fail-closed on detector error by
  default (`fail_open` / `failOpen` opts out). Ships two reference detectors
  behind optional extras: `PresidioDetector` (PII) and `OpaDetector` (Open
  Policy Agent policy-as-code). No new hard runtime dependency; additive.

## [0.6.5] - 2026-06-27

### Fixed

- **Preflight fails closed on a non-object `/status` response in TypeScript, matching Python.**
  If the status endpoint returns a truthy non-object body (a string, number, or any primitive),
  the TypeScript client now treats the check as failed and blocks the action. A well-formed
  response is an object. Anything else is anomalous and should not leave the advisory gate
  cleared.

## [0.6.4] - 2026-06-27

### Fixed

- **Preflight fails closed on a fetch error in TypeScript, matching Python.** The
  TypeScript `PreflightResult` now carries a `checksComplete` boolean that is set
  false when the `/status` or `/policies` fetch fails, and `cleared` folds it in, so
  a failed check blocks the action instead of clearing.
- **Preflight fails closed on non-list `/policies` responses.** If the policy
  endpoint returns anything other than an array, preflight blocks the action rather
  than silently passing. The SDK preflight is a client-side advisory check. This
  fix stops a malformed or empty response from acting as a bypass.
- **Namespace normalization strips case and whitespace before policy matching.**
  `DATA:WRITE:SQL:DELETE` and `  data:write:sql:delete  ` now resolve to the same
  candidate, closing a bypass via case or padding variation.
- **Extended destructive-verb set covers SQL mutation verbs beyond DELETE.**
  `GRANT`, `REVOKE`, `REPLACE`, `COPY`, and `UPSERT` are now included alongside
  `DELETE`, `DROP`, `TRUNCATE`, and `ALTER` when the preflight checks for
  destructive SQL under a `data:write` namespace. Both halves carry the same list.

## [0.6.3] - 2026-06-27

### Fixed

- **The neutral verifier fails clean on malformed input (Python and TypeScript).**
  A non-object receipt, a non-string `alg`, and a timezone-mismatched `revoked_at`
  or `issued_at` return a verdict instead of raising. No forged, tampered,
  downgraded, or cross-format receipt verifies as valid.
- **The TypeScript verifier checks signing-key revocation, for parity with
  Python.** A receipt signed by a revoked key fails in both verifiers.
- **Preflight blocks destructive SQL routed under a `data:write` namespace.**

### Documentation

- Correct the SDK license to MIT in the Python and TypeScript READMEs.

## [0.6.2] - 2026-06-23

### Added

- **litellm cloud-signing callback and `verify-litellm-log` CLI.** An opt-in
  `AsqavLogger` signs each litellm record server-side, and the CLI verifies a
  litellm log offline.

## [0.6.1] - 2026-06-23

### Added

- **`asqav hook` CLI subcommand for Claude Code harness hooks**, with a hooks
  guide covering fail-open and fail-closed modes.

### Fixed

- **`AsyncAgent.preflight` fails closed on a fetch error** and resolves semantic
  policy pattern aliases.
- **The verifier gates on signing-key status in `run_structured`**, so a receipt
  signed by a revoked key fails offline.
- **Offline verification rejects a revoked signing key.**

### Removed

- **Dropped the `asqav-mcp` config generation** from the SDK.

## [0.6.0] - 2026-06-20

### Added

- **Offline / air-gapped receipt verification (Python + TypeScript).**
  `fetch_jwks()` / `fetchJwks()` snapshots the Asqav JWKS while online.
  `verify_receipt_offline(receipt, jwks)` / `verifyReceiptOffline(receipt, jwks)`
  then verifies ML-DSA-65 signatures, Ed25519, and ES256 entirely in-process
  without any network call. See `docs/offline-verification.md` for the full guide.

- **Real-cloud ML-DSA-65 known-answer conformance vector (Python + TypeScript).**
  A signed receipt produced by the Asqav cloud against the live ML-DSA-65 key
  (agent `asqav-06`) is committed to the test suite as a KAT vector and
  verified on every CI run. Python uses `dilithium-py>=1.0.0`; TypeScript uses
  `@noble/post-quantum ^0.6.1`. Both halves must pass the same vector.

### Changed

- **`llamaindex` and `all` extras drop `llama-index-core` from their pin list.**
  `llama-index-core` has a hard dependency on `nltk`, which carries
  CVE-2026-54293 with no upstream patch. The integration module
  (`extras/llamaindex.py`) remains; install `llama-index-core` manually
  alongside `asqav` and the lazy-import guard picks it up automatically.

### Dependencies

- Python: `dilithium-py>=1.0.0`, `cryptography>=42` added to the `verify` extra.
- TypeScript: `@noble/post-quantum ^0.6.1` added as a runtime dependency.
