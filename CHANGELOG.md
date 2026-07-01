# Changelog

All notable changes to the Asqav SDK are documented here.
Both language halves version together; tags are independent (`py-v*`, `ts-v*`).

## [Unreleased] - License change

### Changed

- Relicensed asqav-sdk from MIT to Elastic License 2.0 (ELv2) for versions
  released after 0.7.0. ELv2 permits free use and modification but restricts
  offering the SDK as a hosted or managed service to third parties.
  Version 0.7.0 and all earlier published releases remain MIT-licensed
  irrevocably. The conformance test vectors in `conformance/` remain
  Apache-2.0 licensed.

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
