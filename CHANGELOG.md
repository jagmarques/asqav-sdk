# Changelog

All notable changes to the Asqav SDK are documented here.
Both language halves version together; tags are independent (`py-v*`, `ts-v*`).

## [Unreleased]

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
