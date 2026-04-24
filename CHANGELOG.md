# Changelog

All notable changes to asqav (the SDK) will be documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [SemVer](https://semver.org/).

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
