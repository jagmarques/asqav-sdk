# Changelog

All notable changes to asqav (the SDK) will be documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [SemVer](https://semver.org/).

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
