# Changelog

All notable changes to asqav (the SDK) will be documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [SemVer](https://semver.org/).

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
