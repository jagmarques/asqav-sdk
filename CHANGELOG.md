# Changelog

All notable changes to the Asqav SDK are documented here.
Both language halves version together; tags are independent (`py-v*`, `ts-v*`).

## [Unreleased]

## [0.10.5] - 2026-09-01

### Added

- **Acceptor-side admission control, in both halves.** `check_peer_receipt` /
  `checkPeerReceipt` answer whether an inbound agent-to-agent action may be
  admitted given the receipt the peer presented, returning an `AcceptorDecision`
  that names one refusal edge rather than a wall of axes. It runs off the same
  shared oracle the offline verifier uses, so an acceptor and an auditor cannot
  disagree about the same bytes.
- Three of its rules deliberately do **not** follow from the verdict, because a
  peer can weaken the evidence without ever producing an unverified receipt.
  *Expiry*: the verifier reports expiry on its own axis and never folds it, which
  is right for a verifier and wrong for a party deciding about an action happening
  now, so a lapsed receipt is refused here. *Seq downgrade*: absence of a counter
  stays legal in general, but an acceptor holding a predecessor that carried one
  is watching the exact transition that hides a withheld receipt. *Challenge*: a
  challenge the acceptor issued and the receipt does not answer proved nothing, so
  it is required once issued rather than checked only when present. Refusals are
  ordered, so the reason is deterministic for the same inputs.
- **Framework adapters for that decision**: `AcceptorMiddleware` (ASGI, Python)
  and `acceptorMiddleware` (Connect-style, TypeScript), plus
  `DEFAULT_RECEIPT_HEADER`. They add plumbing and no policy — an acceptor that
  mounts the middleware and one that calls the function directly refuse the same
  receipts. They **fail closed**: a request carrying no receipt, or one whose
  header does not parse, is refused, since middleware that admitted an unsigned
  request while refusing a badly-signed one would make sending nothing the
  cheapest bypass. Non-HTTP scopes (lifespan, websocket) pass through, carrying no
  receipt and no inbound action. `predecessor_for` and `challenge_for` are
  caller-supplied hooks, because where that state lives is the deployer's choice.
- **The verifier now reports which check failed first**: `first_failing_edge` /
  `firstFailingEdge` on the result. A verdict alone hides an ordering divergence —
  two verifiers can agree a receipt is unverified while disagreeing about which
  check failed first, which is the difference between a debuggable report and a
  guess. The definition is not "the first non-PASS axis"; it mirrors
  `fold_verdict`'s two exclusions, since the expiry axis never folds the verdict
  and a SKIPPED chain is tolerated where any other SKIPPED blocks. An edge is
  named for exactly the unverified verdicts, and the shared axis prefix order is
  pinned so a refactor that reorders the checks fails a gate instead of quietly
  renaming which edge is first.
- Two key-binding conformance vectors in the shared corpus (`asqav-21`
  key thumbprint binds, `asqav-22` key substituted), with a generator that mints
  them from a published seed.

## [0.10.4] - 2026-09-01

### Fixed

- **A valid receipt could be reported as provably bad, decided by nothing but the
  second of the minute it was signed in.** The TypeScript ISO parser range-checked
  the seconds field as `Number(ss) > 59`. On a fractional stamp that string is
  `"59.656"`, and `59.656 > 59`, so the parser returned `null` and the axis called
  the value unreadable. `Date#toISOString` always emits milliseconds, so roughly
  one receipt in sixty — any minted during the 59th second — was refused. It
  reached both axes sharing the parser: `expires_at` FAILed as `unverifiable`, and
  `issued_at` FAILed as **`invalid`**. It was also a cross-language split, since
  Python range-checks a two-digit capture and verified the same bytes. Fixed to
  range-check whole seconds only, mirroring Python; second 60 is still refused.
  **Anyone verifying receipts with the TypeScript half should upgrade.**

### Added

- Four `seq` continuity conformance vectors in the shared corpus, and the
  fractional-second cases in `verifier/axis-parity-cases.json` — the shared table
  both language suites drive, so a future divergence of this kind fails in CI
  rather than in a caller's verdict.

## [0.10.3] - 2026-09-01

### Added

- **`seq` continuity verification axis, in both halves.** The platform binds a
  server-built per-agent counter into every compliance receipt, so a gap in the
  series proves receipts were withheld *without needing the withheld receipts*.
  The chain axis alone cannot do this: hash linkage detects modification,
  reordering and removal of interior records, but a gap and a duplicate position
  are indistinguishable to it. A gap now FAILs as terminal `invalid` and names
  the count, e.g. `seq gap: 4 receipt(s) withheld between 1 and 6`; a
  non-monotonic counter and a malformed one FAIL too.
- The axis is **never SKIPPED**. A counter-less receipt PASSes with a note, since
  `fold_verdict` blocks on any non-chain SKIPPED and a bare skip would regress
  every receipt minted before the counter shipped from verified to unverified. A
  corpus-wide gate holds that property for future changes.
- A counter is only compared within one format's own series, and a `seq` sitting
  on a hash-mode receipt binds nothing (hash mode signs the flat field set only),
  so it is treated as an unsigned claim rather than read as evidence.

### Fixed

- The TypeScript ACTA adapter named revision `-01` while its Python sibling named
  `-02`. Both compute the signing input as the canonical JCS bytes of the payload
  with no pre-hash, which is the `-02` rule, so the TypeScript label was stale.
  Comment only; behaviour is unchanged.

## [0.10.2] - 2026-08-31

### Added

- **`payload_digest` verification axis: the digest is recomputed from the
  `context` the receipt carries itself.** A receipt carrying both is checkable
  with no external data, and the two disagreeing proves one of them is a lie.
  Nothing compared them before, so an issuer could sign a benign context beside
  a digest committing to something else entirely and every verifier passed it.
  A mismatch is terminal invalid. Absence PASSes on both sides: hash mode carries
  no context, and a payload-mode receipt may legitimately omit it under redaction.
- **`counterparty` verification axis.** `counterparty_binding` asserts that a
  counterparty acknowledged the Action. The hosted verifier resolves it against
  its own database, but an offline third party has none, so a fabricated binding
  pointing at a receipt that never existed previously reached a plain `verified`
  verdict with the corroboration claim unexamined. Absence PASSes; a malformed
  binding FAILs as invalid; a binding the verifier cannot resolve reports SKIPPED,
  which blocks, so an unchecked claim can never read as corroborated. Supplying
  the originating receipt recomputes the envelope digest and cross-checks
  `expect_ack_from`.
- **The oracle path now emits the `skew` axis.** `check_skew` existed and the
  standalone verifier used it, but the oracle never emitted it, so the oracle
  accepted a receipt claiming an `issued_at` in 2099 that the standalone verifier
  had always refused. Closes a parity gap between the two offline surfaces.
- 18 shared cross-language vectors in `verifier/axis-parity-cases.json` drive both
  language halves.


## [0.10.1] - 2026-08-31

### Added

- **`key_binding` verification axis: the signed `key_thumbprint` is recomputed
  from the resolved signing key and compared.** Both language halves gain the
  check for the JWK Thumbprint (RFC7638) over the draft-ietf-cose-dilithium AKP
  form `{alg, kty, pub}`, with `pub` in unpadded base64url. A receipt whose
  bound digest names a key other than the one that verified reports
  `key_substituted`; `key_binding` is an invalid-fail axis in both cores, so the
  verdict folds to `unverified` / `invalid` and is never softened to a warning
  or reported as verified.
- Absence of `key_thumbprint` PASSes the axis with a "binding not checked" note
  rather than skipping, so every receipt issued before the member existed stays
  conformant; a skip outside `chain` blocks a verdict. The two cases that cannot
  be recomputed - no key resolved, and a resolved key that is not raw ML-DSA of
  the width its own `alg` fixes, which is how a KMS-backed row storing a PEM
  presents - report SKIPPED, which also blocks, so an unverifiable binding is
  never read as verified.
- `key_thumbprint` joins the hash-mode unsigned-claim guard. Hash mode signs a
  flat 11-field object with no room for the member, so a thumbprint pasted onto
  a hash-mode receipt FAILs the structure axis instead of appearing bound.
- 15 axis cases and 4 thumbprint digest vectors in the shared
  `verifier/axis-parity-cases.json`, read by both suites, including a vector
  that pins the base64url alphabet: an implementation reusing the directory's
  standard-base64 `public_key` alphabet computes a digest no other verifier
  reproduces, and a case binding exactly that digest must FAIL.


## [0.10.0] - 2026-08-27

### Added

- **New oracle format: W3C Verifiable Credentials 2.0 secured by
  DataIntegrityProof with the eddsa-jcs-2022 cryptosuite (W3C TR
  vc-di-eddsa).** The seventh format the universal neutral verifier checks, in
  both language halves together. The signature is Ed25519 over
  `SHA-256(JCS(proofOptions)) || SHA-256(JCS(unsecuredDocument))` under strict
  RFC 8785 JCS, with proofOptions the proof minus `proofValue` and the
  unsecured document the credential minus `proof`; `proofValue` is the raw
  signature multibase base58btc encoded. The cryptosuite's proof `@context`
  prefix transform is enforced (the document `@context` must start with the
  proof's, and the proof's substitutes it before canonicalization). The adapter
  is fail-closed stricter than the suite spec: `proofPurpose` must be
  `assertionMethod` and the verificationMethod DID must equal the issuer DID,
  mirroring the agentreceipts adapter. `validFrom`/`validUntil` report on the
  expiry axis, which never folds the verdict (criterion 426). A sibling
  cryptosuite reports as an algorithm mismatch (`invalid`), never a silent
  re-dispatch. Eight new conformance vectors (`w3c-vc-01..08`) pin the did:web
  and did:key happy paths, tamper rejection, a wrong published key, fail-closed
  offline resolution, the expiry rule, and strict ingest; the corpus lock is
  re-frozen at v1 and both lock paths re-derive every pin.

- **The shared DID resolver accepts an injected DID document** in the
  `did_map.json` slot: the bytes the did:web fetch would have returned. The
  resolver walks `verificationMethod` and `assertionMethod` and extracts an
  Ed25519 key from `publicKeyMultibase` (Multikey), `publicKeyJwk`
  (OKP/Ed25519), or `publicKeyBase58`. The oracle still never fetches a DID
  document: an unmapped DID reads `unverified`/`unverifiable`, fail closed.
  The raw-key injection shape (hex string or bytes) is unchanged.

### Removed

- **The `asqav shadow-ai` CLI subcommand, its templates, and its tests.**
  Shadow-AI detection is retired as a product focus; platform observations and
  SIEM ingest stay, unbranded.

## [0.9.0] - 2026-08-26

### Changed

- **BREAKING: the verifier speaks a three-verdict vocabulary and never collapses
  failure classes (criteria 418/438).** Every public verifier surface — the
  standalone `verify_receipt.py` (text + structured + exit codes), the oracle
  `verify()` / `VerifyResult`, the oracle runner, the TypeScript `verify()` /
  `verifyReceiptOffline`, and `verify_attestation_offline` — now reports
  `verified` | `verified_keyed` | `unverified` instead of PASS | FAIL |
  INCOMPLETE. Every `unverified` verdict carries a `failure_class` of `invalid`
  (a check ran and a cryptographic/policy binding failed: signature mismatch,
  chain-link mismatch, invalid anchor, counterparty binding mismatch, revoked or
  changed signer key, algorithm mismatch, or `issued_at` future-skew) or
  `unverifiable` (recomputation could not complete: unresolvable key, missing or
  broken chain predecessor, malformed member, canonicalisation or parse failure,
  unresolvable policy digest, or a pending anchor without proof). The two are
  never collapsed, and a receipt is never reported verified when recomputation
  failed. A keyed digest (e.g. HMAC-SHA256) that fully checks reports
  `verified_keyed`, never plain `verified`. Exit codes keep the stable mapping:
  `verified`/`verified_keyed` → 0, `unverified`+`invalid` → 1,
  `unverified`+`unverifiable` → 2 (the blocked state INCOMPLETE used to carry).
  The per-axis PASS/FAIL/SKIPPED tokens stay internal. Both language halves and
  the conformance corpus agree byte for byte on verdict and failure_class.

- **BREAKING: duplicate JSON member names are rejected at any depth (criterion
  419).** Every receipt- and record-parsing path — `verify_receipt` ingest, the
  oracle runner and CLI, the attestation signed-message re-parse, the doors OTel
  receipt recovery, the CLI JSON arguments and signing-log records, the API
  response parsers, and the TypeScript `parseJsonPreservingFloats` plus the
  receipt/bundle/doors/DSSE loaders — now fails closed on a duplicated member
  name, before any hashing, canonicalisation, or signature check. The stdlib
  last-wins behaviour would hash the bytes an attacker kept and drop the ones
  they replaced, so a duplicate is a terminal parse failure, reported
  `unverified`/`unverifiable`.

- **BREAKING for hand-assembled receipts: an anchor `value` must be one unwrapped
  base64 token.** A value carrying whitespace, including a trailing newline or MIME
  line wrapping, reported PASS on the anchors axis and now reports FAIL. Both language
  halves change together. This bites exactly one workflow: piping a shell base64 tool
  into an anchor field. `openssl base64` wraps at 64 characters, GNU `base64` wraps at
  76, and BSD `base64` appends a trailing newline, so all three produced a value that
  the axis accepted and now refuses.

  Migration is one flag: `openssl base64 -A`, or `base64 -w0` on GNU. Any value already
  produced by the Asqav signer or the SDK is unaffected, because those encode through
  `base64.b64encode` and its TypeScript equivalent, neither of which wraps.

  The trade is deliberate. Whitespace is dropped by the same lenient decode that let a
  forged value launder junk into real bytes, so tolerating an embedded newline in the
  anchors axis means tolerating `MTIz NA==` and, on the same path, `QU!JD`. Anchors sit
  outside the signed bytes, so the axis prefers a canonical value over a permissive one.
  A `value` is a JSON string and JSON has no line-length limit, so wrapping carries no
  benefit inside a receipt.

### Fixed

- **The published `@asqav/sdk` LICENSE file did not match the license
  `package.json` declares.** The relicense to Elastic License 2.0 (#344)
  updated the root `LICENSE`, `package.json`, and `pyproject.toml`, but left
  `typescript/LICENSE` on its pre-relicense MIT text. `@asqav/sdk@0.8.2` and
  `0.8.3` on npm both shipped declaring `LicenseRef-Elastic-License-2.0`
  while bundling an MIT-worded LICENSE file inside the tarball. Those two
  published versions are immutable and keep shipping that text.
  `typescript/LICENSE` now matches the repo's canonical Elastic License 2.0
  text, so the next publish carries the correct grant. A test
  (`tests/license-consistency.test.ts`) pins the packaged LICENSE file
  against the declared license going forward.

- **A forged anchor value cannot report as a present anchor.** The anchors axis
  decoded the value leniently, and a lenient base64 decode drops every character
  outside the alphabet, so an all-punctuation value such as `!!!!` decoded to zero
  bytes and still read as an anchor. `anchors` sits outside the signed bytes, so a
  receipt holder picks that value freely. The axis now requires the real alphabet and
  at least one decoded byte, in the Python verifier and in its TypeScript mirror.
- **The anchor verdict is the same on every supported interpreter.** The alphabet and
  padding decision lives in an explicit regex rather than in
  `base64.b64decode(validate=True)`, whose strictness changed between CPython 3.11 and
  3.12. Delegating to it made surplus padding such as `AAAA====` decode to real bytes
  on 3.11 and raise on 3.12, which read as a PASS on one supported interpreter and a
  FAIL on another. Measured across Python 3.11, 3.12 and 3.14 and the TypeScript half,
  the 971-value corpus now answers identically in all four.

- **A rewritten `signature.kid` cannot buy a PASS for a revoked key.** `kid` sits
  outside the signed payload bytes, so a receipt holder can change it without
  disturbing the signature. The signature axis resolved its key through an
  agent-id fallback that the `key_status` and `issuer_bind` axes did not share, so
  a kid the directory answers for nothing left those two axes emitting nothing at
  all. The verdict aggregate can block on an axis reporting SKIPPED but not on one
  that was never emitted, so a receipt signed by a REVOKED key read PASS offline.
  Both surfaces now resolve through one shared entry, so every axis weighs the key
  that actually signed. A revoked-key receipt reports FAIL whichever route
  resolves it.

- **An org with two agent keys can verify its own receipts.** A cloud receipt puts
  the org id in `signature.kid` and signs with the agent's own key, and the public
  directory publishes `issuer_id` on every key the org owns, so an org-shaped kid
  matched each sibling alike and list position decided which one answered. A
  receipt from the second agent was checked against the first agent's public key,
  so a sound signature read as a mismatch and the `key_status` and `issuer_bind`
  axes reported a key that never signed. Signing-key resolution takes an exact key
  id first, then the `agent_id` plus `issuer_id` pair the signed bytes carry, and
  only then the bare-kid issuer match. The TypeScript verifier gains the same
  three-step resolution and reads every axis off the one resolved entry, so both
  halves answer identically.

- **Every docs link in the README reaches a real page.** The wire-field
  reference and the self-hosted signer guide named five docs slugs the site
  serves no page for, so a reader following one out of the published package
  description landed on a 404. Each link moves to the live page that documents
  the field: `expires_at` and `nonce` to time-bound receipts,
  `config_manifest_digest` to configuration change receipts,
  `cve_inventory_digest` to the CVE inventory digest page, and the air-gapped
  signer pointer to offline and air-gapped verification.

### Security

- **Corrected security claim: the `nonce` wire field carries no replay
  protection.** The shipped docstrings and both README wire-field tables told
  readers that the cloud verifier rejects a duplicate nonce per
  `(agent_id, action_ref)` inside the validity window. No such control exists.
  The caller-supplied nonce is written to the signature record and read back by
  nothing: no unique index covers it, no query looks a value up to decide whether
  it was seen, and it reaches the signed bytes in none of the three signing modes.
  Re-sending a sign request with the same nonce produces a second accepted
  signature.

  Anyone who read that sentence and treated the field as a replay control had no
  such protection, which is why this lands as a corrected security claim rather
  than a wording tidy. The enforced half of the pair is real and unchanged:
  `valid_seconds` and `expires_at` set the record's validity horizon, and verify
  answers `signature_expired` past it. That is the control to bound replay with.

  The nonce bullet also drops its time-bound-receipts pointer. That page repeats
  the same claim, and a corrected sentence sitting next to a link that asserts the
  opposite still misleads. The `expires_at` bullet keeps the link.

  The 0.8.3 package descriptions on PyPI and npm are immutable, so a reader
  landing on either registry page keeps seeing the old sentence until the next
  release republishes the README.

### Documentation

- **The rule-9 guard message both READMEs quoted as verbatim was never the
  real one.** `python/README.md` and `typescript/README.md` claimed a missing
  `config_manifest_digest` on a `configuration_change` receipt raises
  `false_attestation_guard: ...`. The real prefix, in both language
  implementations, is `configuration_change_missing_config_manifest_digest:
  ...`; `false_attestation_guard` is the real prefix for the separate rule-8
  guard only. A reader who pattern-matched the documented string would never
  catch the real exception. Corrected in both READMEs.
- **`python/README-ZH.md` still stated the pre-0.8.1 MIT license and a false
  `Agent.create()` capability.** The license line was never updated past the
  MIT-to-Elastic-License-2.0 relicense below. Separately, the ZH doc claimed
  `Agent.create(algorithm="ed25519"|"es256")` is accepted for "classical
  identity"; the client's own docstring says the cloud returns 400 for either,
  matching the (correct) EN docs. Also repointed a dead `asqav.com/roadmap`
  link (404) to `asqav.com/docs`.
- **`CONTRIBUTING.md`'s setup steps never accounted for the `python/` +
  `typescript/` split.** Every documented command (`uv sync`, `pip install -e
  ".[httpx]"`, `pytest tests/`, `ruff`, `mypy`) was written as if run from the
  repo root, which has no `pyproject.toml` or `tests/` of its own; each one
  fails immediately as written. Added the missing `cd python` / `cd
  typescript` steps, a TypeScript setup path (previously absent), and fixed
  the plain-venv extra (`.[httpx]` alone never installs `pytest`/`ruff`/`mypy`;
  needs `.[dev,httpx]`).
- **`SECURITY.md`'s Supported Versions table was frozen at `0.5.x`, not the
  published `0.8.3`.** Replaced the version-number table with a policy
  statement (latest published version only) so it cannot go stale the same
  way again.

## [0.8.3] - 2026-07-20

Patch release with a verifier correctness fix and a docs alignment.

### Fixed

- **Offline verify must not treat anchor presence as trusted timing.** The
  neutral verifier client returned an `anchored` verdict when an anchor was
  present on the receipt, even if the anchor was not independently verified.
  The client reports anchor presence as metadata only, and the verdict
  reflects the cryptographic verification state alone. (#376)

### Changed

- Aligned the README license sections with the actual LICENSE file (ELv2 for
  the SDK code, Apache-2.0 for the conformance vectors). (#374)

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
