# Changelog

All notable changes to the Asqav SDK are documented here.
Both language halves version together; tags are independent (`py-v*`, `ts-v*`).

## [Unreleased]

### Fixed

- **The `llamaindex`, `dspy`, `smolagents` and `crewai` extras install the framework they name.**
  They were empty, so `pip install asqav[llamaindex]` pulled in no LlamaIndex. Each now carries a
  tested floor (`llama-index-core>=0.11.0`, `dspy>=2.6.0`, `smolagents>=1.6.0`, `crewai>=0.95.0`),
  and CI runs each adapter's smoke test on the newest release and on that floor. The base
  `pip install asqav` still installs no framework, and `all` stays without them.

## [0.10.14] - 2026-10-21

### Fixed

- **The published package pages state the nonce replay bound.** Both language
  READMEs name the HTTP 409 refusal a re-used nonce meets while an earlier live
  receipt holds it, and when the token frees.
- **`sign_batch()` and the offline queue's `sync()` now honour hash-only mode** (criterion 797).
  Both built their request bodies by hand with the raw `context`, so against the Asqav cloud they
  put prompts and tool arguments on the wire (the cloud then refused them with
  `MODE_NOT_ALLOWED`). They now use the same body builder as `sign()`, so the context is hashed
  locally and only the digest and whitelisted labels are sent.

### Added

- **Egress test** (`python/tests/test_no_sensitive_egress.py`,
  `typescript/tests/noSensitiveEgress.test.ts`). Plants a prompt, a file body and a secret-shaped
  value, captures every request at the HTTP transport and fails if any of them is sent. A
  full-payload run is the negative control and must be caught.

## [0.10.13] - 2026-10-16

### Changed

- **Tool results are bound by default** (criterion 787). When `sign()` is given a `result`, the
  client sends its digest as `result_digest` (`sha256:<64 hex>`; dicts and lists are
  JCS-canonicalised first, and anything JCS refuses falls back to `str()`). An explicit
  `result_digest` still wins. Every integration now forwards its completion and error outputs,
  from the extras in both languages down to the core decorators, session errors, phased signing
  and budget records. Previews never throw on BigInt or circular values. The async Python client now
  forwards `result_digest`, `receipt_type`, `action_ref` and `compliance_mode` like the sync
  client. New `attach_result()` (TypeScript: `attachResult`) signs a
  `protectmcp:observation:result_bound` receipt linked to an earlier receipt.

### Added

- **`asqav archive sync`** (criterion 761) pages `GET /audit-pack/archive` into a directory you
  own (`--out`) or into S3 under Object Lock COMPLIANCE retention (`--s3 s3://bucket/prefix`,
  `--retention-days`), using your own credentials. It verifies each page's full-envelope digest
  and signature offline before writing, resumes from a cursor without gaps or duplicates, and
  never rewrites an identical final page.
- **Oracle assessments you can keep and review** (criterion 488): a hash-chained assessment
  store with reassessment labels, an agreement reference that compares seven comparanda, a
  disagreement classifier with four families and a spec-bug evidence bar, and a live compare
  that verifies the receipt bytes itself before classifying. In Python,
  `python -m asqav.verifier.oracle review <store> <receipt>` reviews a live receipt against its
  retained original. Python and TypeScript agree on every case, including non-ASCII terms and
  absent inputs.
- Anchor-entry members are declared once (`ANCHOR_ENTRY_MEMBERS`, draft -09), and the corpus is
  pinned to agree with them (criterion 676).

### Fixed

- Timestamps on the key-status, acceptor and W3C VC adapter paths are checked against one pinned
  grammar first, so the ISO basic form and hour 24 get the same verdict in Python and
  TypeScript (criterion 336).
- Non-contract but decodable signature history verifies under the named historical encoding
  policy; undecodable input is refused instead of reported as a crypto mismatch.
- The org-salt documentation no longer claims a verifying signature it does not have.
- The READMEs no longer show a GitHub star count.

## [0.10.12] - 2026-10-13

### Added

- **Agent-to-agent receipts: acknowledge a peer's receipt.** Agent A signs an action and hands
  B `response.receipt()` (TypeScript: `receiptOf(response)`), usually in the `x-asqav-receipt`
  header via `encode_receipt_header` / `encodeReceiptHeader`. B checks it with the existing
  acceptor (`check_peer_receipt`, `AcceptorMiddleware`) and signs
  `agent.acknowledge(peer_receipt)`: a `protectmcp:acknowledgment` receipt whose
  `counterparty_binding` commits to A's exact payload and signature bytes, so both sides can prove
  what was sent and what was acknowledged. `sign()` takes `counterparty_binding` /
  `counterpartyBinding` directly too, fenced to acknowledgment receipts under compliance mode like
  the cloud. The cloud admits the binding only when A's receipt is visible to B's organization
  (the same org, or a published receipt) and the bytes match.

- **`asqav audit-pack verify --offline <pack>` checks a whole exported Audit Pack with no
  network** (criterion 760). Every receipt runs through `verify_receipt.py`, chained to the
  receipt with the previous `seq` from the same agent, and the bundle digest and signature are
  checked. Each receipt is reported `verified`, `not_verifiable_offline` with the reason, or
  `invalid`, and the pack gets a summary. Trust comes only from files you supply: `--jwks` for
  keys and `--tsa-key` for the TSA's own signing certificate or key (not a root CA). Without `--jwks` the pack's own snapshot is used and the
  best result is `self-consistent, not anchored to a trusted key` with exit 2. TSA entries the
  pack carries are never trusted. A missing bundle signature, an unknown bundle key, an
  unsupported or mismatched bundle algorithm, a bundle key of another organization, a chain gap,
  a fork and a duplicate entry are all invalid. A bundle key revoked or suspended after the
  export leaves the pack not verifiable offline; it is invalid only when the signed window start
  or a receipt's trusted timestamp proves the bundle was signed after the revocation. A pack
  exported today cannot reach exit 0: its OpenTimestamps anchors are pending and it carries no
  TSA certificates.
  Exit 0 needs `--jwks` with everything verified. Anything invalid exits 1; any other result
  exits 2. A lapsed `expires_at` on a receipt signed before it is an info line, not a
  failure. The TypeScript CLI has the same command and trust rules, but it does not check
  timestamp tokens.

- **Witness policy is signed-optional, not required** (criterion 696). A producer-declared
  threshold outside the signature cannot be re-derived by an offline third party, so the wire
  format neither carries nor enforces verifier policy. Decision record (workspace-local):
  `.asondy/eng/DECISION-witness-policy-signed-optional-696.md`.

### Fixed

- **Every Python verifier refuses a parsed dict holding an integer past 2^53.** The
  standalone `run_structured` verified a receipt given as `json.loads` dicts when such an
  integer sat in a member no signature covers, and so did `pack_verify.verify_entry`, which
  calls it. Raw-text ingest already refused it, as does `asqav.verify_receipt_offline`.
  Both now report `unverified` / `unverifiable` on an input axis. `run_structured` also takes
  the whole predecessor envelope that the CLI reads. It checks every member before it uses
  the payload. `pack_verify.verify_pack` reports a pack dict holding one as not verifiable
  offline. `asqav verify-litellm-log` stops crashing with a traceback on an index line holding
  one: the line counts as a failure and the command exits 1. The verdict parity test
  now runs `run_structured` on dicts on every row, and `verify_entry` on every row without
  an origin.
- **The Python oracle returns a verdict for an integer past 2^53.** `run_one` and
  `python -m asqav.verifier.oracle` raised `UnsafeIntegerError` on a receipt or origin holding
  an integer outside ±2^53, where `verify_receipt.py` and the TypeScript `runOne` report
  `unverified` / `unverifiable`. Both now return that verdict, and the oracle CLI exits 2
  instead of crashing with exit 1. `asqav.verify_receipt_offline` gives the same answer for a
  parsed dict holding such an integer anywhere, members no signature covers included, so it no
  longer verifies a receipt that raw-text ingest refuses. `verify_counterparty_binding`
  reports `unresolved` for an origin holding one, as the TypeScript raw-text path does. The
  verdict parity fixtures gain eleven cases with such integers in signed and unsigned positions
  of receipts, predecessors and counterparty envelopes, plus two 2^53 controls, asserted on every
  Python and TypeScript verifier. Conformance vectors are unchanged.
- Outer copies in an exported entry are checked against the signed payload (criterion 742).
  An Audit Pack entry restates signed values (agent, action, time, signature string, key
  and signed bytes) as plain top-level members no signature covers, so editing one still
  read "signature valid". The Python and TypeScript offline verifiers and the standalone
  `verify_receipt.py` now report a new `outer_copies` axis: any copy that disagrees with the
  signed value it restates fails the verdict as `unverified` / `invalid`, and members with no
  signed counterpart, or that sit under a name the receipt never signed, are listed in the note
  as not signed, on a pass too. The copies are weighed against the normalised signature object,
  so a flat `sig` string or `signature_envelope` is checked like the object form. The digest and
  key copies are compared only when the signature verified, `signed_bytes_withheld` is listed and
  never switches a check off, and an observation time that is withheld or unreadable is listed
  instead of failed. Axis count is 13 for the offline APIs and 14 for the standalone verifier.
  A flat hash-mode receipt lists its unsigned members too, including a `payload`, a
  `signature` other than the one verified, and a `key_id` other than the kid of the verifying key.
  The single-entry signature cache in both languages verifies the bytes it holds, and the
  TypeScript one caches `Uint8Array` input only. Conformance vectors `asqav-43` to `asqav-52` pin agreeing entries and edited copies in each
  signature spelling, and `verifier/outer-copy-cases.json` pins byte-identical results and notes
  in both languages. `docs/offline-verification.md` lists which members are signed.

- **One verdict for a signed decimal across the verifiers.** The current receipt profile
  (`v: 1`) admits only integers within ±(2^53 - 1) as digest-covered numbers.
  `verify_receipt.py` and the TypeScript verifier refused an integer-valued float such as
  `500.0` or `-0.0`. The Python oracle and `asqav.verify_receipt_offline` verified it,
  because their parser keeps no lexeme. `profile_range_note` now judges every parsed
  Python float, so all of them report such a receipt `unverified` / `unverifiable`. The
  oracle and the TypeScript verifier also apply the range to the signature object of an
  anchored receipt and to the originating envelope of an `envelope_minus_anchors`
  binding, as `verify_receipt.py` does. A binding under another scope never hashes its
  origin, so no verifier range-checks that origin.
- **`verifyReceiptOffline` accepts raw JSON text** for the receipt, keys, predecessor and
  originating envelope. It reads the text with the lexeme-preserving parser and returns
  the conformance runner's verdict. Given an object from `JSON.parse`, it writes a
  fraction or a value past 2^53 as Python does and refuses a signed `-0` as
  `unverifiable`. A parsed object cannot carry the number's spelling, so a signed `500.0`
  reads back as `500` and that receipt reports `invalid`, while a signed integer respelled
  as an integer-valued float, such as `500.0`, `5e2` or `0.0`, verifies on that path though
  raw text and Python refuse it. Callers who need the strict profile must pass raw text;
  the TypeScript README explains the difference.
- **Behaviour change for Python users.** `asqav.verify_receipt_offline` and the oracle
  now refuse, as `unverified` / `unverifiable`, a receipt they verified in 0.10.11 when a
  signed number is an integer written as a float, such as `500.0` or `-0.0`. A fraction
  such as `0.5` still verifies. A cloud receipt can carry such a number in the
  `metadata` values of a hash-mode receipt, for example a `timestamp` sent as
  `1791676800.0`.
- **Signed parity fixtures.** `verifier/verdict-parity-cases` holds 54 receipts signed with
  a local key over `1e-07`, the 1e16 zone, `-0.0`, `500.0` and `0.5`, respelled and
  tampered, plus signed integers `500` and `0` respelled as `500.0`, `5e2`, `0.0`, `-0`
  and `-0.0`, and signed `-0.0` and `0.0` respelled `-0`. A Python test and a TypeScript
  test assert one outcome per case from every verifier, including `verifyReceiptOffline`
  given `parseJsonPreservingFloats` values, and the counterparty binding label from raw
  text, the float-preserving parser and `JSON.parse`. No conformance vector changes its
  verdict.
- **`verifyReceiptOffline` reads a `-0` in a float-preserving value as Python does.** Given
  `parseJsonPreservingFloats` output, it refused every `-0` as `unverifiable`. That parser
  keeps `-0.0` as a float, so a plain `-0` there is the integer Python reads as `0`, and
  the receipt now gets Python's verdict.
- **TypeScript `canonicalJson` writes a plain fraction as Python does.** 0.10.11 gave the
  Python spelling only to a float-preserving value and left plain `JSON.parse` input as
  JavaScript writes it, so `computeEnvelopeHash` and `verifyCounterpartyBinding` returned
  `mismatch` for a genuine binding over a decimal such as `1e-07` or `0.00001` when the
  origin came from `JSON.parse`. A plain fraction is now written as `1e-07` and `1e-05`.
  Both functions also accept the origin as raw JSON text and read it as Python does, so
  `-0` hashes as `0` and `-0.0` as `-0.0`. A `JSON.parse` origin holding `-0` reports
  `unresolved`, since it could have been either, and `computeEnvelopeHash` and
  `computeCounterpartyBinding` throw `counterparty_origin_unavailable` for it. A whole-valued float such as `500.0` still reads back from
  `JSON.parse` as `500`, so that binding stays `mismatch`, and an origin that signed the
  integer `500` but is respelled `500.0` reports `matches` there while Python reports
  `mismatch`. Pass raw text or a `parseJsonPreservingFloats` value to get Python's answer.
- **Digest change for TypeScript helpers.** For input holding a non-integer number with a
  magnitude from 1e-9 up to 1e-4, these now return a different value, the one the Python
  helpers return: `canonicalJson`, `computeToolFingerprint`, `computeConfigManifestDigest`,
  `computeCveInventoryDigest`, the `action_ref` that compliance mode derives from
  `context`, `deriveChainHash`, `verifyChain` and the CLI chain check, the `chainHash`
  that `verify()` returns, and the receipt doors `canonicalReceiptBytes`, `receiptDigest`,
  `toW3cVc`, `toCloudEvent`, `toOtelGenaiAttributes`, `toC2paAssertion`,
  `toErc8004ValidationRequest` and `wrapAll`. Other numbers give the same bytes as before.
  `canonicalize`, `canonicalizeAction` and `hashAction` are not affected and still write
  such a number the JavaScript way.
- **Values saved from TypeScript 0.10.11 may not recompute.** A counterparty binding that
  0.10.11 computed over an origin holding such a number now reports `mismatch`, and one
  over a `JSON.parse` origin holding `-0` reports `unresolved`. A digest, fingerprint,
  chain hash or door output saved from 0.10.11 over such a number differs from the new
  output, and so does an ERC-8004 `receiptCommitment` recorded from it, on chain or
  elsewhere. Python never matched the 0.10.11 value for a fraction or a `-0.0` origin.
  Recompute these values from the original object.
- **`asqav verify` keeps signature validity apart from anchoring.** A fresh receipt whose
  timestamp anchor is still pending prints `Signature: valid` and `Anchor: pending`, in the
  Python and TypeScript CLIs. A receipt the server rejects prints
  `Signature: invalid (<label>)`. `asqav.verify()` in both languages returns
  `signature_valid` (`signatureValid`) and `anchor` beside `verified`.
- **One API base order everywhere.** `init()`, the async client, the CLI and `asqav doctor`
  resolve the base as: the `base_url` argument, `ASQAV_API_URL`, `ASQAV_API_BASE`, the base
  saved by `asqav login --api-base`, then the hosted default. A base passed to `init()` stays
  in force when a later call in the same process omits it. `asqav doctor` prints the
  base it checks. The Python async client reads the key and base set by `init()`.
- **Network and gateway failures raise Asqav errors.** An HTML or other non-JSON error page
  becomes a one-line `APIError`, and a non-JSON success body raises `AsqavResponseError`.
  Without httpx the server's `detail` and `code` reach the `APIError` message as they do
  with it.
- **`asqav migrate run` and `asqav org halt` send credentials only to the resolved API
  base** in both CLIs, never to the hosted default when another base is configured.
- **`asqav quickstart` prints code that runs.** It prints a first-receipt snippet and the
  command that verifies it. Its help says it writes no files.

### Changed

- **Connection failures raise `AsqavConnectionError`.** A connection that fails or times
  out raises `AsqavConnectionError`, an `APIError` with status 0, in both languages,
  with a one-line message naming the URL. Python code that caught `httpx.RequestError` or
  `urllib.error.URLError` around SDK calls should catch `asqav.AsqavConnectionError`.
- **`asqav verify` exits 1 when the server rejects the receipt.** A pending anchor alone
  still exits 0.
- **One first-receipt snippet.** The READMEs, the CLI reference, `asqav quickstart` and
  `asqav init` share the snippet the dashboard shows: `export ASQAV_API_KEY`, then
  `asqav.init()` and one `agent.sign(...)`.

## [0.10.11] - 2026-10-10

### Added

- **`AsqavClient` (TypeScript) keeps its own connection.** A client keeps the API key and
  the settings it was built with, and every Agent it creates or retrieves uses them. Two clients can run side by side in one process, and the module
  default set by `init` stays independent of both. The README shows credential rotation
  through a new client and Agent.
- **An optional invocation pointer on sign.** `Agent.sign(..., invocation_ref=...)` in
  Python and `invocationRef` in TypeScript put a tool-invocation identifier into the
  signed receipt, so a pre-call and a post-call receipt can be linked. It is a
  producer-asserted pointer and promises nothing about uniqueness. The Python hooks
  forward the tool's invocation id on their own.
- **The originating receipt can be passed to counterparty checks.** The standalone
  verifier takes `--counterparty <file>`, and the Python and TypeScript offline
  verification entry points take the full originating envelope as an option, so the
  counterparty scope check compares bytes locally. The builders in both languages hash
  exactly the payload and the original signature object. An unknown scope never passes.
- **`verify_receipt_offline` in Python returns the same keys as its TypeScript twin:**
  `signer`, `first_failing_edge`, `not_checked`, `coverage` and the resolved `context`.
- **The verification context takes evaluation hints:** `evaluation_time`, `trust_roots`,
  `revocation_snapshot`, `policy_revision`, `evidence_knowledge_cutoff` and `evaluator`
  (`evaluationTime` and so on in TypeScript), all unset by default. `evaluation_time`
  drives the expiry and clock-skew checks, so a receipt can be judged as of a chosen
  moment, and the result records the context it used. With the defaults every verdict is
  unchanged. A verifier adapter that overrides `extra_axes` with two arguments keeps
  working; one that takes `context` by keyword or through `**kwargs` receives it.
- **Conformance vectors** for an Article 50 lifecycle disclosure, an unknown member
  preserved inside the signature, invocation-pointer pre/post linking, counterparty
  bindings, a keyed payload digest, an unregistered receipt namespace and the
  ±(2^53 - 1) integer boundary. The vectors live in the repository, beside the code.

### Changed

- **Colorado framework entry follows the operative act.** `colorado_ai` in
  `asqav.compliance.FRAMEWORKS` and in the TypeScript `compliance frameworks` listing
  reads "Colorado ADMT Act", SB 26-189 (developer duties 6-1-1702(4), deployer duties
  6-1-1703). The key `colorado_ai` is unchanged.
- **`@asqav/sdk` declares the Node versions it runs on:** `>=20.19.0 <21 || >=22.12.0`,
  the release lines on which the CommonJS entry loads. Node 21 does not load it.
- **Python wheels and source archives carry their license files.** `LICENSE` (Elastic
  License 2.0 for the SDK), the Apache-2.0 text for the standalone verifier and a
  `NOTICE` naming which files each covers ship in both, and the package metadata reads
  `LicenseRef-Elastic-License-2.0 AND Apache-2.0`.
- **A receipt type outside the registry reports SKIPPED, not FAIL, on the structure
  check,** naming the type. The verdict stays unverified, so nothing unrecognised
  verifies. Both languages.
- **A key revoked with no published `revoked_at` reports SKIPPED / unverifiable** on the
  key-status check instead of FAIL / invalid, in both languages. A key revoked with a
  timestamp reports as before, and a bad signature still fails.
- **A keyed payload digest (`hmac-sha256`) reports SKIPPED** instead of a digest
  mismatch, since a third party cannot recompute it.
- **The `org_salt` / `orgSalt` documentation states that the hosted API refuses hash-only
  bodies with a caller-computed HMAC** (`keyed_digest_requires_context`). Keyed digests
  are computed server-side: send `context`.
- **Documentation names the checks the offline APIs report,** including the clock-skew
  check, which they do evaluate. The hooks guide states that the Mode B pretool gate is
  deny-by-default and that a gate which cannot start (exit 126 or 127) does not block.
  The verifier README describes both key-resolution paths and notes that the standalone
  tool resolves through the JWK Set. The coverage table cites `letta-client` 1.12.1 and
  classes smolagents as `in_path_proxy`.

### Fixed

- **`asqav hook pretool` accepts a genuine allow receipt.** The gate rebuilt the request
  digest without the lowercased action type and the `_trace_id` context key, so every
  real permit was refused as a digest mismatch. It now reads the digest from the body
  `Agent.sign` sends, so before-hooks and detectors run once. `--dry-run` shows the same
  body.
- **The TypeScript verifier checks the wire version.** A receipt with no `v` member, or
  with `v: 99`, verified in TypeScript and failed in Python. Both verifiers recognise
  `v` 1 and 2; an absent or unsupported `v` makes the receipt unverifiable, on
  payload-mode and hash-mode receipts alike. Booleans and floats never count as a version.
- **Integers beyond ±(2^53 - 1) are refused as unverifiable** by the strict parsers, the
  verifier and the oracle in both languages, and the CLI exits 2 with no traceback.
- **A float spelling of an integer in a digest-covered field is refused** by the strict
  parsers in both languages, so it cannot bind digits that were never validated.
- **An explicit `"size": null` in `payload_digest` is malformed** in both verifiers. An
  absent `size` stays allowed.
- **The TypeScript acceptor middleware parses receipt headers strictly,** as Python does.
  A large integer in a header was rounded by `JSON.parse`, so a peer receipt could be
  accepted under an identifier nobody sent.
- **The anchor block id is derived from the block header.** When the six header fields
  are present, the verifier rebuilds the 80-byte header and requires its hash to match
  the claimed block and meet the target, so a swapped block hash fails.
- **TypeScript Agents stay on their connection.** An Agent keeps the connection selected
  when it was created or retrieved. Calling `init` for another account affects Agents
  obtained afterwards, pending operations keep their destination and signing mode, the
  salt is copied, and a failed `init` leaves the default intact.
- **The TypeScript verifier writes decimals as Python's `repr(float)`.** `1e-7` becomes
  `1e-07`, `0.00001` becomes `1e-05`, `76355072847686510.0` becomes
  `7.635507284768651e+16` and `-0.0` stays `-0.0`. `asqavJcs`, `jcs` and
  `asqavJcsPreCutover` wrote the JavaScript spelling there, so a receipt carrying a
  decimal such as `1e-07` failed its signature check in TypeScript and passed in Python,
  and `counterpartyEnvelopeHash` returned a different hash. The strict JCS output and plain
  numbers from `JSON.parse` are unchanged. A receipt carrying `-0.0` or a decimal such as
  `500.0` is still refused by the TypeScript number-range check while the Python oracle
  accepts it. The Unreleased section aligns the oracle with that check.
- **TypeScript counterparty hashing matches Python for floats.** `computeEnvelopeHash` and
  `verifyCounterpartyBinding` hash a float-preserving value as Python's `json.dumps` writes
  it (`1.0` stays `1.0`). A float Python refuses raises `canonical bytes unavailable` and
  the binding reports `unresolved`.
- **TypeScript `canonicalJson` writes a float-preserving value as Python does,** for
  example `{"v":1e-07}`, where it wrote an object with `lexeme` and `value` members. Plain
  `JSON.parse` input is unchanged.
- **The TypeScript nesting-depth cap treats a decimal as a leaf,** so the same bytes pass
  at 198 levels in both languages, and a decimal in a counterparty envelope passes at 197
  wrappers as in Python.
- **The TypeScript CommonJS build defines `RawFloat` and `Agent` once.** Each CommonJS
  entry bundled its own copy, so a decimal parsed through `@asqav/sdk/verifier` failed the
  `instanceof` check inside `require('@asqav/sdk')`, and `instanceof Agent` failed across
  entries. CommonJS and ESM now give the same verdicts; the ESM files are unchanged.

### Security

- **`asqav hook pretool` only opens for a key published for the gated agent.** The gate
  verified a permit against the whole key directory, so a receipt naming the gated agent
  but signed by a sibling agent's key, another organisation's key, or a key picked by
  `kid` alone was accepted. It now keeps only the directory entries whose `agent_id`
  equals `ASQAV_AGENT_ID`; an entry with no `agent_id` verifies nothing.
- **`asqav hook pretool` binds the receipt to the call.** The receipt must name the same
  agent and action digest, fall inside the gate's deadline, and carry the event's
  `tool_use_id` as `invocation_ref`. An event with no `tool_use_id` is blocked, a receipt
  is refused whenever the gate sent no digest (so `--clear-context` always blocks), and
  any error in the gate, including the `cli` extra being absent or an event nested too
  deep to parse, exits 2 so the call is blocked.
- **Python lock pins moved past open advisories:** pyjwt 2.13.0 to 2.15.0, anyio 4.12.1
  to 4.14.2, urllib3 2.7.0 to 2.8.0, fsspec 2026.2.0 to 2026.6.0, multidict 6.7.1 to
  6.9.1 and litellm 1.88.0 to 1.88.6. The `asqav` package declares no runtime
  dependencies; these arrive only through optional extras (pyjwt through `openai-agents`
  and `strands`; fsspec and litellm through `litellm`; multidict through `litellm` and
  `instructor`; anyio through most extras, including `httpx` and `cli`; urllib3 through
  `langchain`, `litellm`, `haystack`, `instructor`, `openai-agents`, `presidio` and
  `strands`). If you install one of those extras, upgrade these packages in your own
  environment.
- **TypeScript development dependencies updated:** source-map-js 1.2.2, nanoid 3.3.18 and
  vitest 4.1.11. None of them ship in the published package.

## [0.10.10] - 2026-09-05

### Changed

- **The package pages are shorter and lead with what a reader can check themselves.**
  `python/README.md` (rendered on PyPI) goes from 545 lines to 200 and
  `typescript/README.md` (rendered on npm) from 560 to 214, mirroring each other
  section for section. Reference material moves to a linked table pointing at
  `asqav.com/docs`; every link was checked to resolve.

### Fixed

- **The documented verification example no longer fails when you run it.** The pages
  told readers to assert `verified` on the public example id
  `sig_example_regulator_cold_verify_2026`. That receipt is a shape example whose
  signature bytes are placeholders and whose `kid` resolves to no key, so the API
  reports `verified: false` by design and the snippet raised `AssertionError` for
  anyone who pasted it. Both languages now print the real value and explain that a
  verifier reporting `false` on absent evidence is the intended behaviour.
- **The install line beside the `asqav verify` example names the extra that provides
  the command.** It said `asqav[verify]`, which carries the offline crypto; the
  command comes from `asqav[cli]`. Following the page left the reader without the
  binary it then told them to run.
- **Two links that resolved only on GitHub.** `../docs/offline-verification.md` is
  dead on PyPI and npm, which render the README standalone; both now point at the
  live docs page.

## [0.10.9] - 2026-09-05

### Added

- **The full verifier verifies Ed25519 and ES256 signatures natively**, beside
  ML-DSA-65, through the optional `cryptography` dependency it already declares;
  the standalone file still imports nothing from the package. The published
  requirement map lists the signature requirement as exercised by every
  asqav-native vector rather than by the four post-quantum ones. (#477)
- **Every result carries a `coverage` block in the shape the reviewer's public
  tool emits** (`stopped_at`, `checks_not_evaluated[{id, reason, status}]`),
  derived from the same table as `not_checked`, so the two tools' declarations
  read side by side. Both languages. (#475)
- **Per-vector anchor material.** A vector directory may ship `tsa_trust.pem`
  and `bitcoin_headers.json`; the requirement-map builder passes them through.
  asqav-24 ships both, so the anchoring requirement is exercised offline and
  `unmapped_requirements` is empty. (#478)
- **A machine-readable registry of outside recomputations**
  (`verifier/independent-runs.json`), each pinned to a commit and the vectors it
  re-derived, with a test that every pinned commit and vector exists. (#474)
- **Vectors** for the anchor `status` and `anchor_block_hash` members (#468),
  for receipts carrying both `context` and `payload_digest` (#465), and for the
  counterparty binding's profile `payload_digest` shape (#473).
- **A scheduled artifact probe** runs the published-package probe weekly, on
  dispatch, and after every successful publish, uploading its output as a job
  artifact. (#476)
- The TypeScript verifier declares its non-coverage (`notChecked`) on every
  result, in parity with Python. (#464)

### Fixed

- **The OpenTimestamps attestation parser reads the reference framing.** An
  attestation is tag + varbytes(payload), with the block height a little-endian
  (LEB128) varuint inside the payload. The production proof in asqav-24 lands in
  block 965451 and parses to its last byte. (#478)
- **The ACTA -03 chain-link form verifies alongside the -02 one.** (#469)
- **Timestamp-authority tokens signed under the bare `rsaEncryption` algorithm
  identifier verify.** (#470)
- **The requirement-map builder hands the verifier the predecessor payload**, so
  the chain axes in the published map reflect a real run. (#471)
- A PEM file with several certificate blocks yields one trusted TSA key per
  block. (#478)
- The artifact probe reports PyPI simple-index lag as a sentence rather than a
  traceback. (#466)

### Changed

- The TypeScript README states the verifier's shipped verdict vocabulary (#467);
  every README states the three plans and their real limits (#472).

## [0.10.8] - 2026-09-04

### Added

- **Non-coverage is declared in the output, on every result including passing
  ones.** Each verification result carries a machine-readable `not_checked` array
  naming every check the tool does not perform, with the requirement it belongs
  to, why it is skipped, and the input that would enable it where the caller can
  close the gap. A verifier that reports only what it checked lets a reader
  mistake silence for coverage, and a README does not travel with the result.
- **A published vector-to-requirement mapping**, with the requirements no vector
  exercises published beside it (`conformance-vectors/requirement-map.json`).
  Coverage is derived by running each vector and reading the axes off the result,
  so an axis the verifier skipped is not counted whatever a vector's notes claim.
  It reports one gap on generation: no vector exercises the anchoring requirement,
  because the corpus ships no pinned TSA key material.
- **A probe that checks the published packages**, not the working tree, for every
  entry point their own pages document, and runs the conformance corpus through
  the installed package (`verifier/artifact_probe/`).

## [0.10.7] - 2026-09-03

### Changed

- **A whole-number value outside the canonical range is refused rather than
  reshaped.** The two halves canonicalized such a value differently: Python kept
  every digit, JavaScript rounded to the nearest representable double, and the
  disagreement was pinned as an expected divergence. A receipt carrying one
  therefore produced two different canonical byte strings and two different
  digests, so its signature verified in one implementation and failed in the
  other. Both halves now refuse the value where the document is parsed, beside
  the duplicate-member door, and both canonicalizers refuse it again for values
  assembled in process. The check reads the digits as written, never the parsed
  double, so a value that arrives already rounded cannot pass as exact. The bound
  is the closed range plus or minus two to the fifty-third, one above the
  JavaScript safe-integer limit, because that endpoint is exactly representable,
  both languages print it identically, and the upstream interop corpus pins it as
  canonical. **Callers who send a nanosecond timestamp or a Snowflake-style
  identifier as a bare number will now see a refusal at the parse boundary; send
  the value as a JSON string instead.**
- The README section covering `protectmcp:observation` and
  `capture_topology=passive_telemetry` is titled for that wire vocabulary, so the
  heading names the members the section documents.

### Fixed

- **Both READMEs told readers a replay control does not exist when it does.** The
  `nonce` entry stated the cloud "runs no uniqueness check" and that "the field
  gates no replay". The platform refuses a nonce a live signature for the same
  agent in the same organisation already holds and answers HTTP 409. The text now
  states what the field bounds and when the token frees, in both language halves
  and in the four source docstrings that repeated it.
- **The published counterparty digest was unreproducible in five vectors.** The
  value on the wire was taken over the three-key object while the chain seed was
  still in its prefixed form; the change that moved the seed to bare hex
  regenerated the canonical bytes and their digest and left the derived value
  behind, so it matched neither scope and an outside implementer could not
  reproduce it under any reading. It is re-pinned to the envelope-minus-anchors
  scope, and every binding that carries one declares `scope` on the wire so the
  value is read under the scope it was taken over. Nothing caught this because
  both tests naming the member built their expectation by calling the production
  helper against a locally built envelope, so neither read the published file; a
  corpus integrity gate now checks the published inputs against their own
  canonical forms and digests.

### Added

- Worked forms for the canonical-range rule in the conformance corpus, so an
  implementer can copy what a conformant document looks like instead of reading
  only a prohibition: the document that must be refused, carried as text because
  a document that never parses has no canonical form to pin; the same numeric
  value carried the conformant way, as a JSON string; and the boundary value
  itself pinned as accepted. A corpus check requires each vector to carry exactly
  one of a parsed input or a text input, and requires a text-input vector to pin
  a refusal and publish no canonical form. Tests assert the published document as
  a literal and require the shipped parser to refuse it, in both languages, so
  the corpus cannot advertise a rule the code does not implement.

## [0.10.6] - 2026-09-02

### Fixed

- **The anchors axis called genuine receipts invalid.** It hashed every top-level
  member of an Audit Pack entry, while the signer commits
  `sha256(JCS({payload, signature}))` over the wire signature string. An untouched
  production receipt therefore verified as `unverified` with
  `failure_class: invalid`, the class reserved for a binding proven broken, and
  exited 1. The envelope is now projected to exactly the three canonical members
  and the anchored bytes are the two committed ones. Where an export carries the
  signature in the other base64 alphabet, the commitment is checked against the
  alphabet the signer used and the report says which one matched; that tolerance
  is a migration measure and never launders a tamper, because a changed payload
  fails under both alphabets. Pinned against a real production receipt.
- **The gate's cache predicate ignored a key rotation.** It tested the agent bind
  before the signed key thumbprint, so a stale cache answered "present" after a
  rotation and the gate blocked every tool call for up to a day.
- **Receipt verification could outlive the gate's deadline.** The verification and
  any key-set refresh now run inside the same budget, so the gate answers before
  the harness timeout it documents.
- **A deny or rate_limit decision read as `signed` and exited 0.** It now blocks
  with the decision and a reason on stderr.
- **A shared key thumbprint resolved by row order.** It is broken by the signed
  agent id and then the envelope kid, so a receipt naming a revoked duplicate
  still reaches the key-status axis with that row.
- **The agent-bind fallback took the first row rather than the right one.** It now
  keeps the candidate whose signature verifies, leaving revocation to the
  key-status axis where it belongs.
- **The two languages disagreed on three inputs.** The thumbprint pattern uses
  fullmatch, an empty identifier counts as missing on both sides, and a lone
  surrogate is no longer read as a supplementary member name.

### Changed

- `JCS_UTF16_CUTOVER` is pinned to `2026-09-02T18:05:09+00:00`, the measured
  instant the platform deploy carrying the RFC 8785 emitter went live.
- Offline verification is documented from the Audit Pack export. The hosted
  verify response is a display projection that omits identifier members of the
  signed payload, so a receipt saved from it can never reproduce the signature.
- The differential fuzzer prints the engine roster it compared, because engines
  that are absent cannot disagree and a degraded roster otherwise reads as green.
- The JWK Set cache trust boundary is written down in the hooks guide, and every
  quotation of the harness documentation there is verbatim with its source and
  access date.


### Changed

- **Harness hooks send a digest by default.** `asqav hook pretool` and `asqav hook
  posttool` now sign in hash-only mode whatever host they talk to: the tool
  arguments are canonicalised and hashed locally and never leave the machine.
  `--clear-context` is the opt-in that sends them for the platform to hold, and
  `--dry-run` prints the exact body either way. Against the Asqav cloud the SDK
  already resolved to hash-only; self-hosted deployments sent the arguments in
  clear, and a dry run showed clear context regardless of host.
- **The gate has a deadline.** `asqav hook pretool` waits
  `ASQAV_HOOK_DEADLINE_SECONDS` (default 5) for the signer and then blocks with
  exit 2; the same value is the SDK's per-request timeout (`init(timeout=...)`
  is new). A hung signer can no longer ride the harness's 600-second default and
  let the tool run unsigned. The posttool hook proceeds unsigned on the same
  deadline.
- **The gate verifies the receipt it is handed.** Before exit 0 the pretool
  hook runs the returned receipt through the standalone verifier against the
  platform's published JWK Set (cached 24 hours at `~/.asqav/jwks-cache.json`);
  a receipt that does not verify blocks the tool call.
- The pretool hook prints `signed <id>` for the observation the cloud records
  for in-process capture and reserves `permit <id>` for a wire decision of
  `allow`; `hooks/README.md` states what the cloud records for an in-process
  gate today.

### Fixed

- **Signing-key resolution on the standalone verifier.** `verify_receipt.py`'s
  `run()` and `run_structured()` resolved the envelope `kid` through the loose
  issuer match, which on a multi-key organisation returns whichever sibling is
  listed first, and only fell back to the agent bind after a signature failure.
  A receipt whose signature could not be checked at all (an algorithm outside the
  profile) therefore reported the sibling's revocation status and a
  `key_substituted` binding failure that were both artefacts of the wrong entry.
  Every path now resolves through one matcher: the signed `key_thumbprint` first
  (it names the exact key even after a rotation leaves an agent with two published
  keys), then the exact key id, then the agent bind, then the bare-kid issuer
  match. The TypeScript verifier and the Python oracle adapter take the same
  order. A thumbprint naming a key the signature was not made with is still
  caught: the signature fails against it and `key_binding` reports the
  substitution.



- **RFC 8785 member order on the two verification paths that still sorted by code
  point.** The standalone `verify_receipt.py` (`canonical_json`, the bytes it checks
  every signature, chain link and carried context against) and the TypeScript
  verifier's asqav dialect (`asqavJcs`, used by the asqav-native adapter) now order
  member names by UTF-16 code unit, matching the SDK emitters and the platform. The
  two agree with the emitter across the Basic Multilingual Plane and diverged for any
  member name above U+FFFF, which no production receipt has ever carried.
- **Dated dialect cutover.** `JCS_UTF16_CUTOVER` (Python and TypeScript) pins the
  instant the platform switched. A receipt issued before it whose member names reach
  above U+FFFF and whose signature verifies only under the old code-point order is
  reported as the pre-cutover dialect on the signature axis, never as verified; a
  receipt issued after it gets no retry.
- `verifier/differential_fuzz.py` now compares the standalone verifier and the
  TypeScript verifier bundle as engines, so the fuzz gate covers the bytes the
  verifiers check rather than only the bytes the emitters produce.
- `docs/fingerprint-spec.md`, `doors.py` and `demo.py` no longer describe or use
  code-point key order.

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
  expiry axis, which never folds the verdict. A sibling
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
  failure classes.** Every public verifier surface — the
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

- **BREAKING: duplicate JSON member names are rejected at any depth.**
  Every receipt- and record-parsing path — `verify_receipt` ingest, the
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
