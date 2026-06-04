# Upstream-derived conformance vectors

The `aerf-up-*` vector directories are derived from the AERF conformance
corpus at <https://github.com/aerf-spec/aerf>, cloned at commit
`59fce60fe30bde35318812f502a55ab4bace4650`.

The `agentreceipts-up-*` vector directories and the vendored interop files
under `agentreceipts-upstream-interop/` are derived from the agent-receipts
corpus at <https://github.com/agent-receipts/ar>, cloned at commit
`16772507e6d00b1dfea6f079ea0e0980d324993f`. The repository code (including
`cross-sdk-tests/`) is licensed Apache-2.0 and the protocol spec under `spec/`
(including the did:key vectors) is licensed MIT.

`agentreceipts-upstream-interop/canonicalization_vectors.json` is reproduced
verbatim from `cross-sdk-tests/canonicalization_vectors.json`; our
`jcs_rfc8785` canonicaliser must produce byte-identical output for every one of
its 32 `canonicalization_vectors` entries, asserted directly in the oracle test
suite. `agentreceipts-upstream-interop/did_key_vectors.json` is reproduced from
`spec/test-vectors/did-key/vectors.json`; the shared DID resolver decodes each
to the expected raw Ed25519 key.

The `agentreceipts-up-*` FAIL directories reproduce the six single-field
mutations from `cross-sdk-tests/malformed_vectors.json` (receipts every SDK MUST
reject) plus the tampered middle-chain receipt. The valid `agentreceipts-up-00`
and `-up-08` directories re-sign that corpus's base receipt with the corpus's
own Ed25519 keypair over `jcs_rfc8785` bytes, so the PASS vectors carry a real
upstream-keypair signature rather than a self-authored one.

The agent-receipts spec example receipts under `spec/examples/` are NOT ingested:
they use `did:agent:` identifiers with no published key and carry placeholder
`previous_receipt_hash` values, so they are documentation illustrations rather
than cryptographically verifiable vectors.

INTEROP FINDING (canonicalization dialect divergence). Both AERF SPEC.md and the
agent-receipts spec cite "full RFC 8785 (JCS)", but their reference producers
disagree on number serialisation and key ordering. The AERF Go canonicaliser
(`verifiers/go/internal/aerf/canonical.go`) writes numbers via `json.Number`
verbatim and sorts keys with `sort.Strings` (UTF-8 byte order), so an AERF
receipt carrying `1250.0` is signed over the bytes `1250.0`. The agent-receipts
producers sign true RFC 8785, where the same value canonicalises to `1250`. The
two are not byte-compatible, so the oracle keeps two canonicalisers: `jcs` (the
AERF/ACTA dialect, numbers as Python emits them) and `jcs_rfc8785` (strict RFC
8785, used only by the agent-receipts adapter). This was verified directly:
the upstream AERF signature on `vectors/01-genesis-happy-path` verifies under the
verbatim-number canonicaliser and fails under the strict one, while all 32
agent-receipts canonicalization vectors byte-match only under the strict one.

The upstream code and vector data are licensed Apache-2.0; the upstream
prose is licensed CC-BY. The receipt artifacts and public keys are
reproduced from that corpus. Each upstream public key (SPKI PEM) is
translated into our `keys.json` key map as raw Ed25519 hex, keyed by the
upstream `key_id` (the leading 16 hex characters of the SHA-256 of the
public key).

## Authproof

The `authproof-*` vectors target the shipping Authproof JS SDK
(<https://github.com/Commonguy25/authproof-sdk>, `src/authproof.js`), which is
authoritative for the wire shape: ES256 (ECDSA P-256 / SHA-256) over
insertion-order `JSON.stringify` of the receipt minus its `signature`, signature
hex-encoded as raw `r||s` (64 bytes), signer key embedded as a P-256 JWK at
`signerPublicKey`. `authproof-01-genesis-real-sdk` is a receipt minted by that
SDK and is a one-directional interop PASS (we verify a real Authproof artifact);
`authproof-02/03` are its self-authored negatives. No published portable vector
corpus exists upstream, so bidirectional interop against a third-party verifier
of our output is PENDING.

Two upstream variants deliberately diverge and are NOT targeted: the bundled
draft-nelson text specifies a base64url signature, a content-hash
`delegationId`, and canonical JSON; the repo's separate Python SDK signs DER over
sorted-key JSON with snake_case fields. A receipt from either would not verify
under the JS-SDK rules, and the JS SDK is the published product.

## Outcome mapping

Each upstream vector declares an outcome of `PASS`, `FAIL`, or
`KNOWN_LIMIT`. Our `expected.json` carries the verdict our verifier
produces for the receipt bytes:

- `PASS` and `FAIL` map directly to our `PASS` / `FAIL` verdicts.
- `KNOWN_LIMIT` maps to `PASS`. An upstream `KNOWN_LIMIT` is a residual the
  receipt layer cannot close, where a conformant verifier exits success
  because every signature genuinely verifies. Our verifier reaches the same
  success on the bytes. The mapping is the honest outcome the bytes
  produce, not a relabel; the per-vector `notes` record why the residual
  sits outside what any receipt verifier can detect.

## Multi-signer axes

Our AERF adapter verifies the issuer signature and the chain link, and also
the conditionally-required parent counter-signature and the policy-decision
binding signature (both plain Ed25519 over canonical bytes). Vectors that
exercise a missing-but-required counter-signature or a policy verdict bound to
the wrong context reach `FAIL` for the same reason the upstream verifier records.

The two upstream transparency-log inclusion vectors are NOT ingested here. Their
inclusion-proof verification is deferred to the shared inclusion-proof checker
that lands with the Asqav transparency log, which reuses a vetted library rather
than an adapter-local proof walk; this corpus ingests the other ten upstream
vectors now.

## Chain vectors

The upstream multi-receipt chain vectors ship a `receipts/` directory. Our
runner verifies one receipt against one predecessor, so each chain vector is
ingested as the load-bearing receipt plus its immediate predecessor:

- the happy-path chain ingests the final link against its predecessor;
- the tamper-in-chain vector ingests the mutated middle receipt against its
  valid predecessor, where the mutated receipt's own issuer signature is
  what fails.
