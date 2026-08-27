# Oracle conformance vectors

Test vectors for the multi-format verifier in `../oracle/`. The layout is the
AERF dir-per-vector convention used as a superset: one directory per vector, a
top-level `manifest.json`, and per-vector key material plus an `expected.json`.

## Layout

```
manifest.json                  array of {dir, format, outcome, failure_class, reason_code, notes}
<vector-dir>/
  receipt.json                 the receipt under test
  expected.json                {format, outcome, failure_class, reason_code, notes}
  predecessor.json             (chain vectors only) the prior receipt
  jwks.json                    (asqav-native) issuer key directory
  keys.json                    (aerf) {key_id: ed25519_pubkey_hex}
  acta-keys.json               (acta) JWKS-shaped key set
  did_map.json                 (agentreceipts, w3c-vc) {did: ed25519_pubkey_hex} or
                               {did: did_document} for did:agent / did:web
```

`outcome` speaks the shipped verdict vocabulary - `verified`, `verified_keyed`,
or `unverified` - and the runner asserts the oracle verdict equals it. An
`unverified` entry also pins `failure_class` (`invalid` or `unverifiable`), so
the two are never collapsed and both languages agree byte for byte on each.
`unverified`/`unverifiable` carries a vector whose signature axis cannot be
checked in the CI environment (for example a real ML-DSA-65 prod receipt when
`dilithium-py` is absent), pinning that the verifier downgrades rather than
emitting a false `verified`. `reason_code` follows the shared taxonomy
(`issuer_signature`, `chain`, `schema`, `key`, `signature_skipped_no_dilithium`,
`duplicate_member`).

Receipts are parsed with duplicate-member rejection (criterion 419): a receipt
that repeats a JSON member name at any depth is a terminal parse failure,
reported `unverified`/`unverifiable` before any hashing. `asqav-11` and
`asqav-12` pin the top-level and nested cases.

## Run

```bash
python -m oracle.runner          # from the verifier/ directory
```

## Vectors

- `asqav-01..03` - valid Asqav-native receipts (genesis permit, genesis deny,
  chain link), Ed25519-signed so the signature axis verifies without the
  post-quantum dependency.
- `asqav-04-tamper-sig` - decision flipped after signing; signature fails.
- `asqav-05-hash-mode-prod` - a real default-mode prod `/sign` receipt (ML-DSA-65).
  The reconstructed signing input byte-matches the prod-signed message and the
  signature verifies with `dilithium-py`; absent that optional dep the signature
  axis SKIPs, so the outcome is `unverified`/`unverifiable`. Guards the
  production hash-mode path against a false `verified` on the post-quantum
  signature the CI base cannot check.
- `asqav-06-mldsa65-payload-prod` - a real payload-mode prod ML-DSA-65 receipt;
  its signed `expires_at` lapsed, so the expiry axis FAILs alone while the
  verdict stays verified (criterion 426).
- `asqav-07-revoked-key` - a valid signature from a key whose JWKS status is
  `revoked`; the key_status axis FAILs the verdict (parity vector).
- `asqav-08-v2-signer-canary` / `asqav-09-v2-signer-tampered` - a v:2 receipt
  whose in-signed-body `signer` the neutral verifier surfaces, and its tampered
  twin proving `signer` sits inside the signed coverage (parity vectors).
- `asqav-10-hash-mode-multikey` - hash-mode receipt signed by the last of three
  sibling keys sharing issuer/org ids; the agent bind resolves the actual signer
  (anti-vacuous parity vector).
- `asqav-11-dup-member-toplevel` / `asqav-13-dup-member-nested` - the valid
  genesis receipt with a duplicated JSON member name at the top level and two
  levels down. Both are terminal parse failures (criterion 419): the strict
  parser rejects them before any hashing, so they never verify.
- `asqav-12-time-edge-expiry` - deterministic ML-DSA-65 vector with an extreme
  positive UTC offset around midnight and a lapsed signed `expires_at`
  (time-edge conformance, criterion 422; parity vector).
- `aerf-01..02` - valid AERF receipts (genesis, chain link). Genesis omits
  `previous_receipt_hash`; the chain hash excludes the signature, per the AERF
  spec.
- `aerf-03-tamper-evidence` - action mutated after signing; signature fails.
- `aerf-04-tamper-chain` - `previous_receipt_hash` mutated; chain fails.
- `acta-01..05` - ACTA genesis, chain link, tampered signature, and an
  unsupported commitment-mode receipt that fails the baseline verifier.
- `aerf-up-*` - upstream-derived AERF vectors (see `UPSTREAM.md`).
- `agentreceipts-01..06` - W3C-VC AgentReceipt: did:key genesis (key resolves
  inline), chain link, tampered payload, tampered proofValue, a genesis missing
  `previous_receipt_hash` (malformed), and a wrong-DID signature mismatch.
- `agentreceipts-up-*` - upstream agent-receipts vectors (six malformed
  single-field mutations, a tampered chain, and two upstream-keypair PASS
  cases). See `UPSTREAM.md`.
- `authproof-01-genesis-real-sdk` - a receipt minted by the real Authproof JS
  SDK (ES256 over insertion-order `JSON.stringify`, embedded P-256 JWK); a
  cross-implementation interop PASS. `authproof-02/03` are its forged-signature
  and tampered-scope negatives. The signer key is embedded, so no key file. See
  `UPSTREAM.md`.
- `pipelock-ev2-01-proxy-decision` / `pipelock-ev2-02-tamper-payload` - a valid
  Pipelock evidence-v2 receipt and its verdict-flipped twin (keys.json carries
  the signer key id -> raw Ed25519 hex).
- `w3c-vc-01-didweb-happy-path` - a W3C VC 2.0 credential with a
  DataIntegrityProof `eddsa-jcs-2022` signature (W3C TR vc-di-eddsa): Ed25519
  over `SHA-256(JCS(proofOptions)) || SHA-256(JCS(unsecuredDocument))`. The
  did:web issuer resolves from the injected DID document (offline mode).
- `w3c-vc-02-tamper-subject` / `w3c-vc-03-tamper-proofvalue` - credentialSubject
  flipped and one proofValue base58 character replaced after signing; both fail
  the signature axis.
- `w3c-vc-04-wrong-key-injected` - the injected DID document publishes an
  unrelated key; verification fails against the published key, never a false pass.
- `w3c-vc-05-no-did-document` - no injected DID document and the oracle never
  fetches, so the signature axis SKIPs: `unverified`/`unverifiable`, fail closed.
- `w3c-vc-06-expired` - signature verifies but signed `validUntil` lapsed; the
  expiry axis FAILs alone while the verdict stays verified (criterion 426).
- `w3c-vc-07-dup-member` - `credentialSubject` appears twice; strict ingest
  rejects the duplicate at parse time (criterion 419).
- `w3c-vc-08-didkey-happy-path` - a did:key issuer self-resolves inline from
  the multicodec frame; no key file needed.

The keys are generated from fixed seeds so the vectors are reproducible. AERF
public keys are derived per spec as the first 16 hex of `SHA-256(pubkey)`.
AgentReceipt did:key receipts need no key file - the resolver decodes the key
from the `did:key` identifier; did:agent / did:web vectors carry `did_map.json`.
Vector provenance, the upstream commit SHAs, and the re-signing of the
agent-receipts PASS vectors with the upstream keypair are recorded in `UPSTREAM.md`.

## Corpus freeze at v1 (criterion 420)

This corpus is frozen at version 1. `manifest.lock.json` pins every file in
the tree by SHA-256 and byte length, and carries the digest of the lock
itself; any drift fails CI.

- Lock digest: pinned in the lock's own `digest` field and reproduced in the
  repo as the `VERIFIER_LOCK_DIGEST` constant in
  `python/tests/test_corpus_lock.py`; a regenerated lock must update both.
- Verification path A: `python/tests/test_corpus_lock.py` re-derives every
  pin with `hashlib` + `pathlib`.
- Verification path B: `verifier/check_corpus_lock.sh` re-derives every pin
  through the `sha256sum` binary and `wc`.
- Regeneration after an intentional edit: `python verifier/freeze_corpus_lock.py`.

### Published signing seeds

Locally regenerated vectors sign with the seeds published in the lock's
`signing` section, so their signature bytes reproduce exactly:

- `ed25519_acta` - the ACTA generator seed (mirrors `gen_acta_vectors.py`).
  Ed25519 is deterministic per RFC 8032, so the seed reproduces every ACTA
  signature byte for byte.
- `mldsa65` - one corpus-wide ML-DSA-65 seed (seed hex and public key in the
  lock), used by `asqav-12-time-edge-expiry`. Signature-byte parity uses
  `dilithium-py` `ML_DSA_65.sign(sk, m, deterministic=True)`, the FIPS 204
  pure variant; CI re-derives the key and the pinned KAT signature from the
  seed on every run.

Upstream and prod-signed receipts (the `*-up-*` vectors, `authproof-01`,
`asqav-05/06`) carry public keys only - their producers' seeds are not ours
to publish. Their signature bytes are pinned by the per-file SHA-256 entries
above, and verification of them is deterministic, so conformance does not
depend on regeneration. Production ML-DSA-65 signing randomises (FIPS 204
hedged variant); verification stays deterministic, which is what audits.
