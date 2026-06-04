# Oracle conformance vectors

Test vectors for the multi-format verifier in `../oracle/`. The layout is the
AERF dir-per-vector convention used as a superset: one directory per vector, a
top-level `manifest.json`, and per-vector key material plus an `expected.json`.

## Layout

```
manifest.json                  array of {dir, format, outcome, reason_code, notes}
<vector-dir>/
  receipt.json                 the receipt under test
  expected.json                {format, outcome, reason_code, notes}
  predecessor.json             (chain vectors only) the prior receipt
  jwks.json                    (asqav-native) issuer key directory
  keys.json                    (aerf) {key_id: ed25519_pubkey_hex}
  acta-keys.json               (acta) JWKS-shaped key set
  did_map.json                 (agentreceipts) {did: ed25519_pubkey_hex} for did:agent / did:web
```

`outcome` is `PASS` or `FAIL`; the runner maps it to the oracle verdict and
asserts equality. `reason_code` follows the shared taxonomy (`issuer_signature`,
`chain`, `schema`, `key`).

## Run

```bash
python -m oracle.runner          # from the verifier/ directory
```

## Vectors

- `asqav-01..03` - valid Asqav-native receipts (genesis permit, genesis deny,
  chain link), Ed25519-signed so the signature axis verifies without the
  post-quantum dependency.
- `asqav-04-tamper-sig` - decision flipped after signing; signature fails.
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

The keys are generated from fixed seeds so the vectors are reproducible. AERF
public keys are derived per spec as the first 16 hex of `SHA-256(pubkey)`.
AgentReceipt did:key receipts need no key file - the resolver decodes the key
from the `did:key` identifier; did:agent / did:web vectors carry `did_map.json`.
Vector provenance, the upstream commit SHAs, and the re-signing of the
agent-receipts PASS vectors with the upstream keypair are recorded in `UPSTREAM.md`.
