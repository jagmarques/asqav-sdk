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

The keys are generated from fixed seeds so the vectors are reproducible. AERF
public keys are derived per spec as the first 16 hex of `SHA-256(pubkey)`.
