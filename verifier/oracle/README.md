# Oracle - universal neutral verifier

Verify agent receipts across formats behind one seam. The oracle extends the
standalone `../verify_receipt.py` to verify rival and sibling receipt formats,
not only Asqav-native receipts.

It proves only what the bytes prove: a valid signature over the canonical bytes,
a reproducible hash-chain link, and structural presence at time T. It never
attests the behaviour or correctness of the recorded action.

## What is shared, what is per-format

Shared infrastructure (one implementation, every format reuses it):

- signature verify dispatch - Ed25519 (via `cryptography`) and ML-DSA-65 (via the
  `dilithium-py` path the standalone tool already uses). An absent dependency
  SKIPs that algorithm with a clear status; it never silently passes.
- JCS / RFC 8785 canonicalisation - `canonical.jcs` (with NFC) and
  `canonical.asqav_jcs` (the cloud's historical form, byte-identical to the
  signer).
- SHA-256 chain walk and the verdict aggregation in `core.verify`.
- the conformance-vector runner in `runner.py`.

Per-format behaviour lives behind the six-method `FormatAdapter` seam:

| method | purpose |
|---|---|
| `detect(doc)` | cheap structural fingerprint, no crypto |
| `extract_signature(doc)` | signature bytes, alg, kid |
| `resolve_key(doc, provider)` | public key from a JWKS / key map / embedded key |
| `signing_input(doc)` | the exact bytes the signature covers (strip + canonical + optional pre-hash) |
| `chain_step(doc)` | previous-hash field, genesis sentinel, and how to recompute the link |
| `schema(doc)` | structural check |

`signing_input` is the most error-prone divergence between formats and is encoded
explicitly per adapter: Asqav-native signs the canonical payload directly;
AERF signs the JCS payload directly after stripping `signature`, `timestamp`,
`parent_signature`, `parent_key_id`, `log_inclusion_proof`.

## Adapters this slice

- **asqav-native** - the `verify_receipt` signature, chain, and structure-
  presence axes behind the seam. ML-DSA-65 over the canonical payload; chain via
  the all-zero genesis seed. Scope boundary: as a standalone offline verifier the
  oracle runs only those three axes; it does NOT run the standalone
  `verify_receipt` clock-skew (`issued_at`) or anchor-liveness axes, which need
  wall-clock freshness and network the oracle deliberately omits. A receipt the
  oracle PASSes can still fail those two liveness axes at ingestion time.
- **aerf** - Ed25519 over RFC 8785 JCS, signed directly. Genesis OMITS
  `previous_receipt_hash`; the chain hash EXCLUDES the signature, per the AERF
  spec (the agentmint reference producer includes it; this adapter targets the
  spec).
- **authproof** - ES256 (ECDSA P-256 / SHA-256) over insertion-order
  `JSON.stringify` of the receipt minus its `signature`, hex raw r||s, signer key
  embedded as a P-256 JWK. Targets the shipping JS SDK; the bundled draft and the
  repo's Python SDK both diverge (see `conformance-vectors/UPSTREAM.md`).

## Use

From the source tree:

```python
from oracle import ADAPTERS, verify

result = verify(receipt, ADAPTERS, key_provider=keys, predecessor=prev)
print(result.fmt, result.verdict)
```

From an installed `asqav` wheel:

```python
from asqav.verifier.oracle import ADAPTERS, verify
```

Importing the oracle puts the bundled verifier directory on `sys.path` so the
standalone `verify_receipt` resolves as a top-level module. That is the one name
it exposes globally; it loads only when you explicitly import the oracle, never
from `import asqav` alone.

Run the bundled corpus:

```bash
python -m oracle.runner
```
