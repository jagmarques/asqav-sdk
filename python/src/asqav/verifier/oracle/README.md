# Oracle - universal neutral verifier

Verify agent receipts across formats behind one seam. The oracle extends the
standalone `../verify_receipt.py` (package sibling `asqav.verifier.verify_receipt`)
to verify rival and sibling receipt formats, not only Asqav-native receipts.

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

## VC export shim (export-only, EU-lane presentation)

`asqav.verifier.vc_export.to_vc_envelope(receipt, *, format=None)` presents an Asqav
receipt in W3C Verifiable Credential 2.0 shape for EU-lane consumers. It is export-only:
NOT a wire token, NOT a first-class internal type, and NOT a verification path. The
converter is pure and offline (no signing, no network) and reuses the oracle's detection
to map each format's action/agent fields into `credentialSubject`.

The proof is Asqav-native, not a standard VC suite. An Asqav receipt is signed over the
Asqav canonical bytes of its own payload; a registered VC proof signs the VC's own bytes.
Relabelling the Asqav signature as `Ed25519Signature2020` (or any registered suite) would
produce a credential a conformant VC verifier rejects, so the export instead emits an
explicit non-standard `proof.type` of `AsqavReceiptSignature2026` that carries the
original signature verbatim, its algorithm and key id, and a `signingInputBase64` pointer
to the exact bytes it covers. A top-level `proofIsAsqavNative: true` marks the boundary.
Re-checking the proof needs an Asqav-aware verifier; a generic VC verifier MUST decline it.
This envelope is NOT a standard-suite-signed VC and cannot masquerade as one.

## Use

From a source checkout or an installed `asqav` distribution:

```python
from asqav.verifier.oracle import ADAPTERS, verify

result = verify(receipt, ADAPTERS, key_provider=keys, predecessor=prev)
print(result.fmt, result.verdict)
```

The oracle ships inside the `asqav` package (`asqav.verifier.oracle`); its
adapters import the standalone verifier as the package sibling
`asqav.verifier.verify_receipt`. It loads only when you explicitly import the
oracle, never from `import asqav` alone.

Run the bundled corpus (the conformance-vectors corpus stays at the repo root,
so this resolves in a source checkout):

```bash
python -m asqav.verifier.oracle.runner
```
