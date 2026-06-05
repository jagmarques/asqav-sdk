# Verify an Asqav receipt yourself, one dependency

`verify_receipt.py` is a single readable file that verifies an Asqav
Compliance Receipt on your own machine, without liboqs. It ships inside the
`asqav` package at `asqav/verifier/verify_receipt.py`, so `pip install asqav`
gives you the module and you can still copy that one file into an audit
environment and read every line.

It does the check that actually matters for third-party trust: it verifies the
post-quantum **ML-DSA-65 (FIPS 204)** signature over the receipt's canonical
bytes, resolves the issuer's public key from Asqav's public
`/.well-known/jwks.json` (so you never take our word for which key signed),
re-walks the SHA-256 hash chain, and reports the anchor binding and timestamp
skew. Everything except the signature math is Python standard library.

## Install

Standard library covers structure, canonical bytes, the hash chain, the anchor
binding, and the skew check with zero installs. The single dependency is the
post-quantum signature verify:

```bash
pip install dilithium-py
```

`dilithium-py` is a pure-python FIPS 204 implementation; its verify path uses
only stdlib SHAKE, so nothing compiles. If you skip it, every other check still
runs and the signature axis reports `SKIPPED`; the overall verdict is then
`INCOMPLETE`. The tool never prints `PASS` unless the signature was actually
verified.

## Verify a live receipt

Fetch a receipt and the key directory straight from the API and verify:

```bash
python -m asqav.verifier.verify_receipt --id sig_abc123
```

The public worked example needs no API key:

```bash
python -m asqav.verifier.verify_receipt --id sig_example_loan_decision_2026
```

This pulls `https://api.asqav.com/api/v1/verify/<id>` and
`https://api.asqav.com/.well-known/jwks.json`, then prints a per-axis report.
(The public `/verify/example` fixture ships a placeholder signature to document
the receipt shape; point `--id` at one of your own signed receipts to exercise
the full cryptographic path against a real signature.)

## Verify fully offline

Save the receipt and the key directory, then verify with no network at all:

```bash
curl https://api.asqav.com/api/v1/verify/sig_abc123 > receipt.json
curl https://api.asqav.com/.well-known/jwks.json    > jwks.json
python -m asqav.verifier.verify_receipt --receipt receipt.json --jwks jwks.json --offline
```

To check the hash-chain link, also save the predecessor receipt and pass it:

```bash
python -m asqav.verifier.verify_receipt --receipt receipt.json --jwks jwks.json \
    --predecessor previous.json --offline
```

## What it checks

| Axis | Checked | How |
|---|---|---|
| signature | ML-DSA-65 over canonical bytes | `dilithium-py` against the jwks public key |
| canonical bytes | JCS reproduction | stdlib `json` (sorted keys, no whitespace, UTF-8) |
| issuer_key | key resolution by `kid` | matched against `/.well-known/jwks.json` |
| chain | SHA-256 link to predecessor | stdlib `hashlib` |
| anchors | which envelope each anchor binds | stdlib `hashlib` + `base64` |
| skew | `issued_at` within 300s of now | stdlib `datetime` |
| structure | required fields and type namespace | stdlib |

It does **not** perform the full RFC3161 certificate-chain walk or
`policy_digest` artefact resolution; both need server state or ASN.1. For those,
use the hosted `/verify` endpoint or the full Asqav SDK.

## Exit codes

- `0` - PASS (every non-skipped axis passed, signature verified)
- `1` - FAIL (at least one axis failed)
- `2` - INCOMPLETE (a blocking axis, typically the signature, was skipped)
