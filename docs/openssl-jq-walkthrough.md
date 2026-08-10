# Verify an Asqav Receipt With Only openssl, jq, and sha256sum

A receipt can be audited without installing any asqav component. This
walkthrough reproduces verification of a published asqav receipt using three
tools every audit machine already has: `jq` (extraction + canonicalisation),
`sha256sum` (digest recomputation), and `openssl` (signature verification).

Every step pins its expected value. The executable form is
`verifier/docs/openssl-jq-verify.sh`; it asserts every pin and exits nonzero
on any mismatch. It never reports a step it did not run.

## Inputs

The fixtures in `verifier/docs/fixtures/` are byte copies of the frozen
conformance corpus (their bytes are pinned by the script and by
`python/tests/test_corpus_lock.py`):

| Fixture | What it is |
|---|---|
| `published-receipt.json` | A real `api.asqav.com` payload-mode receipt, ML-DSA-65 signed, with an RFC 3161 anchor |
| `published-jwks.json` | The public key entry the receipt's `signature.kid` resolves to |
| `ed25519-receipt.json` | An asqav-native Ed25519 corpus receipt (runs on any OpenSSL 3.x) |
| `ed25519-jwks.json` | Its public key |

## Tool requirements

- `jq` and `sha256sum` (or `shasum -a 256`): every recomputation step.
- OpenSSL 3.0+: the Ed25519 signature step and all recomputation.
- OpenSSL **3.5+**: native ML-DSA-65 verification (the published receipt's
  algorithm). GitHub's `ubuntu-latest` runners ship OpenSSL 3.0.x without
  ML-DSA, so in CI the ML-DSA step is documented, not executed; see the
  honesty notes at the bottom.

## Step 0 - fixture byte pins

```sh
sha256sum verifier/docs/fixtures/*.json
```

Pinned output (file order may vary):

```
233a208e9564acea9f24faa2ffaa8bebce04d752bee685e581eef16c2b9c51a4  ed25519-jwks.json      (217 bytes)
c02fd4fc8cc26d4784b99515f84d2c61faec8f8d18c1ce6ae72f7406ac3cf084  ed25519-receipt.json   (832 bytes)
2ca81e3233f23ebdfa9f230c6914025f0dfd30a90c9cf322f70198c61d098bc4  published-jwks.json    (2851 bytes)
e5c81f01d570a9a1cf0aca8f6c773feba8ce01b9580d44ffac3a26dddbf22e63  published-receipt.json (11222 bytes)
```

## Part A - the published ML-DSA-65 receipt

### A.1 Rebuild the canonical payload bytes

Payload-mode receipts sign `canonical_json(payload)`: JCS key-sorted, no
whitespace, UTF-8. On this value domain `jq -cjS` produces the same bytes:

```sh
jq -cjS '.payload' verifier/docs/fixtures/published-receipt.json > canonical.bin
sha256sum canonical.bin
wc -c < canonical.bin
```

Pinned: SHA-256 `cc147376ec356902755eae69eb2134f5b8be9d570cd96ef685c6f708db1d2241`,
1097 bytes.

### A.2 Rebuild the anchored bytes

`anchors` is unsigned by design, so what the anchor binds is the envelope with
`anchors` removed:

```sh
jq -cjS 'del(.anchors)' verifier/docs/fixtures/published-receipt.json > env.bin
sha256sum env.bin
```

Pinned: SHA-256 `1252c422e91062dae94cf17bd8dc553132d8864ad671cf635476931c3469867e`,
5607 bytes.

### A.3 Extract the RFC 3161 anchor token

```sh
jq -r '.anchors[0].type' verifier/docs/fixtures/published-receipt.json   # rfc3161
jq -r '.anchors[0].value' verifier/docs/fixtures/published-receipt.json \
  | tr '_-' '/+' | base64 -d > anchor.der
sha256sum anchor.der
```

Pinned: 3559 bytes, SHA-256
`80e3f6e83c416e33cd1f2f67593a80b294c42fe9bde99211e8975e6f275b07f2`.

### A.4 The timestamp commits to these bytes

The token's TSTInfo carries a `messageImprint`: the SHA-256 of the anchored
bytes from A.2. Read it with `openssl asn1parse`:

```sh
openssl asn1parse -inform DER -in anchor.der | grep 1252C422
```

Pinned: one line containing the imprint OCTET STRING
`...04201252C422E91062DAE94CF17BD8DC553132D8864AD671CF635476931C3469867E...`
(`04 20` = OCTET STRING of 32 bytes, then the A.2 digest). This is the offline
proof the timestamp was taken over exactly these envelope bytes.

### A.5 Verify the ML-DSA-65 signature

Extract the signature and wrap the raw public key in SPKI PEM (the 22-byte
prefix is `id-ml-dsa-65`, OID 2.16.840.1.101.3.4.3.18):

```sh
jq -r '.signature.sig' verifier/docs/fixtures/published-receipt.json \
  | tr '_-' '/+' | base64 -d > sig.bin                     # 3309 bytes
jq -r '.keys[0].public_key' verifier/docs/fixtures/published-jwks.json \
  | base64 -d > pk.raw                                     # 1952 bytes
{ printf '\x30\x82\x07\xb2\x30\x0b\x06\x09\x60\x86\x48\x01\x65\x03\x04\x03\x12\x03\x82\x07\xa1\x00'
  cat pk.raw; } > pk.der
openssl base64 -in pk.der | { echo "-----BEGIN PUBLIC KEY-----"; cat; echo "-----END PUBLIC KEY-----"; } > pk.pem
```

Signature verification (OpenSSL 3.5+ only):

```sh
openssl pkeyutl -verify -pubin -inkey pk.pem -rawin -in canonical.bin -sigfile sig.bin
```

Pinned output, captured on OpenSSL 3.6.2: `Signature Verified Successfully`,
exit 0. On OpenSSL below 3.5 this command cannot run - ML-DSA support landed
in 3.5. The CI script documents the step there instead of claiming it.

Signature bytes pin: SHA-256
`d573c39f39dff27a3001edc5d9bcc45b3b2da23774a48de9bd7e18a65a8f066e` (3309 bytes).

### A.6 Full RFC 3161 chain walk (documented form)

The token itself is signed with ML-DSA-44, so a full `openssl ts` chain walk
needs an OpenSSL that parses ML-DSA inside CMS **and** the TSA trust anchor
asqav publishes for this receipt:

```sh
openssl ts -verify -token_in -in anchor.der -data env.bin -CAfile <asqav-tsa-chain.pem>
```

Neither dependency is assumed on an audit machine, so this step is documented,
never claimed. The A.4 imprint check is the part that runs fully offline.

## Part B - an Ed25519 receipt (runs on any OpenSSL 3.x)

```sh
jq -cjS '.payload' verifier/docs/fixtures/ed25519-receipt.json > canonical.bin
sha256sum canonical.bin          # 88051bbc8ba5f41fd1626ade433b923d12ba70a1767201ee78c8b407fc56b580 (533 bytes)

jq -r '.signature.sig' verifier/docs/fixtures/ed25519-receipt.json | base64 -d > sig.bin   # 64 bytes
jq -r '.keys[0].public_key' verifier/docs/fixtures/ed25519-jwks.json | base64 -d > pk.raw  # 32 bytes
{ printf '\x30\x2a\x30\x05\x06\x03\x2b\x65\x70\x03\x21\x00'; cat pk.raw; } > pk.der
openssl base64 -in pk.der | { echo "-----BEGIN PUBLIC KEY-----"; cat; echo "-----END PUBLIC KEY-----"; } > pk.pem

openssl pkeyutl -verify -pubin -inkey pk.pem -rawin -in canonical.bin -sigfile sig.bin
```

Pinned output: `Signature Verified Successfully`, exit 0.

## Honesty notes

- The ML-DSA-65 step's pinned output was captured against OpenSSL 3.6.2. On
  machines with OpenSSL below 3.5 the step is printed with its pinned expected
  output and labelled NOT EXECUTED; the script exits 0 only in that declared
  mode (`--mldsa-document-only`), and exits 1 when the capability is missing
  and no declaration was asked for.
- `jq -cjS` reproduces the asqav canonical bytes on the value domain of these
  receipts (string/int/bool/null values, no floats, ASCII). Outside that
  domain, canonicalisation must follow `docs/fingerprint-spec.md` byte for
  byte before any digest comparison.
- The standalone single-file verifier (`python/src/asqav/verifier/verify_receipt.py`)
  performs these same checks in pure Python; see `docs/offline-verification.md`.
