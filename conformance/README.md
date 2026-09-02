# Asqav Conformance Vectors

Reference inputs for any third-party implementation that wants to verify an Asqav signed record. Use these vectors to confirm that your fingerprint format and hashing match ours, bit-for-bit.

## What these vectors cover

1. The exact JSON fingerprint format we use before signing.
2. The SHA-256 digest of the canonical bytes.
3. The shape of the `_counterparty` field.

These vectors intentionally do NOT pin ML-DSA-65 signature bytes. FIPS 204 supports both deterministic and randomized signing; our server uses the randomized variant so signatures vary across calls for the same input. Verification is still deterministic, which is the property auditors care about.

## Fingerprint format

Asqav canonicalizes per RFC 8785 (JCS):

- UTF-8 encoded.
- Object keys sorted by UTF-16 code units per RFC 8785 Section 3.2.3; a code point or UTF-8 byte sort is not conformant for keys outside the Basic Multilingual Plane. The two orders agree across the BMP and diverge above U+FFFF, because a supplementary character's UTF-16 form begins with a surrogate in `0xD800..0xDBFF`, which sorts below every BMP character from `U+E000` up. Python's `sorted()` and `json.dumps(sort_keys=True)`, and Rust and Go byte ordering, are code point order and must not be used directly; JavaScript string comparison is already UTF-16 order. Vector `asqav-24-jcs-astral-key-order` pins this.
- No insignificant whitespace between tokens.
- Strings escaped per JSON RFC 8259.
- Integers serialized bare; floats follow Python's shortest-repr, so float forms outside the corpus domain are out of scope (NaN/Infinity are rejected).

Before signing, the server formats `{"action_type": ..., "context": ...}` (plus server-side metadata) into the standard form and hashes with SHA-256. The hash is what ML-DSA-65 signs.

## Verifying an Asqav signature

1. Fetch the record: `GET https://api.asqav.com/api/v1/verify/{signature_id}`.
2. Re-build the canonical bytes from the returned `payload` using your JCS implementation.
3. SHA-256 the canonical bytes; compare to the `chain_hash` field.
4. Verify the `signature` bytes against the agent `public_key` using any FIPS 204 / ML-DSA-65 library (liboqs, BoringSSL, pqcrypto, OpenSSL 3.5+).

## Vectors

See `vectors.json`. Each vector has:

- `input`: the wire object. Every member here is a real wire member.
- `canonical`: the JCS-canonical byte sequence, shown as a UTF-8 string.
- `sha256`: the SHA-256 of the canonical bytes, hex-encoded.
- `expected`: values that are NOT wire members - what a correct implementation should
  produce, plus vector metadata. Do not implement a field from this block.

If your implementation produces the same `canonical` and `sha256` for each `input`, you are ready to verify live Asqav records.

### Counterparty binding vectors

A counterparty acknowledgment carries `counterparty_binding.envelope_hash`: the SHA-256
of the ORIGINATING envelope, base64. The scope of that digest is declared on the wire in
`counterparty_binding.scope`:

- `envelope_minus_anchors` - the digest covers the JCS bytes of `{payload, signature}`
  only. Anchors are excluded because they are OPTIONAL and change after issuance, so a
  digest covering them would break every acknowledgment already emitted the moment the
  originator's timestamp was upgraded.
- An ABSENT `scope` member means a legacy draft-04..-08 three-key binding, which a
  verifier reports as `unverifiable, legacy scope`. A verifier MUST NOT try both scopes
  and accept whichever matches.

Each such vector names the envelope its digest is taken over in
`expected.originating_envelope_ref`, so the value is recomputable from this file alone.
`counterparty_binding_envelope_byte_equality` is that originating envelope and is a pure
canonicalization fixture: its own `sha256` is the digest of the full three-key object and
is deliberately NOT the binding value.

### Derived members, and how they are kept honest

`canonical`, `sha256` and the `envelope_hash` family are DERIVED - regenerate them, never
hand-edit them:

    python verifier/regen_fingerprint_vectors.py           # rewrite every derived member
    python verifier/regen_fingerprint_vectors.py --check   # CI gate; nonzero on any drift
    python verifier/freeze_corpus_lock.py                  # then move the pins

The `--check` pass recomputes every derived member from that vector's own `input` and
fails on any difference. It exists because the freeze below proves the file has not
CHANGED, which is not the same as proving it is RIGHT: the counterparty `envelope_hash`
was stale in five vectors while every pin and every test stayed green, because the only
tests that named it built their expected value by calling the function under test.
`python/tests/test_corpus_integrity.py` now pins each derived member as a literal string.

## Reporting mismatches

If your implementation produces a different `canonical` or `sha256` for one of the provided inputs, open an issue at https://github.com/jagmarques/asqav-sdk/issues with your implementation, language, and JCS library name.

## Corpus freeze at v1 (criterion 420)

This corpus is frozen at version 1. `manifest.lock.json` pins every corpus
file by SHA-256 and byte length, and carries the digest of the lock itself;
any drift fails CI.

- Lock digest: pinned in the lock's own `digest` field and reproduced in the
  repo as the `FINGERPRINT_LOCK_DIGEST` constant in
  `python/tests/test_corpus_lock.py`; a regenerated lock must update both.
- Verification path A: `python/tests/test_corpus_lock.py` re-derives every
  pin with `hashlib` + `pathlib`.
- Verification path B: `verifier/check_corpus_lock.sh` re-derives every pin
  through the `sha256sum` binary and `wc`.
- Regeneration after an intentional edit: `python verifier/freeze_corpus_lock.py`.

### Published signing seed (ML-DSA-65)

The server signs the SHA-256 fingerprint digest with ML-DSA-65. For
signature-byte parity this lock publishes a corpus signing seed and a
deterministic known-answer signature:

- Seed derivation: `SHA-256("asqav conformance corpus v1 ML-DSA-65 signing seed")`;
  the seed hex and the derived public key are in the lock's `signing.mldsa65`.
- KAT message: the 32-byte digest of the `minimal_read` vector (the hash is
  what ML-DSA-65 signs). KAT signature: `signing.mldsa65.known_answer.signature_hex`,
  produced with `dilithium-py` `ML_DSA_65.sign(sk, m, deterministic=True)` -
  the FIPS 204 pure variant. Re-signing from the seed reproduces the exact
  bytes, and CI re-derives them on every run.
- Production signing uses the randomized (hedged) FIPS 204 variant, so live
  signature bytes are intentionally not reproducible; verification is
  deterministic, and that is the property this corpus pins.

## License

These vectors are public domain (CC0).
