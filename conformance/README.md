# Asqav Conformance Vectors

Reference inputs for any third-party implementation that wants to verify an Asqav signed record. Use these vectors to confirm that your canonicalization and hashing match ours, bit-for-bit.

## What these vectors cover

1. The exact JSON canonicalization rule we use before signing.
2. The SHA-256 digest of the canonical bytes.
3. The shape of the `_counterparty` field.

These vectors intentionally do NOT pin ML-DSA-65 signature bytes. FIPS 204 supports both deterministic and randomized signing; our server uses the randomized variant so signatures vary across calls for the same input. Verification is still deterministic, which is the property auditors care about.

## Canonicalization rule

Asqav uses RFC 8785 JSON Canonicalization Scheme (JCS):

- UTF-8 encoded.
- Object keys sorted lexicographically by Unicode code point.
- No insignificant whitespace between tokens.
- Numbers serialized per ECMAScript `ToString(ToNumber(x))`.
- Strings escaped per JSON RFC 8259.

Before signing, the server canonicalizes `{"action_type": ..., "context": ...}` (plus server-side metadata) and hashes with SHA-256. The hash is what ML-DSA-65 signs.

## Verifying an Asqav signature

1. Fetch the record: `GET https://api.asqav.com/api/v1/verify/{signature_id}`.
2. Re-build the canonical bytes from the returned `payload` using your JCS implementation.
3. SHA-256 the canonical bytes; compare to the `chain_hash` field.
4. Verify the `signature` bytes against the agent `public_key` using any FIPS 204 / ML-DSA-65 library (liboqs, BoringSSL, pqcrypto, OpenSSL 3.5+).

## Vectors

See `vectors.json`. Each vector has:

- `input`: the raw action payload.
- `canonical`: the JCS-canonical byte sequence, shown as a UTF-8 string.
- `sha256`: the SHA-256 of the canonical bytes, hex-encoded.

If your implementation produces the same `canonical` and `sha256` for each `input`, you are ready to verify live Asqav records.

## Reporting mismatches

If your implementation produces a different `canonical` or `sha256` for one of the provided inputs, open an issue at https://github.com/jagmarques/asqav-sdk/issues with your implementation, language, and JCS library name.

## License

These vectors are public domain (CC0).
