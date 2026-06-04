# Upstream-derived conformance vectors

The `aerf-up-*` vector directories are derived from the AERF conformance
corpus at <https://github.com/aerf-spec/aerf>, cloned at commit
`59fce60fe30bde35318812f502a55ab4bace4650`.

The upstream code and vector data are licensed Apache-2.0; the upstream
prose is licensed CC-BY. The receipt artifacts and public keys are
reproduced from that corpus. Each upstream public key (SPKI PEM) is
translated into our `keys.json` key map as raw Ed25519 hex, keyed by the
upstream `key_id` (the leading 16 hex characters of the SHA-256 of the
public key).

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
