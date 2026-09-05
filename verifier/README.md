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
`unverified` with `failure_class=unverifiable`. The tool never reports
`verified` unless the signature was actually verified.

## Verify a live receipt

Fetch a receipt and the key directory straight from the API and verify:

```bash
python -m asqav.verifier.verify_receipt --id sig_abc123
```

The public worked example needs no API key:

```bash
python -m asqav.verifier.verify_receipt --id sig_example_regulator_cold_verify_2026
```

This pulls `https://api.asqav.com/api/v1/verify/<id>` and
`https://api.asqav.com/.well-known/jwks.json`, then prints a per-axis report.
This is a public documentation fixture. Its signing key is a reserved example
identity that is not published in the public JWKS, so the offline verdict is
`unverified` (issuer-key resolution cannot complete) rather than a `verified`
outcome. For a real receipt the hosted `/verify/<id>` JSON is a display
projection without the identifier members of the signed payload, so `--id` can
show the hosted verdict but cannot reproduce the signature; the full
cryptographic path runs on an Audit Pack entry, as the next section shows.

## Verify fully offline

Offline verification needs the signed bytes, and only the Audit Pack export carries them:
the hosted `/api/v1/verify/<id>` JSON is a display projection that omits the identifier
members of the signed payload (`agent_id`, `org_id`, `issuer_id` and the digests), so a
receipt saved from it can never reproduce the signature. Ask the receipt holder for an
Audit Pack exported with `include_signed_bytes` (`POST /api/v1/audit-pack/export` with an
API key carrying the `audit-pack:read` scope), take one entry of its `receipts` array as
`receipt.json`, save the key directory, and verify with no network at all:

```bash
curl https://api.asqav.com/.well-known/jwks.json > jwks.json
python -m asqav.verifier.verify_receipt --receipt receipt.json --jwks jwks.json --offline
```

The export carries more top-level members than the signer anchored; the verifier keeps
only `payload`, `signature` and `anchors`, and every anchor is checked against
`sha256(JCS({payload, signature}))`, the two-key object the signer committed. The
standalone tool checks ML-DSA-65 signatures only; a receipt signed with another level
reports the signature axis as SKIPPED with `unsupported alg`.

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
| anchors | each anchor's token must commit `sha256(JCS({payload, signature}))`, the two-key object the signer anchored; RFC3161 TSA signature against pinned TSA key material, OpenTimestamps merkle path against supplied bitcoin headers | stdlib DER/ots parse + `dilithium-py`/`cryptography` for the TSA signature |
| skew | `issued_at` not more than 300s in the future (a forward bound only; a receipt from the past never fails here) | stdlib `datetime` |
| structure | required fields and type namespace | stdlib |

An anchor never PASSes on presence alone: a token whose check runs and fails is
`invalid`, and one the check cannot complete offline (no pinned TSA key via
`--tsa-key`, no header source via `--bitcoin-headers`, a `pending`/`failed`
status, an unknown type) reports `unverifiable`. A verified anchor whose proven
time lands at or before a key's `revoked_at` lets a pre-revocation receipt pass
the `key_status` axis; a forged or unverifiable one never does.

A vector directory may carry the material those inputs need:
`tsa_trust.pem` (PEM certificates the offline verifier trusts for that
vector's timestamp-authority token) and `bitcoin_headers.json` (block headers
keyed by height, each with `hash`, `merkle_root` and `time`). Both are public
material — the certificates are embedded in the token itself, the headers are
Bitcoin public data; `conformance-vectors/asqav-24-anchor-block-hash-prod`
ships them, and `ANCHOR-MATERIAL.md` there records the two independent header
sources. A vector without them keeps its anchors axis SKIPPED by design, and
the requirement map says so.

## What it does not check

The list below is **not** the authority. Every verification result carries a
machine-readable `not_checked` array naming each check this tool does not
perform, and it is present on passing results too, so the boundary of the claim
travels with the claim instead of living in this file:

```json
{
  "verdict": "verified",
  "axes": [...],
  "not_checked": [
    {"check": "tsa_certificate_path",
     "requirement": "anchor trust",
     "reason": "no X.509 chain walk from the RFC 3161 signing certificate to a public root; offline trust comes only from the TSA keys the caller pins",
     "condition": "--tsa-key pins the key this tool will trust; it does not build a path to it"},
    ...
  ],
  "coverage": {
    "stopped_at": null,
    "checks_not_evaluated": [
      {"id": "tsa_certificate_path", "reason": "not_implemented", "status": "not_implemented",
       "requirement": "anchor trust",
       "condition": "--tsa-key pins the key this tool will trust; it does not build a path to it"},
      ...
    ]
  }
}
```

`condition` is `null` when the check is never performed at any invocation, and
names the input that would enable it when the gap is one you can close. Read the
current list from `asqav.verifier.verify_receipt.NOT_CHECKED`, or from any
result. The `coverage` block carries the same boundary in the shape the
reviewer's verifier publishes, so the two tools read side by side: `stopped_at`
is `null` when the full axis sequence ran, and names the axis evaluation stopped
at otherwise; `checks_not_evaluated` lists one `not_implemented` entry per
declared gap, then any axes evaluation stopped short of as `not_reached`. The
headline gaps are the TSA certificate path and its revocation state,
`policy_digest` artefact resolution, aggregate-anchor inclusion proofs, and the
caller-supplied framework taxonomies, which are carried under the signature but
never evaluated. For those, use the hosted `/verify` endpoint or the full SDK.

## Which requirements the corpus exercises

`conformance-vectors/requirement-map.json` maps each asqav-native vector to the
normative requirements it exercises, and publishes the requirements **no** vector
exercises alongside it. Both halves are generated by
`verifier/build_requirement_map.py`, which derives coverage by running each
vector through this verifier and reading the axes off the result: an axis the
verifier skipped is not coverage, whatever a vector'"'"'s notes say. A test fails if
the committed map drifts from the corpus.

## Outside recomputations

`independent-runs.json` records recomputations of this corpus run by third
parties from the published text alone, one entry per run. The file is
append-only: an entry is never edited or removed. An entry is the runner's own
claim, with the runner's own non-claims — it is not an Asqav endorsement, and
the file's purpose line says so. A test keeps the list honest: it refuses an
entry whose pinned asqav-sdk commit is not in this repository's history or
whose vector names are not directories in the corpus at that commit.

## The published artifact, probed

`artifact_probe/probe_published_artifacts.py` installs the PUBLISHED wheel and
the PUBLISHED npm package into throwaway environments with the source tree off
the path, and checks every entry point the packaged READMEs document — the
repository's own tests prove the repository, not the artifact a user installs.
The `Artifact probe` workflow (`.github/workflows/artifact-probe.yml`) runs it
weekly, on demand, and after every successful Publish, failing when the probe
exits non-zero and uploading the probe's output as a job artifact.

## The same axes from TypeScript

This file is Python. The TypeScript SDK reaches the same axes through
`@asqav/sdk/verifier`, which exports `normaliseEnvelope`, `checkAnchors` and
`checkSkew` alongside `verify`. `docs/offline-verification.md` carries the paired
snippets and the measured list of stamp spellings the two halves read differently.
There is no single-file TypeScript download equivalent to this script.

## Verdict vocabulary

The verifier reports one of three verdicts (criteria 418/438):

| Verdict | Meaning |
|---|---|
| `verified` | Every non-skipped axis passed and the signature was actually checked. |
| `verified_keyed` | Same as `verified`, but the digest is keyed (e.g. HMAC-SHA256) and so is internally consistent yet not third-party re-derivable. Never reported as plain `verified`. |
| `unverified` | The receipt is not verified. Carries a `failure_class` of `invalid` or `unverifiable`. |

Every `unverified` verdict names why, and the two classes are never collapsed:

- `invalid` - a check ran and a cryptographic/policy binding failed: signature
  mismatch, chain-link mismatch, anchor invalid, counterparty binding mismatch,
  signer key revoked/changed, algorithm mismatch, or `issued_at` future-skew
  bound violation.
- `unverifiable` - recomputation could not complete: unresolvable key, missing
  or broken chain predecessor, malformed member, canonicalisation or parse
  failure (including a duplicated JSON member name), or a pending anchor without
  cryptographic proof.

## Exit codes

- `0` - `verified` or `verified_keyed`
- `1` - `unverified`, `failure_class=invalid` (a binding was proven broken)
- `2` - `unverified`, `failure_class=unverifiable` (verification could not
  complete; a blocked check is never reported as verified)
