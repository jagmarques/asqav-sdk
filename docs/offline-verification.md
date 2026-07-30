# Offline / Air-Gapped Receipt Verification

Asqav receipts can be verified entirely offline once you have a JWKS snapshot.
No network call is made during verification; all crypto runs in-process.

## Python

### 1. Install the verify extra

```
pip install "asqav[verify]"
```

The `verify` extra adds `dilithium-py` (ML-DSA-65) and `cryptography` (Ed25519, ES256).
Without it, the signature axis reports `SKIPPED`/`INCOMPLETE`, never a false PASS.

### 2. Snapshot the JWKS while online

```python
import asqav, json

jwks = asqav.fetch_jwks()               # hits https://api.asqav.com/.well-known/jwks.json
json.dump(jwks, open("jwks.json", "w")) # save for air-gap use
```

### 3. Verify offline

```python
import asqav, json

receipt = json.load(open("receipt.json"))
jwks    = json.load(open("jwks.json"))

result = asqav.verify_receipt_offline(receipt, jwks)
# result: {"verdict": "PASS"|"FAIL"|"INCOMPLETE", "axes": [...], "fmt": "..."}

assert result["verdict"] == "PASS", result["axes"]
```

To check a hash-chain link, supply the predecessor receipt:

```python
result = asqav.verify_receipt_offline(receipt, jwks, predecessor=prev_receipt)
```

### Verdicts

| Verdict      | Meaning                                                    |
|--------------|------------------------------------------------------------|
| `PASS`       | All axes checked and passed.                               |
| `FAIL`       | At least one axis failed (signature mismatch, bad chain).  |
| `INCOMPLETE` | A blocking axis was skipped (e.g., dilithium-py missing).  |

## TypeScript / Node

The TypeScript SDK ships ML-DSA-65 via `@noble/post-quantum` (pure JS, no WASM).

```typescript
import { fetchJwks, verifyReceiptOffline } from "@asqav/sdk";
import { writeFileSync, readFileSync } from "node:fs";

// Snapshot while online
const jwks = await fetchJwks();
writeFileSync("jwks.json", JSON.stringify(jwks));

// Verify offline
const receipt = JSON.parse(readFileSync("receipt.json", "utf-8"));
const jwksSaved = JSON.parse(readFileSync("jwks.json", "utf-8"));

const result = verifyReceiptOffline(receipt, jwksSaved);
// result.verdict: "PASS" | "FAIL" | "INCOMPLETE"
// result.axes: AxisResult[]  (axis, result, note)

if (result.verdict !== "PASS") {
  throw new Error(`Receipt invalid: ${JSON.stringify(result.axes)}`);
}
```

## Anchor binding and clock skew

`verify_receipt_offline` / `verifyReceiptOffline` cover structure, signature and the
hash chain. Two further axes are checked on request in both languages: anchor binding,
and `issued_at` within 300 seconds of the wall clock. `anchors` sits outside the signed
bytes, so that axis is the one an altered envelope can move without breaking the
signature.

Normalise the envelope first. The Python standalone verifier does it before any axis
runs, and an envelope that skips it digests different bytes.

Python:

```python
from asqav.verifier.verify_receipt import check_anchors, check_skew, normalise_envelope

env = normalise_envelope(receipt)
print(check_anchors(env))                         # ("PASS"|"FAIL"|"SKIPPED", note)
print(check_skew(env["payload"]["issued_at"]))
```

TypeScript:

```typescript
import { checkAnchors, checkSkew, normaliseEnvelope } from "@asqav/sdk/verifier";

const env = normaliseEnvelope(receipt);
console.log(checkAnchors(env));                   // ["PASS"|"FAIL"|"SKIPPED", note]
console.log(checkSkew(env.payload.issued_at));
```

An absent or empty `anchors` reports SKIPPED, and a present non-list value FAILs. An
anchor `value` is read as present only when it is genuinely base64 and decodes to at
least one byte, so an out-of-alphabet character is refused rather than dropped and a
value carrying no bytes never reads as an anchor. `verifier/axis-parity-cases.json`
and `verifier/anchor-value-cases.json` pin the cases both languages answer alike, and
both suites assert them.

An anchor value is one unwrapped base64 token. Surrounding or embedded whitespace is
refused, so MIME line-wrapped base64 of the kind `base64` and `openssl base64` emit by
default does not read as an anchor. Pass the unwrapped form (`base64 -w0`, or
`openssl base64 -A`).

### Where the two halves are not identical

Measured rather than asserted, over a 971-value anchor corpus and a 134-stamp corpus:

| Axis | Agreement | Residual |
|---|---|---|
| anchors, per value | 971 of 971 | none |
| skew, per stamp | 121 of 134 | 13 stamps, all ones TypeScript refuses and Python accepts |

The anchors row is measured on Python 3.11, 3.12 and 3.14, and all three answer alike.
The alphabet and padding rule lives in an explicit regex rather than in
`base64.b64decode(validate=True)`, whose strictness changed between 3.11 and 3.12.

Every residual runs in the direction where TypeScript is the stricter half, so a value
one language treats as valid is never read as valid by the other and then trusted. The
13 are the ISO 8601 basic forms (`20200101T000000`), the week date (`2020-W01-1`), and
the comma decimal separator, which `datetime.fromisoformat` reads from CPython 3.11 on.
Emit the extended form (`2026-06-19T00:00:00+00:00`) and both halves agree exactly.

## JWKS endpoint

```
GET https://api.asqav.com/.well-known/jwks.json
```

Public, unauthenticated. Re-snapshot when you rotate keys or when a key's `status`
changes to `revoked`. The verifier rejects signatures from revoked keys.

## Verification status by algorithm

| Algorithm | Status |
|-----------|--------|
| Ed25519 | Fully validated with real known-answer (tamper) vectors. |
| ES256 | Fully validated with real known-answer (tamper) vectors. |
| ML-DSA-65 | Fully proven. Known-answer conformance vector `asqav-06-mldsa65-payload-prod` was minted from a real api.asqav.com payload-mode receipt (2026-06-19, agent `agt_LBe47lJwgA0DfVom`, key `mxYqaLBR_T76ThNw0Kiekw`). Both Python (`test_verify_receipt_offline_mldsa65_real_cloud_kat`) and TypeScript test suites exercise the signature axis against this vector and assert PASS; tamper tests assert FAIL. |
