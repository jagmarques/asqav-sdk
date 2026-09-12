# Offline / Air-Gapped Receipt Verification

Verification can run offline using the complete signed receipt, a saved signer
JWKS and the inputs required by each check. Anchor verification in the standalone
tool also needs caller-selected TSA keys or Bitcoin block data, as described below.
No network call is made during offline verification.

## Python

### 1. Install the verify extra

```
pip install "asqav[verify]"
```

The `verify` extra adds `dilithium-py` (ML-DSA-65) and `cryptography` (Ed25519, ES256).
Without the dependency required by the receipt's signature algorithm, that check
reports `SKIPPED` and prevents a passing verdict. Dependencies installed separately
also work. An independent demonstrated failure can still make the failure class `invalid`.

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
# result: {"verdict": "verified"|"verified_keyed"|"unverified",
#          "failure_class": "invalid"|"unverifiable"|None, "axes": [...], "fmt": "..."}

assert result["verdict"] == "verified", result["axes"]
```

To check a hash-chain link, supply the predecessor receipt:

```python
result = asqav.verify_receipt_offline(receipt, jwks, predecessor=prev_receipt)
```

### Verdicts

| Verdict          | Meaning                                                    |
|------------------|------------------------------------------------------------|
| `verified`       | The verifier reports a pass for its evaluated checks; inspect the axes and documented limits.                               |
| `verified_keyed` | Passing result, but the digest is keyed (e.g. HMAC-SHA256) and not third-party re-derivable. Never reported as plain `verified`. |
| `unverified`     | Not verified; carries `failure_class` `invalid` or `unverifiable`. |

Every `unverified` verdict names its `failure_class`, and the two are never collapsed:

- `invalid` - a check ran and a cryptographic/policy binding failed (signature
  mismatch, chain-link mismatch, invalid anchor, counterparty binding mismatch,
  revoked/changed signer key, algorithm mismatch, or `issued_at` future-skew).
- `unverifiable` - recomputation could not complete (unresolvable key, missing
  or broken chain predecessor, malformed member, canonicalisation or parse
  failure, unresolvable policy digest, or a pending anchor without proof).

## Standalone single-file verifier (no asqav install)

`python/src/asqav/verifier/verify_receipt.py` is one Apache-2.0 file that runs
without the Asqav package. Its imports are the Python standard library and two
optional dependencies, both loaded inside the checks that use them:

- `dilithium-py` verifies ML-DSA-65 receipt and timestamp signatures.
- `cryptography` verifies Ed25519/ES256 receipts, decodes TSA certificates and
  keys, and verifies supported classical TSA signatures.

Copy the file and install the dependencies before entering the offline environment:

```sh
cp /path/to/asqav-sdk/python/src/asqav/verifier/verify_receipt.py ./
python -m pip install dilithium-py cryptography
python verify_receipt.py --receipt receipt.json --jwks jwks.json --offline
```

`--offline` prevents network access. The JWKS identifies receipt-signing keys;
applicable checks can need further inputs. `--predecessor FILE` supplies chain
bytes, `--tsa-key FILE` supplies a caller-pinned TSA public key or certificate
(PEM, DER or base64), and `--bitcoin-headers FILE` supplies a JSON map from block
height to block data containing `merkle_root` and `time` for OpenTimestamps.
Archive those inputs through a source you trust. The tool does not build a TSA
certificate path to a public root or validate Bitcoin consensus and chain history.

An applicable check lacking its required dependency or trust input reports
`SKIPPED` and prevents a passing verdict; another proven failure can still make
the failure class `invalid`. Missing `dilithium-py` does not disable Ed25519 or
ES256 verification when `cryptography` is installed. A signature pass alone does
not establish that the receipt's anchors passed.

`python/tests/test_standalone_verifier_surface.py` checks the import surface and
lazy loading of both dependencies. It also executes a copied file with Asqav
imports and outbound sockets refused. The OpenSSL alternative is documented in
[the walkthrough](openssl-jq-walkthrough.md), including its algorithm requirements.

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
// result.verdict: "verified" | "verified_keyed" | "unverified"
// result.failureClass: "invalid" | "unverifiable" | null
// result.axes: AxisResult[]  (axis, result, note, failureClass)

if (result.verdict !== "verified" && result.verdict !== "verified_keyed") {
  throw new Error(`Receipt not verified (${result.failureClass}): ${JSON.stringify(result.axes)}`);
}
```

## Counterparty bindings

Supply the complete originating signing envelope to check an acknowledgment's byte
commitment. Python accepts `originating_envelope` as a keyword; TypeScript accepts it
after the predecessor argument. Both APIs keep this input within the verification call.

```python
result = asqav.verify_receipt_offline(
    receipt, jwks, originating_envelope=originating_envelope,
)
```

```typescript
const result = verifyReceiptOffline(receipt, jwksSaved, null, originatingEnvelope);
```

The standalone CLI accepts `--counterparty originating-envelope.json`. Direct oracle
callers pass `VerificationContext(originating_envelope=...)` in Python, or a context
with `originatingEnvelope` in TypeScript.

The `envelope_minus_anchors` scope hashes JCS UTF-8 of exactly `{payload, signature}`.
It preserves every signature member and the signature string's spelling. Anchors and
export metadata can change without changing this commitment. Re-encoding the signature
string can change it, even when the decoded signature bytes stay equal.

| Binding evidence | Binding result | Failure class if other checks pass |
|---|---|---|
| Matching digest and expected acknowledging `signature.kid` | `matches` | none |
| Digest or acknowledging kid differs | `mismatch` or `kid_mismatch` | `invalid` |
| Scope is absent | `legacy_scope` | `unverifiable` |
| Scope is null or unsupported | `unrecognised_scope` | `unverifiable` |
| Supported scope, origin bytes unavailable | `unresolved` | `unverifiable` |
| Malformed binding or required digest/reference | `malformed` | `invalid` |

The scope check runs before origin lookup or hashing. An absent binding has no effect
on verification. An applicable unknown binding blocks a pass; a demonstrated integrity
failure on another axis still makes the overall result `invalid`.

For hosted signing, build the binding from the envelope supplied by the peer:

```python
from asqav.counterparty import compute_counterparty_binding

binding = compute_counterparty_binding(
    originating_envelope, receipt_ref=originating_signature_id,
).to_wire()
```

Pass the originating `signature_id` explicitly. The helper's `payload.action_id`
fallback is an opaque offline reference; the signing endpoint resolves `signature_id`.
The redacted public verification response does not supply the original signing bytes.
The server admits a binding only after resolving a visible compliance receipt and
recomputing the digest. Manually constructed bindings acquire no scope by default.

These checks establish byte equality. They do not independently verify the origin's
signature, fetch an origin over the network, or add anchor verification to the public
offline APIs. Evaluate those trust inputs separately.

## Anchor binding and clock skew

`verify_receipt_offline` / `verifyReceiptOffline` cover structure, signature and the
hash chain. Two further axes are checked on request in both languages: anchor binding,
and `issued_at` not more than 300 seconds ahead of the wall clock (a forward bound; a
receipt from the past never fails it). `anchors` sits outside the signed bytes, so that
axis is the one an altered envelope can move without breaking the signature; every anchor
commits to `sha256(JCS({payload, signature}))`, the two-key object the signer anchored,
never to the export's other top-level members.

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

Presence is not proof, and the draft says an anchor must not yield a valid verdict
without a successful cryptographic check. The Python verifier therefore evaluates the
token itself: an RFC 3161 anchor PASSes only when its `messageImprint` commits
`sha256(JCS(envelope minus anchors))` AND the TSA signature verifies against
caller-pinned TSA key material (`trusted_tsa_keys`, or `--tsa-key` on the CLI);
an OpenTimestamps anchor PASSes only when the proof commits the same digest and
its merkle path matches caller-supplied block data (`bitcoin_headers` /
`--bitcoin-headers`). Without usable supplied block data, placement is unverifiable; a supplied root that disagrees makes it invalid. A token whose check runs and fails reports
FAIL (`invalid`); one the check cannot complete offline — junk token, no pinned TSA
key, no header source, `status: pending`/`failed`, unknown type — reports SKIPPED
(`unverifiable`), never PASS. The TypeScript shim carries no CMS/ots evaluation, so
every shape-valid entry reports unverifiable there; that residual is conservative,
never permissive.

An anchor value is one unwrapped base64 token. Whitespace is refused, including a
trailing newline and MIME line wrapping, so a value piped from a shell base64 tool
reports FAIL. `openssl base64` wraps at 64 characters, GNU `base64` wraps at 76, and
BSD `base64` appends a trailing newline. Pass the unwrapped form instead
(`openssl base64 -A`, or `base64 -w0` on GNU). Values the Asqav signer and the SDK
produce are unaffected, since neither wraps.

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
| Ed25519 | Signature checked against known-answer and tamper vectors. |
| ES256 | Signature checked against known-answer and tamper vectors. |
| ML-DSA-65 | Signature checked against known-answer conformance vector `asqav-06-mldsa65-payload-prod`, which was minted from a real api.asqav.com payload-mode receipt (2026-06-19, agent `agt_LBe47lJwgA0DfVom`, key `mxYqaLBR_T76ThNw0Kiekw`). Both Python (`test_verify_receipt_offline_mldsa65_real_cloud_kat`) and TypeScript test suites exercise the signature axis against this vector and assert `verified`; tamper tests assert `unverified`/`invalid`. |

## Canonical member order and the dialect cutover

Every signed byte string is JCS (RFC 8785): member names ordered by UTF-16 code unit
(section 3.2.3), no whitespace, UTF-8, no NaN or Infinity. The standalone verifier's
`canonical_json`, the SDK's `asqav._jcs.canonical_json`, the TypeScript emitter and the
TypeScript verifier's `asqavJcs` all produce the same bytes, and the differential fuzzer
compares all of them on every run. A verifier sorting by code point (Python's
`json.dumps(sort_keys=True)`, Go and Rust byte order) agrees on every member name inside
the Basic Multilingual Plane and diverges on any name containing a character above
U+FFFF; conformance vector `asqav-24-jcs-astral-key-order` pins the difference.

`JCS_UTF16_CUTOVER` (exported by both verifiers) is the instant the issuing platform
switched to RFC 8785 order on the wire. A receipt issued before it whose member names
reach above U+FFFF, and whose signature verifies only under the earlier code-point order,
is reported on the signature axis as the pre-cutover dialect and the verdict stays
`unverified`. A receipt issued after the cutover gets no such retry. No production receipt
issued before the cutover carries such a member name (measured over the whole ledger on
2026-09-02), so the diagnostic exists for completeness rather than for live data.
