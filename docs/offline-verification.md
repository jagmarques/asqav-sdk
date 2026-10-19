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

## Whole Audit Pack offline

`asqav audit-pack verify --offline` checks an exported Audit Pack on your machine and
makes no network call. Install `pip install "asqav[cli,verify]"`: without the `verify`
extra no ML-DSA signature can be checked. Export the pack with `--include-signed-bytes`,
because a pack without the signed bytes has nothing to verify.

```sh
asqav audit-pack verify --offline bundle.json --jwks saved-jwks.json --tsa-key tsa.pem
asqav audit-pack verify --offline bundle.json --jwks saved-jwks.json --output json
```

### Where trust comes from

Trust comes only from files you supply. `--jwks FILE` is a key set you saved yourself, for
example from `/.well-known/jwks.json` while Asqav was reachable. `--tsa-key FILE`
(repeatable) pins the TSA's own signing certificate or public key, not a root CA. The
verifier checks each RFC3161 token against the pinned key itself and builds no certificate
path, so a pinned root CA verifies nothing. A pinned key that does not match the token's
signer makes that anchor fail, which marks the receipt `invalid`. The pack's own `jwks`
snapshot and any TSA entries it carries in `tsa_trust` are written by whoever produced the
pack, so they are never treated as trusted.

### Trust-bundle revocation policy

The shipped bundle (`verifier/trust/tsa-trust-bundle.json`) records each
root's `trusted_list_status_asserted_at`: that date is the revocation
assertion, not a live check. Offline verification performs no CRL or OCSP
fetch, so a root revoked after the bundle's `issued_at` still verifies until
the bundle is replaced. Refreshing means replacing the file with a newer
`bundle_version`; every result carries its `trust_bundle` identity (version,
digest, source), so compare it against the bundle file you hold and
re-verify when they differ.

Without `--jwks`, the command checks the pack against its own snapshot. The best result is
then `self-consistent, not anchored to a trusted key`, with exit code `2`, and the report
says the keys are self-asserted. It never exits `0` in that mode. The report prints the
bundle key's thumbprint so you can compare it with a key you hold.

### What it checks

- each receipt with `run_structured` from `verify_receipt.py`, which covers the signature,
  issuer binding, key status and anchors
- each receipt's chain link to the receipt with the previous `seq` from the same agent. A
  receipt missing from the middle of a chain is `invalid`
- forks and copies: two receipts with the same agent and `seq`, two that claim the same
  predecessor, or one `signature_id` listed twice are all `invalid`
- the bundle. It must carry a signature with a supported algorithm that matches its key's
  published `alg`. The digest is recomputed over the signed members. The key must be in the
  key set and belong to the pack's organization

A missing bundle signature is `invalid`, as is an unknown bundle key or an unsupported
algorithm. No receipt is reported `verified` unless the bundle signature verified.

A bundle key that the key set lists as revoked or suspended makes the bundle not verifiable
offline, because a key revoked after the export does not make an archived pack false. The
bundle is `invalid` only when signing after the revocation is proven: the signed window
`start` is at or after `revoked_at` (or `suspended_at`), or a receipt in the bundle carries
an anchor that a pinned TSA key places at or after that time.

Each receipt gets one of these statuses:

| Status | Meaning |
|---|---|
| `verified` | Every check passed against the keys used, including the timestamp proofs |
| `not_verifiable_offline` | Nothing failed, but a check could not finish offline. The reason is printed |
| `invalid` | A check ran and failed, such as a signature mismatch, a chain gap or a fork |

Common reasons for `not_verifiable_offline`:

- a non-compliance-mode receipt, which has no signed receipt envelope
- a pending OpenTimestamps anchor
- an RFC3161 anchor without a pinned TSA key (`--tsa-key`)
- signed bytes withheld at export

A pack exported today cannot reach exit code `0`. Its OpenTimestamps anchors are pending,
it carries no TSA certificates, and a non-compliance-mode receipt is never verified offline.
Exit `0` needs `--jwks`, a `--tsa-key` that verifies every RFC3161 anchor, and no pending
OpenTimestamps anchor. A completed OpenTimestamps proof is checked against
`--bitcoin-headers FILE`, as in the standalone verifier. TSA entries inside a pack are never
used as trust.

A lapsed `expires_at` on a receipt signed before that time is shown as an `info` line and
does not count as a failure.

Exit codes: `0` only with `--jwks`, when every receipt and the bundle verified. `1` when
anything is invalid. Every other result exits `2`, including a self-consistent pack checked
without `--jwks`.

The TypeScript CLI has the same command and the same trust rules, without `--tsa-key` or
`--bitcoin-headers`. It cannot check RFC3161 or OpenTimestamps tokens, so it lists anchored
receipts as not verifiable offline even where the Python command verifies them. Without
anchor times it cannot prove that a revoked bundle key signed after its revocation from an
anchor alone. A pack that Python marks `invalid` on that ground exits `2` in TypeScript.

## TypeScript / Node

The TypeScript SDK ships ML-DSA-65 via `@noble/post-quantum` (pure JS, no WASM).

```typescript
import { fetchJwks, verifyReceiptOffline } from "@asqav/sdk";
import { writeFileSync, readFileSync } from "node:fs";

// Snapshot while online
const jwks = await fetchJwks();
writeFileSync("jwks.json", JSON.stringify(jwks));

// Verify offline, from the raw text
const receipt = readFileSync("receipt.json", "utf-8");
const jwksSaved = readFileSync("jwks.json", "utf-8");

const result = verifyReceiptOffline(receipt, jwksSaved);
// result.verdict: "verified" | "verified_keyed" | "unverified"
// result.failureClass: "invalid" | "unverifiable" | null
// result.axes: AxisResult[]  (axis, result, note, failureClass)

if (result.verdict !== "verified" && result.verdict !== "verified_keyed") {
  throw new Error(`Receipt not verified (${result.failureClass}): ${JSON.stringify(result.axes)}`);
}
```

Pass the receipt as raw JSON text. The parser keeps each number as written, so the
verdict matches the Python verifiers. The receipt profile only accepts integers in the
safe range, so a receipt whose signed bytes hold `500.0` or `-0.0` comes back `unverified`
/ `unverifiable`. An object from `JSON.parse` cannot carry that spelling. With such an
object, a `-0` in a signed part is refused as `unverifiable`, and `500.0` reads back as
`500`. A float `500.0` the signer wrote then fails the signature as `unverified` /
`invalid`. A signed integer that the file respells as an integer-valued float, such as
`500` as `500.0` or `5e2`, or `0` as `0.0`, verifies on this path while raw text and the
Python verifiers refuse it. An integer past 2^53 anywhere in the receipt is refused as
`unverifiable` on raw text and by every Python verifier, a parsed dict included; a
`JSON.parse` object rounds it, so one outside the signed bytes goes unseen there.
`run_structured` checks only the predecessor it is handed, so pass it the whole
predecessor envelope as the CLI reads it.
Callers who need the strict profile must pass raw text. The
same applies to the predecessor and originating envelope arguments, which also accept raw
text.

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

In TypeScript, `verifyCounterpartyBinding(acknowledgment, origin)` and
`computeEnvelopeHash(origin)` take the origin as raw JSON text or as a
`parseJsonPreservingFloats` value, and then agree with Python's
`verify_counterparty_binding` and `compute_envelope_hash`. An origin from `JSON.parse` has
lost the spelling of its numbers. If it holds `-0`, which could have been the integer `0` or
the float `-0.0`, `verifyCounterpartyBinding` reports `unresolved`, and `computeEnvelopeHash`
and `computeCounterpartyBinding` throw `counterparty_origin_unavailable`. If the origin
signed the integer `500` and the file respells it as `500.0`, the binding reports `matches`
where Python reports `mismatch`; a float `500.0` the origin signed reports `mismatch`. An
integer past 2^53 anywhere in the origin makes the binding `unresolved` on raw text and in
Python, and a `JSON.parse` origin that holds one outside the signed bytes reports `matches`.
Pass raw text when you need Python's answer.

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

## Axes both offline APIs report

`verify_receipt_offline` and `verifyReceiptOffline` report these 13 axes, in this
order, and both languages report the same names on the same receipt. A receipt whose
`structure` axis fails stops there and reports fewer.

| Axis | What it checks |
|---|---|
| `structure` | Required members present, receipt type and wire version recognised, signature algorithm supported |
| `signature` | Ed25519, ES256 or ML-DSA-65 verifies over the canonical signed bytes |
| `chain` | The carried link rederives from the supplied predecessor |
| `seq` | The signed counter advances past the predecessor's counter |
| `expiry` | `expires_at` has not lapsed; this axis reports on its own and never folds the verdict |
| `nonce` | A different receipt reusing this issuer's nonce is flagged as a replay candidate |
| `key_binding` | The signed `key_thumbprint` rederives from the resolved signing key |
| `counterparty` | A claimed cross-agent binding commits to the origin envelope's bytes |
| `payload_digest` | `payload_digest` rederives from the context carried in the same receipt |
| `skew` | `issued_at` is at most 300 seconds ahead of the wall clock |
| `outer_copies` | The plain copies an exported entry carries (agent, action, time, signature, key, signed bytes) equal the signed values they restate |
| `key_status` | The signing key's published status is not revoked |
| `issuer_bind` | The verifying key is published under the issuer the signed bytes name |

`skew` is one of the 13, and it decides the verdict: an `issued_at` more than 300
seconds ahead of the wall clock reports `skew FAIL` and the verdict is `unverified`
with failure class `invalid`. The bound is forward-only, so a receipt from the past
passes it.

### Outer copies in an exported entry

An Audit Pack entry is the signed `payload`, its `signature` object and `anchors`,
plus plain top-level copies of some signed values. The copies are for convenience and
no signature covers them, so an unchecked edit to one would still read "signature
valid". Both offline APIs and the standalone verifier compare each copy with the value
it restates and fail the `outer_copies` axis when they differ. The verdict is then
`unverified` with failure class `invalid`, even though the signature itself verifies.
Read the agent and the time from the signed `payload`.

| Outer member | Signed value it must equal |
|---|---|
| `agent_id` | `payload.agent_id` |
| `action_id` | `payload.action_id` |
| `action_type` | `payload.action_type` (payload-mode receipts only) |
| `receipt_type` | `payload.type` |
| `supersedes` | `payload.supersedes` |
| `signed_at` | `payload.issued_at`, the same instant to the microsecond; an observation receipt may carry its caller-observed `context.ts` instead, because that is the time its record stores |
| `algorithm` | `signature.alg` |
| `signature_b64` | `signature.sig`, the same decoded bytes |
| `message_b64` | the canonical (JCS) bytes of `payload`, which are the bytes that were signed |
| `message_sha256` | the SHA-256 of those bytes, compared whenever the signature verified over them |
| `public_key_b64` | the key the signature verified under; listed instead when the signature did not verify |
| `compliance_mode` | `true`, whenever a `payload` object is present |
| `signature` (flat string), `signature_envelope` | the normalised signature object, the one the signature check used |
| any other top-level member named like a signed member (`issued_at`, `decision`, `org_id`, `type`, `v`, ...) | that signed member |
| `signature.<name>` | the signed member of the same name, when the signature object carries extras |

The signature copies are weighed against the normalised signature object, so a flat
`sig` string or the hosted `signature_envelope` spelling is checked the same way as
the object form. A member that is not a copy of anything signed is listed in the axis
note as "not signed, do not rely on", on a pass as well as on a failure:
`signature_id`, `capture_topology`, `stale_pending`, `corrupted`, `corruption_message`,
`signed_bytes_withheld`, `action_type` on a hash-mode receipt (its signed bytes carry no
action type), and any other name the receipt never signed (`verified`, `kid`, `agent_name`).
The hosted response's own `type: "signature"` is listed the same way.

Some copies cannot be judged and are listed rather than failed. An observation
`signed_at` whose signed `context.ts` is withheld by the default export, or is written in
a form this check does not read (basic ISO, hour-only, week dates, comma fractions), is
listed. The digest and the key are listed when the signature did not verify, because
until then the bytes compared need not be the signed bytes. A member that is absent or
`null` is not compared, except `compliance_mode`, which must be `true` whenever it is
present.

A flat hash-mode receipt (`mode: hash`, no `payload` object) signs its own fields, so it
has no copies. Its signature covers `v`, `mode`, `hash`, `hash_algo`, `metadata`,
`server_timestamp`, `action_id`, `agent_id`, `org_id`, `policy_digest` and
`policy_decision`. Every other member is listed as not signed, apart from these.
`algorithm` and `anchors` are read by their own checks. The signature string that was
verified (`signature_b64`, or `signature` when `signature_b64` is empty) is not listed,
and neither is a second string with the same decoded bytes. Any other `signature`, such as
an object, or a string that differs, is listed. So is a `payload` that is not null. `key_id`
only picks which published key to try, and nothing signs it. It is left unlisted when the
signature verified under the published key whose `kid` it names. In every other case it is
listed. The manual
[openssl and jq walkthrough](openssl-jq-walkthrough.md) reads the `payload` alone, so
it never relies on a copy.

`verifier/conformance-vectors/asqav-43` to `asqav-51` pin the behavior: one entry
whose copies all agree, one each with an edited `signed_at`, `agent_id`, `action_type`,
`message_b64` and `public_key_b64`, and the same refusal when the signature is a flat
string (`asqav-49` agrees, `asqav-50` edits `agent_id`) or sits under
`signature_envelope` (`asqav-51` edits `action_type`). Every verifier refuses the
edited ones. `asqav-52` pins the flat hash-mode listing.

### Where the standalone verifier's vocabulary differs

`run()` and `run_structured()` in `verify_receipt.py` report 14 axes on the same
receipt. Three names differ, and these three are the whole difference:

| Axis | Offline SDK APIs | Standalone verifier |
|---|---|---|
| `anchors` | not reported | reported |
| `issuer_key` | not reported | reported |
| `seq` | reported | not reported |

`python/tests/test_offline_entry_point_axis_declaration.py` executes both APIs and
asserts both tables above against the axis names they return, so a drift in either
direction fails the suite rather than aging quietly in this file.

## Anchor binding

`anchors` sits outside the signed bytes, so that axis is the one an altered envelope
can move without breaking the signature; every anchor commits to
`sha256(JCS({payload, signature}))`, the two-key object the signer anchored, never to
the export's other top-level members. Neither offline SDK API evaluates it, which is
why their `verified` verdict says nothing about the receipt's anchors. Check it on
request with the helpers below.

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

`check_skew` / `checkSkew` apply the same 300-second bound the `skew` axis applies,
for a timestamp you hold outside an envelope. The offline APIs call it for you.

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
FAIL (`invalid`); one the check cannot complete offline - junk token, no pinned TSA
key, no header source, `status: pending`/`failed`, unknown type - reports SKIPPED
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
is reported on the signature axis as the legacy dialect and the verdict stays
`unverified`. A receipt issued after the cutover gets no such retry. No production receipt
issued before the cutover carries such a member name (measured over the whole ledger on
2026-09-02), so the diagnostic exists for completeness rather than for live data.
