# Fingerprint Spec

When the Asqav SDK signs an action, it first builds a small fingerprint of the action data. This page is the spec for that fingerprint, so any verifier can rebuild it independently in any language with no Asqav SDK installed.

The output is a string of the form:

```
sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
```

You send that string as the `hash` field on `/sign`. Asqav signs it with your agent's ML-DSA key and returns a signature. You, or any third party, can later rebuild the fingerprint from the original `(action_type, context)` pair and verify the signature.

## Why a spec at all

Different programming languages serialize JSON in slightly different ways: float formatting, key ordering, Unicode escapes, whitespace. If two implementations disagree on those bytes, they produce different hashes for the same data, and signatures don't verify. The spec pins one format so both sides agree.

## The format

The Python SDK and the TypeScript SDK both produce the fingerprint as:

```python
import hashlib, json

def hash_action(action_type: str, context: dict) -> str:
    payload = {"action_type": action_type, "context": context or {}}
    body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()
```

This is RFC 8785, the JCS standard, applied to the `{action_type, context}` pair, then SHA-256 over the resulting bytes. Stdlib only, no third-party packages. The output is `sha256:<64 hex chars>`.

## Inputs

- **`action_type`**: short string identifying the action, for example `"read:data"` or `"api:call"`. The same string you would have sent in full-payload mode.
- **`context`**: the action payload dict. Any JSON-serializable value. May be empty.

The fingerprint is computed over the combined object `{"action_type": ..., "context": ...}`, so `action_type` cannot be tampered with independently of the context bytes.

## Format rules

The same rules expressed in any language:

- **Object keys**: sorted lexicographically by Unicode code point, which is Python's `sorted()` default.
- **Whitespace**: none, anywhere.
- **Strings**: emitted verbatim as UTF-8, with no `\uXXXX` escaping for characters above U+001F.
- **Numbers**: serialized via Python's `json` module rules. Integers stay integers like `42`, and floats use the shortest round-trip form, so `1.5` not `1.50`.
- **Null / true / false**: lowercase.
- **Arrays**: order preserved.
- **Nested objects and arrays**: recurse, same rules.
- **NaN / Infinity / -Infinity**: rejected under `allow_nan=False`. Filter or convert to `None` before hashing.

## Worked example

```python
action_type = "read:data"
context = {}

# Combined payload, in standard form:
body = b'{"action_type":"read:data","context":{}}'

# SHA-256:
# sha256:991033e23a3e0939c258e10f2f8f88183d9bcdb08de62ef02446e11e0819c671
```

See `conformance/vectors.json` for additional worked examples.

## Subtleties worth reading

1. **Float precision**. `0.1 + 0.2 == 0.30000000000000004` in IEEE 754. Python's `json.dumps` emits `0.30000000000000004`. If you want to hash `0.3`, pass `0.3` literally; do not arithmetic to get there. If your domain needs exact decimals, send strings or scaled integers, not floats.
2. **Unicode normalization**. We do NOT normalize Unicode. `"café"` as the single codepoint U+00E9 and `"café"` as e plus a combining acute hash differently. If your input might be in either form, NFC-normalize before hashing: `unicodedata.normalize("NFC", s)`.
3. **Key ordering of nested dicts**. Recursive: every dict at every depth is sorted independently.
4. **List of dicts**. Lists preserve input order. Each dict inside is formatted in standard form. So `[{"b":1,"a":2},{"a":3}]` becomes `[{"a":2,"b":1},{"a":3}]`, with inner keys sorted and outer order preserved.
5. **Empty values**. `{}` becomes `{}`. `[]` becomes `[]`. Both hash to a stable value.
6. **`None` / `null`**. Allowed. Serializes as `null`.
7. **Tuples**. Python tuples serialize as arrays. So `(1,2,3)` and `[1,2,3]` produce the same bytes. If your code relies on the difference, convert to lists first.
8. **Keys must be strings**. JSON forbids non-string keys. `{1: "x"}` will raise. Convert to `{"1": "x"}` or fail loudly.
9. **`NaN`, `Infinity`, `-Infinity`**. Rejected by `allow_nan=False`. Filter or convert to `None` before hashing.
10. **bytes**. Not JSON. Base64-encode and pass as a string.
11. **datetime / Decimal / UUID**. Not JSON. Convert to string before hashing, for example ISO 8601, `str(decimal)`, or `str(uuid)`. Whatever convention you pick, pick it once and stick to it.
12. **`tool_name` and `model_name`**. These should be static identifiers like `"database.query"` or `"gpt-4o"`, not user input or PII. The metadata whitelist passes them through unchanged, so a name like `"lookup_user_email_for_alice@example.com"` would land in the audit trail verbatim.

## Cross-language guarantee

Both `asqav` for Python and `@asqav/sdk` for TypeScript produce byte-identical output for the same input. The 8 conformance vectors at `conformance/vectors.json` are exercised by CI on both SDKs on every change, so the two stay in lockstep. If you write a verifier in another language, point it at the same vectors first.

## Replay protection

Signatures are NOT replay-stable: re-sending the same `(action_type, context)` produces a fresh signature with a new `server_timestamp` and `action_id`. Verifiers MUST track `action_id` uniqueness within their freshness window and reject duplicates.

## Verifying mode

Each signed record has a `mode` field on `signature_records` with value `"hash"` or `"payload"`. Verifiers SHOULD read `signature_records.mode` to determine which form to expect, rather than inspecting the signed bytes. The hash-mode signing input includes `"v": 1` and `"mode": "hash"` for self-description; the payload-mode signing input keeps the flat shape unchanged.

## Test vectors

See `conformance/vectors.json`. The hash-mode happy-path vectors (`minimal_read`, `tool_call_with_counterparty`, `traced_child_action`) are the ones that exercise this spec; the remaining vectors cover the agent-card verification feature, not GDPR hash signing.

## Versioning

This spec is version 2. The `vectors.json` file declares `version: 2` to align. Hash-mode signing input includes `"v": 1`, the wire version of the signing-input schema, which is distinct from this spec version, so a future v2 wire format will be distinguishable from v1 in every signature.

## See also

- `conformance/vectors.json`, the test vectors both SDKs pass
- README in this repo, covering mode selection: hash-only vs full-payload
