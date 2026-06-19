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

## JWKS endpoint

```
GET https://api.asqav.com/.well-known/jwks.json
```

Public, unauthenticated. Re-snapshot when you rotate keys or when a key's `status`
changes to `revoked`. The verifier rejects signatures from revoked keys.
