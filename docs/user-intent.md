# User-intent binding

A receipt that says "agent X ran this action" answers half the question. The other half is "did the user actually authorize this specific action right now". User-intent binding is the primitive that answers it.

## The shape

When you call `agent.sign(...)`, you can attach an optional `user_intent` envelope. The user produces a signature over the bytes they're asserting (typically a digest of action_type plus context plus a fresh nonce). The SDK passes those bytes to the backend verbatim. The backend verifies the signature with the declared algorithm and stores both the agent signature and the user signature on the same record.

The envelope:

```json
{
  "signature": "<base64>",
  "public_key": "<base64>",
  "algorithm": "ed25519",
  "key_id": "optional, useful for WebAuthn",
  "signed_message": "<base64; what the user actually signed>",
  "signed_at": "2026-04-28T12:00:00Z"
}
```

Supported algorithms today:

- `ed25519` - verified server-side via the cryptography library.
- `ecdsa-p256` - verified server-side via the cryptography library.
- `webauthn` - cryptographically verified server-side. `public_key` is a CBOR-encoded COSE credential key (EdDSA or ES256), `signature` is the assertion signature, `signed_message` is the bytes the authenticator signed (per WebAuthn, `authenticatorData || SHA-256(clientDataJSON)`). Higher-level checks (challenge match, RP ID, origin, user-present flag) live with the relying-party front end.

If verification fails, the backend returns 400 with `code: USER_INTENT_INVALID` and does not sign over a bogus envelope.

## What goes in `signed_message`

Whatever the customer chooses. The recommended pattern is a SHA-256 digest of:

- the action_type
- a stable serialization of the context
- a nonce that's fresh per action

Asqav stores the bytes; it does not interpret them. If you want the receipt to prove "this user authorized exactly this action", make sure your signed bytes commit to enough of the action context for that claim to hold.

## Python example with Ed25519

```python
import base64, hashlib, time
from nacl.signing import SigningKey
from asqav import Agent

# In production, this private key lives on the user's device or hardware.
sk = SigningKey.generate()
vk = sk.verify_key

action_type = "transfer:funds"
context = {"to": "acct_42", "amount_eur": 100}
nonce = "9f3c..."  # fresh per action
digest = hashlib.sha256(
    f"{action_type}|{context}|{nonce}".encode()
).digest()

sig = sk.sign(digest).signature

agent = Agent.get("agt_xxx")
resp = agent.sign(
    action_type,
    context,
    user_intent={
        "signature": base64.b64encode(sig).decode(),
        "public_key": base64.b64encode(bytes(vk)).decode(),
        "algorithm": "ed25519",
        "signed_message": base64.b64encode(digest).decode(),
        "signed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    },
)
assert resp.user_intent_verified is True
```

## Browser example with WebAuthn (sketch)

The actual WebAuthn flow is the customer's responsibility. The sketch below is a starting point:

```js
// 1. On enroll: navigator.credentials.create(...) -> store credential id
// 2. On action: build a digest of action_type + context + nonce
// 3. navigator.credentials.get({ publicKey: { challenge: digest, ... } })
// 4. Pull authenticatorData + clientDataJSON + signature out of the assertion
// 5. Send via the SDK:

await agent.sign({
  actionType: "transfer:funds",
  context: { to: "acct_42", amount_eur: 100 },
  userIntent: {
    signature: b64(assertion.response.signature),
    public_key: b64(storedPublicKey),
    algorithm: "webauthn",
    key_id: assertion.id,
    signed_message: b64(digest),
    signed_at: new Date().toISOString(),
  },
});
```

WebAuthn assertion verification is store-only today. The bytes are persisted on the receipt, but `user_intent_verified` will be `false` until the WebAuthn worker ships.

## What this is not

- It is not an identity primitive. The user public key only identifies "the same key that signed before", not "Jane Smith".
- It is not a content disclosure. The signed bytes are whatever you put in them; in hash-only mode you typically sign a digest, not the raw context.
- It is not a replacement for the agent signature. Both end up on the receipt; they answer different questions.

## Roadmap

- WebAuthn assertion verification (parse `authenticatorData` plus `clientDataJSON`, verify against the stored COSE key).
- A `verifyUserIntent(signatureId)` helper on both SDKs that re-runs verification against the stored bytes.
- Optional policy: require `user_intent_verified=true` for specific action_type patterns.
