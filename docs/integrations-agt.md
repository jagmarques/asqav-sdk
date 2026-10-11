# Microsoft Agent Governance Toolkit: Asqav as the independent signer

Microsoft's Agent Governance Toolkit (AGT) can sign a receipt for every MCP tool
call it governs. Each receipt records the Cedar decision for one call, is signed
with Ed25519 and points at the previous receipt's hash. The signing key lives in
the same process that made the decision, and AGT's receipts proposal says what
that means:

> This is **self-attestation**: it detects later tampering but does not prove
> that an independent party evaluated the action.

(AGT, `docs/proposals/verifiable-compliance-receipts.md`)

The `asqav.agt_receipts` module records each AGT receipt in an Asqav receipt,
signed by the Asqav service with Asqav's key, and links the two. Whoever runs the
agent can rewrite and re-sign their own AGT receipt. They cannot produce a matching
Asqav receipt for the rewritten one.

## Versions

- The `agt` extra pins `agent-governance-toolkit-protocols==5.0.0`, the PyPI
  release that ships AGT's `mcp_receipt_governed` module. AGT is MIT licensed.
- That release has no external-authorization hook. AGT added one on its main
  branch (`authorize_receipt`, `external_authorizer`, `trusted_authorizer_keys`)
  after 5.0.0. It has not reached PyPI yet. This module was tested against AGT
  commit `c767f8333f413bd1a93e82836b7aa226280b61ae` for that path.
- AGT needs Python 3.11 or later, so the extra installs nothing on 3.10.

## Install

```bash
pip install "asqav[agt]"
```

Add `asqav[verify]` as well to check Asqav signatures offline.

## Countersign an AGT receipt (any AGT release)

```python
import asqav
from mcp_receipt_governed import McpReceiptAdapter
from asqav.agt_receipts import AsqavAgtSigner

asqav.init("sk_live_...")
signer = AsqavAgtSigner(agent_name="agt-signer")

adapter = McpReceiptAdapter(cedar_policy=CEDAR_POLICY, signing_key_hex=AGT_KEY_HEX)
receipt = adapter.govern_tool_call("did:example:agent", "read_file", {"path": "/tmp/a"})

link = signer.countersign(receipt)
print(link.signature_id, link.verification_url)
```

`countersign` runs AGT's own `verify_receipt` first and refuses a receipt AGT
rejects. The Asqav receipt binds the AGT receipt id, its payload hash and the
AGT signer key. Store `link.asqav_receipt` (a JSON-safe dict) next to AGT's
exported receipts; `signer.links` keeps every link by AGT receipt id.

## Use Asqav as AGT's external authorizer (AGT with `authorize_receipt`)

```python
signer = AsqavAgtSigner(agent_name="agt-authorizer", authorizer_key_hex=AUTHORIZER_KEY_HEX)

adapter = McpReceiptAdapter(
    cedar_policy=CEDAR_POLICY,
    signing_key_hex=AGT_KEY_HEX,
    external_authorizer=signer,
    trusted_authorizer_keys=[signer.authorizer_public_key],
)
receipt = adapter.govern_tool_call("did:example:agent", "read_file", {"path": "/tmp/a"})
print(receipt.assurance_level, receipt.authorizer_id)
# externally_authorized asqav:sig_...
```

For each tool call the signer gets an Asqav receipt over the AGT receipt plus a
fresh nonce, an expiry and the authorizer public key. It then calls AGT's
`authorize_receipt` with `authorizer_id` set to `asqav:<signature_id>`. AGT's
signed authorization payload covers `authorizer_id`, so AGT's own verifier
carries the Asqav reference, and any edit to it breaks AGT's check.

## Verify

```python
import asqav
from mcp_receipt_governed import verify_receipt_chain
from asqav.agt_receipts import verify_asqav_link

errors = verify_receipt_chain(
    receipts,
    trusted_keys=[AGT_PUBLIC_KEY_HEX],
    trusted_authorizer_keys=[AUTHORIZER_PUBLIC_KEY_HEX],
    require_external_authorization=True,
)
jwks = asqav.fetch_jwks()
asqav_receipts = {rid: link.asqav_receipt for rid, link in signer.links.items()}
for receipt in receipts:
    errors += verify_asqav_link(receipt, asqav_receipts[receipt.receipt_id], jwks=jwks)
assert errors == []
```

Drop `trusted_authorizer_keys` and `require_external_authorization` for
countersigned receipts; AGT 5.0.0 does not accept them. `verify_asqav_link` recomputes the Asqav `action_ref`
from the AGT receipt and compares it with the one Asqav signed. For an
authorized receipt it also checks that `authorizer_id` names that Asqav receipt.
With a JWKS it checks the Asqav signature and signing key offline. For the full
Asqav verdict, anchors included, run `asqav.verify_receipt_offline` on the
Asqav receipt.

## What the link proves

- The Ed25519 authorization AGT checks is made with `authorizer_key_hex`, a key
  you hold. Asqav's service signs its own ML-DSA-65 receipt and does not sign
  AGT's bytes, so the independent evidence is the Asqav receipt that
  `authorizer_id` names. Run the signer in a separate service from the agent so
  the agent process never holds the authorizer key.
- Whoever holds the authorizer key can re-authorize a receipt with a new nonce
  or expiry, and AGT's check still passes. `verify_asqav_link` fails for it,
  because the Asqav receipt binds the original nonce and expiry.
- Signing fails closed. A receipt AGT rejects, an Asqav error, a `deny` or
  `rate_limit` decision from your Asqav policy, or an Asqav receipt that does
  not bind the AGT receipt raises `AgtSigningError`. As an external authorizer,
  that also stops AGT's adapter from storing the receipt.
- With no Asqav policy for `agt:countersign` or `agt:authorize`, Asqav records
  the receipt as an observation instead of a policy decision.
- In hash-only mode (the default against the Asqav cloud) only a digest of the
  link context leaves your network.
- The nonce is a correlation value. Replay prevention across sessions needs
  durable nonce tracking on your side, as AGT's README notes.
