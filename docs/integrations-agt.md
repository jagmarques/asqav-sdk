# Microsoft Agent Governance Toolkit: Asqav as an independent witness

Microsoft's Agent Governance Toolkit (AGT) can sign a receipt for every MCP tool
call it governs. Each receipt records the Cedar decision for one call and is
signed with Ed25519, chained to the receipt before it. The signing key lives in
the same process that made the decision, and AGT's receipts proposal says what
that means:

> This is **self-attestation**: it detects later tampering but does not prove
> that an independent party evaluated the action.

(AGT, `docs/proposals/verifiable-compliance-receipts.md`)

The `asqav.agt_receipts` module records each AGT receipt in an Asqav receipt and
links the two. The Asqav receipt is signed and timestamped by the Asqav service,
under your org's Asqav key, outside the agent's process. Asqav does not evaluate
the Cedar decision or the tool arguments; it witnesses the receipt.

## What the Asqav receipt proves

It proves that an AGT receipt with this exact payload digest existed when Asqav
signed, and that your Asqav org recorded it then.

It does not stop the operator from rewriting history. Whoever runs the agent
holds the AGT key and the Asqav API key, so they can rewrite a receipt, re-sign
it with AGT and countersign it again, possibly under a different Asqav agent in
the same org. The first Asqav receipt stays on record. In hash-only mode Asqav
holds only a digest, so no Asqav record names the AGT receipt. An auditor finds
a rewrite by matching every Asqav receipt in the org against the exported AGT
receipts; one that matches none is a rewritten or dropped receipt.

It also does not show that the Cedar decision was right or that the tool call
happened, and a receipt that was already false at its first countersign stays
false. Hash-only mode is the default against the Asqav cloud. The Asqav key is
held by the Asqav service, not in a hardware security module.

## Versions

- The `agt` extra pins `agent-governance-toolkit-protocols==5.0.0`, the PyPI
  release that ships AGT's `mcp_receipt_governed` module. AGT is MIT licensed.
- That release has no external-authorization hook. AGT added one on its main
  branch (`authorize_receipt`, `external_authorizer`, `trusted_authorizer_keys`)
  after 5.0.0. It has not reached PyPI yet. This module was tested against AGT
  commit `c767f8333f413bd1a93e82836b7aa226280b61ae` for that path.
- AGT needs Python 3.11 or later, so the extra installs nothing on 3.10.

AGT's own verifier references the Asqav receipt only on AGT builds that ship the
external-authorization profile, which is unreleased as of commit `c767f83`. On
the PyPI 5.0.0 release AGT's verifier never sees the Asqav receipt, and the
link is checked by `verify_asqav_link` alone.

## Install

```bash
pip install "asqav[agt]"
```

Add `asqav[verify]` as well; `verify_asqav_link` checks Asqav signatures with it.

AGT 5.0.0 requires `cryptography<50`, so this install downgrades `cryptography`
to 49.0.0 in the whole environment. That release is affected by
GHSA-g6cj-pr64-35w5 (CVE-2026-69247), a padding oracle in PKCS#7 EnvelopedData
decryption that 50.0.0 fixes. Neither this module nor AGT's receipt code calls
PKCS#7 decryption, but other code in the same environment might. Pip
constraints cannot lift an upper bound, so either install the extra in its own
environment or override the pin with uv; this module's tests pass on 50.0.0:

```bash
echo "cryptography>=50.0.0" > overrides.txt
uv pip install --override overrides.txt "asqav[agt]"
```

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
rejects. The Asqav receipt binds the AGT receipt id, its payload hash, the AGT
signer key and the Cedar decision. Store `link.asqav_receipt` (a JSON-safe dict)
next to AGT's exported receipts; `signer.links` keeps every link by AGT receipt id.

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

For each tool call the signer gets an Asqav receipt over the AGT receipt and
the authorization it is about to make (its nonce, expiry and public key). It then calls AGT's
`authorize_receipt` with `authorizer_id` set to `asqav:<signature_id>`. AGT's
signed authorization payload covers `authorizer_id`, so any edit to it breaks
AGT's check.

An authorization needs an `allow` from Asqav, so register an Asqav policy that
allows `agt:authorize`. Without that policy Asqav only records an observation.
The signer then refuses to authorize.

The Ed25519 authorization AGT checks is made with `authorizer_key_hex`, a key
you hold, so AGT's `require_external_authorization=True` passes without
consulting Asqav. The Asqav receipt that `authorizer_id` names is the
independent evidence, and only `verify_asqav_link` checks it. Run the signer in
a separate service from the agent so the agent process never holds that key.

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
    errors += verify_asqav_link(
        receipt, asqav_receipts[receipt.receipt_id], jwks=jwks, expected_org=ASQAV_ORG_ID
    )
assert errors == []
```

Drop `trusted_authorizer_keys` and `require_external_authorization` for
countersigned receipts; AGT 5.0.0 does not accept them.

`verify_asqav_link` first checks the Asqav signature and signing key against
the JWKS, and refuses when no JWKS is given. It then recomputes the `action_ref`
from the AGT receipt and compares it with the signed one. The signed decision
must fit the link: `allow` for an authorization, `allow` or `observation` for a
countersign. `expected_org` pins the Asqav org id. It is read from the signed
`org_id`, because `issuer_id` carries the org's legal entity when one is set,
and a receipt from another org fails. For an authorized receipt it also checks that
`authorizer_id` names the Asqav receipt; that id sits outside the signed
payload, and the cryptographic binding is the `action_ref`. For the full Asqav
verdict, anchors included, run `asqav.verify_receipt_offline` on the Asqav
receipt.

## Behaviour notes

- Whoever holds the authorizer key can re-authorize a receipt with a new nonce
  or expiry, and AGT's check still passes. `verify_asqav_link` fails for it,
  because the Asqav receipt binds the original nonce and expiry.
- Signing fails closed. `countersign` and the authorizer hook raise
  `AgtSigningError` when AGT rejects the receipt or Asqav errors. They also
  raise when the signed Asqav receipt has the wrong decision or does not bind
  the AGT receipt. As an external authorizer, that stops AGT's adapter from
  storing the receipt.
- With no Asqav policy for `agt:countersign`, Asqav records the countersign as
  an observation, which `countersign` accepts.
- The nonce is a correlation value. Replay prevention across sessions needs
  durable nonce tracking on your side, as AGT's README notes.
