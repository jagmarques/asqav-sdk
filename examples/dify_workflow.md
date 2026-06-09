# Dify workflow with asqav signing

Closes #50.

The [asqav Dify plugin](https://github.com/jagmarques/dify-plugin-asqav) wraps
the same backend API as this SDK. A Dify workflow can sign agent actions and
later verify them; the SDK and CLI produce and read the exact same receipts,
so any pipeline is portable across runtimes.

## Install the plugin in Dify

1. Open the Dify Plugin Marketplace, or import from URL.
2. Add the asqav plugin.
3. Open **Plugin Settings** and enter your credentials:
   - **asqav API Key**, in the form `sk_...` - get one at <https://cloud.asqav.com>
   - **Agent ID**, in the form `agent_...` - create with `asqav agents create my-agent`
     or via the dashboard.

## Wire up a workflow

A typical workflow puts the Sign Action tool right after the LLM step that
decides what to do, and the Verify Signature tool wherever a downstream step
or auditor needs to confirm the action ran as recorded.

```
[User input] -> [LLM: choose tool] -> [Sign Action] -> [Execute tool] -> [Output]
                                       |
                                       +-> writes signature_id to workflow vars
```

For high-risk paths, swap **Sign Action** for **Request Action**, gate the
**Execute tool** node on the session reaching `status=approved`, and only then
proceed.

## What the audit trail looks like

Each Sign Action call creates one record on the asqav backend:

```json
{
  "signature_id": "sig_a1b2c3",
  "agent_id": "agent_x7y8z9",
  "action_type": "tool:web-search",
  "algorithm": "ML-DSA-65",
  "timestamp": 1730405400.0,
  "verification_url": "https://asqav.com/verify/sig_a1b2c3",
  "chain_hash": "b94d27b9934d3e08..."
}
```

Anyone with the `signature_id` can verify it without an API key, either at the
URL above or programmatically.

## Verify a Dify-generated signature from the SDK

The receipts produced by the Dify plugin are the same shape the SDK reads, so
the same verification path works from any Python or shell environment:

```python
import asqav

result = asqav.verify("sig_a1b2c3")
assert result.verified
print(result.agent_name, result.action_type)
```

```bash
asqav verify sig_a1b2c3
```

## Bundle a workflow run for an auditor

Group every signature from a Dify workflow run under one `session_id`. The
plugin accepts `context.session_id`, or you can use the same value across all
Sign Action calls. Then export the bundle from the SDK or CLI:

```bash
asqav compliance export --session sess_dify_run_42 --output run-42.json
asqav replay --bundle run-42.json
```

That produces a self-contained file the auditor can verify offline. The
underlying Python API is `get_session_signatures` plus
`asqav.compliance.export_bundle`; see [`examples/quarterly_audit.py`](quarterly_audit.py)
for the programmatic equivalent.

## Limits

- The plugin runs ML-DSA-65 only. The SDK itself supports ML-DSA-44 and
  ML-DSA-87 if you need a different post-quantum security level.
- Bulk batch signing is not exposed by the plugin yet. Use the SDK's
  `agent.sign_batch(...)` for high-volume runs.
- The plugin does not yet drive the multi-party countersigning flow
  via `co_signers`. For that, call `agent.sign(..., co_signers=[...])` from the
  SDK.

## Links

- Plugin source: <https://github.com/jagmarques/dify-plugin-asqav>
- SDK reference: <https://asqav.com/docs/sdk>
- Quickstart: `asqav quickstart`
