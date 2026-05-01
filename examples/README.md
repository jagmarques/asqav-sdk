# Examples

Self-contained scripts that exercise the asqav governance flows end-to-end.
Every example uses both surfaces of the SDK:

- the Python API directly, for programmatic use inside an application
- the equivalent CLI commands in the docstring, for shell scripts and CI

Pick whichever surface fits your runtime; they accept the same arguments and
return the same data.

## Setup

```bash
pip install asqav
export ASQAV_API_KEY=sk_...
```

## Examples

| File | What it shows | Closes |
|------|---------------|--------|
| [`budget_tracking.py`](budget_tracking.py) | Multi-provider budget ceiling, per-call check, signed spend records | #61 |
| [`scope_enforcement.py`](scope_enforcement.py) | Preflight catches out-of-scope actions, violation gets its own signed audit entry | #62 |
| [`quarterly_audit.py`](quarterly_audit.py) | Export bundle, replay timeline offline, verify SHA-256 chain, summarize | #63 |
| [`human_approval.py`](human_approval.py) | Request signing session for a high-risk action, wait for human approval, then sign | #64 |

## CLI parity

Every flow above has a matching CLI command. Run any of these from a shell:

```bash
asqav preflight <agent_id> data:read           # scope check
asqav budget check --agent-id <id> --limit 10 --estimated-cost 0.25
asqav budget record --agent-id <id> --action api:openai --actual-cost 0.23 --limit 10
asqav approve <session_id> <entity_id>         # human approval
asqav compliance frameworks                    # list known frameworks
asqav compliance export --session <sid> --output bundle.json
asqav replay <agent_id> <session_id>           # online replay
asqav replay --bundle bundle.json              # offline replay
```

See `python/src/asqav/cli.py` for the full command surface; every command
wraps a public function from the same package.
