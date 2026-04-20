<p align="center">
  <a href="https://asqav.com">
    <img src="https://asqav.com/logo-text-white.png" alt="asqav" width="200">
  </a>
</p>
<p align="center">
  Governance for AI agents. Audit trails, policy enforcement, and compliance.
</p>
<p align="center">
  <a href="https://pypi.org/project/asqav/"><img src="https://img.shields.io/pypi/v/asqav?style=flat-square&logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="https://github.com/jagmarques/asqav-sdk/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/jagmarques/asqav-sdk/ci.yml?style=flat-square&logo=github&logoColor=white" alt="CI"></a>
  <a href="https://pypi.org/project/asqav/"><img src="https://img.shields.io/pypi/dm/asqav?style=flat-square&logo=pypi&logoColor=white" alt="Downloads"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square&logo=opensourceinitiative&logoColor=white" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/pypi/pyversions/asqav?style=flat-square&logo=python&logoColor=white" alt="Python versions"></a>
  <a href="https://github.com/jagmarques/asqav-sdk"><img src="https://img.shields.io/github/stars/jagmarques/asqav-sdk?style=social" alt="GitHub stars"></a>
  <a href="https://codespaces.new/jagmarques/asqav-sdk?quickstart=1"><img src="https://img.shields.io/badge/Try_it-Open_in_Codespaces-black?style=flat-square&logo=github" alt="Open in Codespaces"></a>
</p>
<p align="center">
  <a href="https://asqav.com">Website</a> |
  <a href="https://asqav.com/docs">Docs</a> | <a href="README-ZH.md">中文文档</a> |
  <a href="https://asqav.com/docs/sdk">SDK Guide</a> |
  <a href="https://asqav.com/compliance">Compliance</a>
</p>

# asqav SDK

Thin Python SDK for [asqav.com](https://asqav.com). All ML-DSA cryptography runs server-side. Zero native dependencies.

## Install

```bash
pip install asqav
```

```python
import asqav

asqav.init(api_key="sk_...")
agent = asqav.Agent.create("my-agent")
sig = agent.sign("api:call", {"model": "gpt-4"})
```

Your agent now has a cryptographic identity, a signed audit trail, and a verifiable action record.

Each signed action returns:
```json
{
  "signature_id": "sig_a1b2c3",
  "agent_id": "agt_x7y8z9",
  "action": "api:call",
  "algorithm": "ML-DSA-65",
  "timestamp": "2026-04-06T18:30:00Z",
  "chain_hash": "b94d27b9934d3e08...",
  "verify_url": "https://asqav.com/verify/sig_a1b2c3"
}
```

## Why

| Without governance | With asqav |
|---|---|
| No record of what agents did | Every action signed with ML-DSA (FIPS 204) |
| Any agent can do anything | Policies block dangerous actions in real-time |
| One person approves everything | Multi-party authorization for critical actions |
| Manual compliance reports | Automated EU AI Act and DORA reports |
| Breaks when quantum computers arrive | Quantum-safe from day one |

## Decorators and context managers

```python
@asqav.sign
def call_model(prompt: str):
    return openai.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])

with asqav.session() as s:
    s.sign("step:fetch", {"source": "api"})
    s.sign("step:process", {"records": 150})
```

## Async support

```python
agent = await asqav.AsyncAgent.create("my-agent")
sig = await agent.sign("api:call", {"model": "gpt-4"})
```

All API calls retry automatically with exponential backoff on transient failures.

## CLI

```bash
pip install asqav[cli]

asqav demo                    # local dashboard on http://localhost:3030, 4 scenarios
asqav quickstart              # one-command project setup
asqav doctor                  # health check for your configuration
asqav verify sig_abc123
asqav agents list
asqav agents create my-agent
asqav sync
```

### `asqav demo`

Opens a local governance dashboard at `http://localhost:3030` with 4 pre-loaded scenarios: `rm -rf`, a fintech wire transfer, a kubernetes scale-to-zero on production, and a clinical CT-with-contrast order. No signup, no Docker, no API key. Each decision produces an HMAC-signed receipt that the dashboard re-verifies in the browser.

Use it to preview the approval card UX and signed-receipt flow before wiring asqav into your own agent.

## Local mode

Sign actions offline when the API is unreachable. Queue syncs when connectivity returns.

```python
from asqav import local_sign

local_sign("agt_xxx", "task:complete", {"result": "done"})
# Later: asqav sync
```

## Batched signing

Sign up to 100 actions in one HTTP roundtrip. Each action passes through the full single-sign pipeline (policy, scan, anchoring), so one bad action does not abort the batch. Returns a list parallel to the input - successful items are SignatureResponse, failed items are None.

```python
import asqav

asqav.init(api_key="sk_live_...")
agent = asqav.Agent.get("agt_xxx")

results = agent.sign_batch([
    {"action_type": "tool:call", "context": {"name": "search"}},
    {"action_type": "memory:write", "context": {"size": 128}},
    {"action_type": "llm:response", "context": {"model": "gpt-4o"}},
])

for r in results:
    if r is None:
        continue
    print(r.signature_id, r.verification_url)
```

## Works with your stack

Native integrations for 9 frameworks. Each extends `AsqavAdapter` for version-resilient signing.

```bash
pip install asqav[langchain]
pip install asqav[crewai]
pip install asqav[litellm]
pip install asqav[haystack]
pip install asqav[openai-agents]
pip install asqav[llamaindex]
pip install asqav[smolagents]
pip install asqav[dspy]
pip install asqav-pydantic          # PydanticAI (external package)
pip install asqav[all]
```

### LangChain

```python
from asqav.extras.langchain import AsqavCallbackHandler

handler = AsqavCallbackHandler(api_key="sk_...")
chain.invoke(input, config={"callbacks": [handler]})

# Observe mode - test governance without API calls
handler = AsqavCallbackHandler(api_key="sk_...", observe=True)
```

### CrewAI

```python
from asqav.extras.crewai import AsqavCrewHook

hook = AsqavCrewHook(api_key="sk_...")
task = Task(description="Research competitors", callbacks=[hook.task_callback])
```

```python
from asqav.extras.crewai import AsqavGuardrailProvider
provider = AsqavGuardrailProvider(agent_name="crew-guard")
# Implements the CrewAI GuardrailProvider interface
```

### LlamaIndex

```python
from asqav.extras.llamaindex import AsqavLlamaIndexHandler

handler = AsqavLlamaIndexHandler(api_key="sk_...")
```

### LiteLLM / Haystack / OpenAI Agents SDK / smolagents / DSPy

```python
from asqav.extras.litellm import AsqavGuardrail
from asqav.extras.haystack import AsqavComponent
from asqav.extras.openai_agents import AsqavGuardrail
from asqav.extras.smolagents import AsqavSmolagentsHook
from asqav.extras.dspy import AsqavDSPyCallback
```

### PydanticAI

```bash
pip install asqav-pydantic
```

```python
from asqav_pydantic import AsqavHooks
```

See [integration docs](https://asqav.com/docs/integrations) for full setup guides.

## Three-phase signing

Break high-stakes actions into intent, decision, and execution phases. Each phase produces a signed receipt, and execution only proceeds if policy allows it.

```python
# Three-phase signing: intent, decision, execution
chain = agent.sign_with_phases("data:delete", {"table": "users"})
print(chain.approved)    # True if policy allowed it
print(chain.intent)      # signed intent receipt
print(chain.decision)    # policy evaluation result
print(chain.execution)   # signed execution receipt (only if approved)
```

## Trace correlation

Link actions across multi-step workflows with a shared trace ID. Useful for debugging distributed agent pipelines and correlating audit trails across services.

```python
# Link actions across multi-step workflows
trace_id = asqav.generate_trace_id()
agent.sign("step:1", ctx, trace_id=trace_id)
agent.sign("step:2", ctx, trace_id=trace_id, parent_id="sig_prev")
```

## Emergency halt

Kill switch that revokes all agents in your organization immediately. Use during security incidents to stop all agent activity until the situation is resolved.

```python
# Kill switch: revoke all agents immediately
asqav.emergency_halt(reason="security incident")
```

## Enforcement

asqav provides three tiers of enforcement for AI agent governance:

**Strong** - the [asqav-mcp](https://github.com/jagmarques/asqav-mcp) server acts as a non-bypassable tool proxy. Agents call tools through the MCP server, which checks policy and signs the decision before allowing execution. With a `tool_endpoint` configured, the server forwards the call and signs request + response together as a bilateral receipt.

**Bounded** - pre-execution gates (`gate_action`) check policy and sign the decision before the agent acts. After completing the action, the agent calls `complete_action` to close the bilateral receipt, linking the approval to the outcome.

**Detectable** - every action gets a quantum-safe signature (ML-DSA-65) hash-chained to the previous one. If logs are tampered with or entries omitted, the chain breaks and verification fails.

Most teams use all three tiers for different tools. High-risk mutations go through the strong path, routine operations use bounded, and everything gets the detectable layer. Bilateral receipts ensure auditors can verify both what was authorized and what actually happened.

### Policies

```python
asqav.create_policy(
    name="no-deletions",
    action_pattern="data:delete:*",
    action="block_and_alert",
    severity="critical"
)
```

### MCP enforcement

```bash
pip install asqav-mcp
export ASQAV_PROXY_TOOLS='{"sql:execute": {"risk_level": "high", "require_approval": true}}'
asqav-mcp
```

The MCP server exposes `gate_action` (bounded) and `enforced_tool_call` (strong) tools to any MCP client - Claude Desktop, Claude Code, Cursor, or custom agents.

## Multi-party signing

Distributed approval where no single entity can authorize alone:

```python
config = asqav.create_signing_group("agt_xxx", min_approvals=2, total_shares=3)
session = asqav.request_action("agt_xxx", "finance.transfer", {"amount": 50000})
asqav.approve_action(session.session_id, "ent_xxx")
```

## Semantic action patterns

Use descriptive labels instead of glob patterns for clearer policies:

```python
sig = agent.sign("sql-destructive", {"query": "DROP TABLE users"})
```

## Pre-flight checks

Combine revocation, suspension, and policy checks in a single call before running sensitive actions. Returns human-readable explanations of what each policy would do:

```python
result = agent.preflight("data:delete")
if not result.cleared:
    raise RuntimeError(f"blocked: {result.reasons}")
# result.explanations contains plain-English descriptions of each check
```

## Reproducibility metadata

Attach system prompt hashes and other reproducibility context to signatures:

```python
sig = agent.sign("llm:call", ctx, system_prompt_hash="sha256:a1b2c3...")
```

## Compliance bundle export

Export signed action trails as regulation-specific compliance bundles:

```python
bundle = asqav.export_bundle(signatures, "eu_ai_act_art12")
```

## Scope tokens

Portable, short-lived tokens that let a receiving service verify what an agent is allowed to do without calling your org:

```python
token = agent.create_scope_token(
    actions=["data:read:customer"],
    ttl=3600
)
# Attach to outgoing API calls
requests.get(url, headers=token.to_header())
# Receiving service verifies without calling your org
verified = asqav.verify_scope_token(token_string)
```

## Replay protection

Each scope token includes a unique nonce. Receivers should track used nonces and reject duplicates to prevent replay attacks.

```python
token = agent.create_scope_token(actions=["data:read"], ttl=3600)
seen = set()
if token.nonce in seen:
    reject()
seen.add(token.nonce)
```

## Replay

Reconstruct what an agent did in a session, with chain integrity verification:

```python
timeline = asqav.replay(agent_id="agt_abc", session_id="sess_xyz")
print(timeline.summary())
assert timeline.chain_integrity  # verify no tampering

# Or replay from offline bundle
timeline = asqav.replay_from_bundle(bundle)
```

## Output verification

Bind a tool output to its input and verify later that nothing was tampered with:

```python
import hashlib, json

prompt = {"query": "top customers"}
input_hash = hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()

output = run_tool(prompt)
sig = agent.sign_output("tool:search", input_hash, output)

# Later, from any verifier
result = asqav.verify_output(sig.signature_id, output)
assert result["verified"]
```

## Governance fields on signatures

Every `SignatureResponse` carries the governance context that produced it, so a third party can reconstruct the exact JCS bytes that were signed and re-verify offline:

```python
sig = agent.sign("tool:call", {"name": "search"})

sig.policy_digest       # sha256 of the policy attestation snapshot
sig.policy_decision     # "permit" or "deny"
sig.authorization_ref   # approval_id when a HITL approval authorized the action, else None
```

`policy_decision` defaults to `"permit"`. When an action is denied, the API raises and you get a signed deny envelope through the standard error path.

## Budget tracking

Client-side spend tracker that persists every cost as a signed record. Fails closed when the limit would be exceeded:

```python
budget = asqav.BudgetTracker(agent, limit=10.0, currency="USD")

decision = budget.check(estimated_cost=0.25)
if not decision.allowed:
    raise RuntimeError(decision.reason)

budget.record("api:openai", actual_cost=0.23, context={"model": "gpt-4"})
```

## Attestations

Generate a portable, signed attestation document proving an agent's governance status. Share it with auditors or partners for independent verification:

```python
attestation = asqav.generate_attestation("agt_abc123", session_id="sess_xyz")

# Anyone with the document can verify it
result = asqav.verify_attestation(attestation)
assert result["valid"] and result["all_valid"]
```

## Features

- **Signed actions** - every agent action gets a ML-DSA-65 signature with RFC 3161 timestamp
- **Decorators** - `@asqav.sign` wraps any function with cryptographic signing
- **Async** - full async support with `AsyncAgent` and automatic retry
- **CLI** - verify signatures, manage agents, quickstart setup, doctor health check
- **Local mode** - sign actions offline, sync later
- **Observe mode** - test governance without API calls using `observe=True`
- **Framework integrations** - LangChain, CrewAI, LiteLLM, Haystack, OpenAI Agents SDK, LlamaIndex, smolagents, DSPy, PydanticAI
- **Semantic action patterns** - descriptive labels instead of glob patterns
- **Reproducibility metadata** - attach system prompt hashes to signatures
- **Compliance bundles** - export regulation-specific bundles with `export_bundle`
- **Three-tier enforcement** - strong (tool proxy), bounded (pre-execution gates), detectable (signed audit trail)
- **Policy enforcement** - block or alert on action patterns before execution
- **Multi-party signing** - m-of-n approval using threshold ML-DSA
- **Pre-flight checks** - combined revocation, suspension, and policy gate in one call
- **Output verification** - bind tool outputs to inputs and verify later with `sign_output` / `verify_output`
- **Budget tracking** - client-side spend limits with signed, replayable records
- **Attestations** - portable signed documents proving an agent's governance status
- **Agent identity** - create, suspend, revoke, and rotate agent keys
- **Audit export** - JSON/CSV trails for compliance reporting
- **Tokens** - scoped JWTs and selective-disclosure tokens (SD-JWT)
- **Scope tokens** - portable, short-lived tokens for cross-org action verification
- **Replay** - reconstruct agent sessions with chain integrity checks
- **Three-phase signing** - intent, decision, and execution phases with signed receipts
- **Trace correlation** - link actions across multi-step workflows with shared trace IDs
- **Emergency halt** - kill switch to revoke all agents immediately during incidents
- **CrewAI GuardrailProvider** - native CrewAI guardrail interface integration

## In the press

- [Help Net Security: Asqav - Open-source SDK for AI agent governance](https://www.helpnetsecurity.com/2026/04/09/asqav-ai-agent-audit-trail/)

## Ecosystem

| Package | What it does |
|---------|-------------|
| [asqav](https://pypi.org/project/asqav/) | Python SDK |
| [asqav-mcp](https://github.com/jagmarques/asqav-mcp) | MCP server for Claude Desktop/Code |
| [asqav-compliance](https://github.com/jagmarques/asqav-compliance) | CI/CD compliance scanner |

## Free tier

Get started at no cost. Free includes 50K signatures/month, 20 agents, 10 policies, 3 team members, OpenTimestamps, MCP server, audit export, and framework integrations. Content scanning, OpenTelemetry, managed KMS, and Replay API on Pro ($19/mo annual). Quarantine, incident management, multi-party signing, compliance reports, and RFC 3161 timestamps on Business ($79/mo annual). See [asqav.com/pricing](https://asqav.com/pricing.html) for the full breakdown.

## Discovery

Machine-readable service descriptor at [`https://asqav.com/.well-known/governance.json`](https://asqav.com/.well-known/governance.json) for external tools that want to auto-discover asqav's endpoints, algorithms, capabilities, and integrations.

## Links

- [Integration Docs](https://asqav.com/docs/integrations)
- [Blog](https://dev.to/jagmarques)
- [Dashboard](https://asqav.com/dashboard)
- [PyPI](https://pypi.org/project/asqav/)

## License

MIT - see [LICENSE](LICENSE) for details.

---

If asqav helps you, consider giving it a star. It helps others find the project.
