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

asqav verify sig_abc123
asqav agents list
asqav agents create my-agent
asqav sync
```

## Local mode

Sign actions offline when the API is unreachable. Queue syncs when connectivity returns.

```python
from asqav import local_sign

local_sign("agt_xxx", "task:complete", {"result": "done"})
# Later: asqav sync
```

## Works with your stack

Native integrations for 5 frameworks. Each extends `AsqavAdapter` for version-resilient signing.

```bash
pip install asqav[langchain]
pip install asqav[crewai]
pip install asqav[litellm]
pip install asqav[haystack]
pip install asqav[openai-agents]
pip install asqav[all]
```

### LangChain

```python
from asqav.extras.langchain import AsqavCallbackHandler

handler = AsqavCallbackHandler(api_key="sk_...")
chain.invoke(input, config={"callbacks": [handler]})
```

### CrewAI

```python
from asqav.extras.crewai import AsqavCrewHook

hook = AsqavCrewHook(api_key="sk_...")
task = Task(description="Research competitors", callbacks=[hook.task_callback])
```

### LiteLLM / Haystack / OpenAI Agents SDK

```python
from asqav.extras.litellm import AsqavGuardrail
from asqav.extras.haystack import AsqavComponent
from asqav.extras.openai_agents import AsqavGuardrail
```

See [integration docs](https://asqav.com/docs/integrations) for full setup guides.

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

## Pre-flight checks

Combine revocation, suspension, and policy checks in a single call before running sensitive actions:

```python
result = agent.preflight("data:delete")
if not result.cleared:
    raise RuntimeError(f"blocked: {result.reasons}")
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
- **CLI** - verify signatures, manage agents, sync offline queue from the terminal
- **Local mode** - sign actions offline, sync later
- **Framework integrations** - LangChain, CrewAI, LiteLLM, Haystack, OpenAI Agents SDK
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

## In the press

- [Help Net Security: Asqav - Open-source SDK for AI agent governance](https://www.helpnetsecurity.com/2026/04/09/asqav-ai-agent-audit-trail/)

## Ecosystem

| Package | What it does |
|---------|-------------|
| [asqav](https://pypi.org/project/asqav/) | Python SDK |
| [asqav-mcp](https://github.com/jagmarques/asqav-mcp) | MCP server for Claude Desktop/Code |
| [asqav-compliance](https://github.com/jagmarques/asqav-compliance) | CI/CD compliance scanner |

## Free tier

Get started at no cost. Free tier includes agent creation, signed actions, three-tier enforcement, policies, audit export, and framework integrations. Threat detection and monitoring on Pro ($39/mo). Compliance reports, multi-party signing, and incident management on Business ($149/mo). See [asqav.com](https://asqav.com) for pricing.

## Links

- [Integration Docs](https://asqav.com/docs/integrations)
- [Blog](https://dev.to/jagmarques)
- [Dashboard](https://asqav.com/dashboard)
- [PyPI](https://pypi.org/project/asqav/)

## License

MIT - see [LICENSE](LICENSE) for details.

---

If asqav helps you, consider giving it a star. It helps others find the project.
