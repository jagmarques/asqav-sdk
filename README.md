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
  <a href="https://pypi.org/project/asqav/"><img src="https://img.shields.io/pypi/dm/asqav?style=flat-square&logo=pypi&logoColor=white" alt="Downloads"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square&logo=opensourceinitiative&logoColor=white" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/pypi/pyversions/asqav?style=flat-square&logo=python&logoColor=white" alt="Python versions"></a>
  <a href="https://github.com/jagmarques/asqav-sdk"><img src="https://img.shields.io/github/stars/jagmarques/asqav-sdk?style=social" alt="GitHub stars"></a>
</p>
<p align="center">
  <a href="https://asqav.com">Website</a> |
  <a href="https://asqav.com/docs">Docs</a> |
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

Your agent now has a quantum-safe identity, a signed audit trail, and a verifiable action record.

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

## Policy enforcement

```python
asqav.create_policy(
    name="no-deletions",
    action_pattern="data:delete:*",
    action="block_and_alert",
    severity="critical"
)
```

## Multi-party signing

Distributed approval where no single entity can authorize alone:

```python
config = asqav.create_signing_group("agt_xxx", min_approvals=2, total_shares=3)
session = asqav.request_action("agt_xxx", "finance.transfer", {"amount": 50000})
asqav.approve_action(session.session_id, "ent_xxx")
```

## Features

- **Signed actions** - every agent action gets a ML-DSA-65 signature with RFC 3161 timestamp
- **Decorators** - `@asqav.sign` wraps any function with cryptographic signing
- **Async** - full async support with `AsyncAgent` and automatic retry
- **CLI** - verify signatures, manage agents, sync offline queue from the terminal
- **Local mode** - sign actions offline, sync later
- **Framework integrations** - LangChain, CrewAI, LiteLLM, Haystack, OpenAI Agents SDK
- **Policy enforcement** - block or alert on action patterns before execution
- **Multi-party signing** - m-of-n approval using threshold ML-DSA
- **Agent identity** - create, suspend, revoke, and rotate agent keys
- **Audit export** - JSON/CSV trails for compliance reporting
- **Tokens** - scoped JWTs and selective-disclosure tokens (SD-JWT)

## Ecosystem

| Package | What it does |
|---------|-------------|
| [asqav](https://pypi.org/project/asqav/) | Python SDK |
| [asqav-mcp](https://github.com/jagmarques/asqav-mcp) | MCP server for Claude Desktop/Code |
| [asqav-compliance](https://github.com/jagmarques/asqav-compliance) | CI/CD compliance scanner |

## Free tier

Get started at no cost. Free tier includes agent creation, signed actions, audit export, and framework integrations. Content scanning and monitoring on Pro ($29/mo). Compliance reports and remediation on Business ($99/mo). See [asqav.com](https://asqav.com) for pricing.

## Links

- [Website](https://asqav.com)
- [Documentation](https://asqav.com/docs)
- [SDK Guide](https://asqav.com/docs/sdk)
- [Integration Docs](https://asqav.com/docs/integrations)
- [Blog](https://dev.to/jagmarques)
- [Compliance](https://asqav.com/compliance)
- [Dashboard](https://asqav.com/dashboard)
- [PyPI](https://pypi.org/project/asqav/)
- [GitHub](https://github.com/jagmarques/asqav-sdk)

## License

MIT - see [LICENSE](LICENSE) for details.

---

If asqav helps you, consider giving it a star. It helps others find the project.
