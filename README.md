# asqav

[![PyPI version](https://img.shields.io/pypi/v/asqav)](https://pypi.org/project/asqav/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/downloads/)

**AI agent governance simplified. Audit trails, policy enforcement, and compliance in 5 lines.**

Thin Python SDK for [asqav.com](https://asqav.com) - all ML-DSA cryptography runs server-side. Zero native dependencies.

## Install

```bash
pip install asqav
```

## Quick start

```python
import asqav

asqav.init(api_key="sk_...")
agent = asqav.Agent.create("my-agent")
sig = agent.sign("api:call", {"model": "gpt-4"})
```

That's it. Your agent now has a quantum-safe identity, a signed audit trail, and a verifiable action record.

## Features

- **Agent identity** - create, suspend, revoke, and rotate agent keys
- **Signed actions** - every agent action gets a ML-DSA-65 signature
- **Audit export** - export trails as JSON or CSV for compliance reporting
- **Tokens** - issue scoped JWTs and selective-disclosure tokens (SD-JWT)
- **Tracing** - built-in spans with OpenTelemetry export
- **Multi-party signing** - m-of-n approval using Shamir over Rq (no single point of authority)
- **Risk rules** - dynamic approval thresholds based on action patterns
- **Delegations** - temporary authority transfer between entities
- **Decorators** - `@asqav.secure` wraps any function with cryptographic signing

## Framework integrations

Works with any Python AI framework. Drop `asqav` into your existing stack:

| Framework | How |
|-----------|-----|
| **LangChain** | Wrap tool calls with `agent.sign()` or `@asqav.secure` |
| **CrewAI** | Sign crew task outputs for audit compliance |
| **MCP** | Sign MCP tool invocations before execution |

```python
# Example: signing a LangChain tool call
@asqav.secure
def call_tool(query: str):
    return my_langchain_tool.run(query)
```

## Tracing

```python
with asqav.span("api:openai", {"model": "gpt-4"}) as s:
    response = openai.chat.completions.create(...)
    s.set_attribute("tokens", response.usage.total_tokens)
```

## Multi-party signing

Distributed approval where no single entity can authorize alone.

```python
config = asqav.create_signing_group("agt_xxx", min_approvals=2, total_shares=3)
asqav.add_entity(config.id, entity_class="B", label="human-operator")
session = asqav.request_action("agt_xxx", "finance.transfer", {"amount": 50000})
asqav.approve_action(session.session_id, "ent_xxx")
result = asqav.group_sign(keypair.id, message_hex="deadbeef...")
```

## Free tier

Get started at no cost. The free tier includes agent creation, signed actions, audit export, and tracing. Multi-party signing and SD-JWT tokens are available on the Business plan. See [asqav.com](https://asqav.com) for pricing.

## Links

- [Documentation](https://asqav.com/docs)
- [AI Readiness Assessment](https://asqav.com/readiness)
- [Compliance](https://asqav.com/compliance)
- [PyPI](https://pypi.org/project/asqav/)
- [GitHub](https://github.com/jagmarques/asqav-sdk)

## License

MIT - see [LICENSE](LICENSE) for details.
