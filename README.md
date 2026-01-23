# asqav

[![PyPI](https://img.shields.io/pypi/v/asqav)](https://pypi.org/project/asqav/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Thin API client** for [asqav.com](https://asqav.com). All ML-DSA cryptography happens server-side. No native dependencies required.

## Installation

```bash
pip install asqav
```

## Usage

```python
import asqav

# Initialize with your API key (get one at asqav.com)
asqav.init(api_key="sk_...")

# Create an agent
agent = asqav.Agent.create("my-agent")

# Sign an action
sig = agent.sign("api:call", {"model": "gpt-4"})

# Issue a token
token = agent.issue_token(scope=["read", "write"])
```

## What this SDK does

| This SDK | asqav Cloud (server-side) |
|----------|---------------------------|
| API calls | ML-DSA key generation |
| Response parsing | Cryptographic signing |
| Error handling | Token issuance |
| OTEL export | Signature verification |

The SDK is intentionally minimal (~900 lines). All quantum-safe cryptography runs on asqav's servers.

## API Reference

### Initialization

```python
asqav.init(api_key="sk_...")  # or set ASQAV_API_KEY env var
```

### Agent

```python
agent = asqav.Agent.create("name", algorithm="ml-dsa-65")
agent = asqav.Agent.get("agt_xxx")

agent.sign("action", {"key": "value"})
agent.issue_token(scope=["read"], ttl=3600)
agent.issue_sd_token(claims={...}, disclosable=[...])  # Business tier
agent.suspend(reason="investigation", note="...")  # Temporary disable
agent.unsuspend()  # Re-enable suspended agent
agent.revoke(reason="manual")  # Permanent revoke
```

### Tracing

```python
with asqav.span("api:openai", {"model": "gpt-4"}) as s:
    response = openai.chat.completions.create(...)
    s.set_attribute("tokens", response.usage.total_tokens)
```

## Requirements

- Python 3.10+

## Get your API key

Sign up at [asqav.com](https://asqav.com)

## License

MIT
