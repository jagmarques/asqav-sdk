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
| Threshold config management | Threshold ML-DSA (Shamir over Rq) |
| Risk rule management | Risk classification engine |
| Delegation management | Key lifecycle (refresh, recovery) |

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

### Threshold ML-DSA Signing (Business+)

Distributed signing where no single entity can authorize alone.

```python
import asqav

asqav.init(api_key="sk_...")

# 1. Create threshold config (2-of-3)
config = asqav.create_threshold_config("agt_xxx", threshold=2, total_shares=3)

# 2. Add signing entities
asqav.add_entity(config.id, entity_class="A", label="agent-signer")
asqav.add_entity(config.id, entity_class="B", label="human-operator")
asqav.add_entity(config.id, entity_class="C", label="policy-engine")

# 3. Generate threshold keypair
keypair = asqav.generate_keypair(config.id)

# 4. Request action approval
session = asqav.request_action("agt_xxx", "finance.transfer", {"amount": 50000})

# 5. Collect approvals
asqav.approve_action(session.session_id, "ent_xxx")

# 6. Threshold sign (standard ML-DSA-65 output)
result = asqav.threshold_sign(keypair.id, message_hex="deadbeef...")
```

**Entity Classes:**

| Class | Role | Description |
|-------|------|-------------|
| A | Agent Signer | AI agent requesting authorization |
| B | Human Operator | Human-in-the-loop approval |
| C | Policy Engine | Automated policy enforcement |
| D | Compliance Verifier | Regulatory compliance validation |
| E | Organizational Authority | Executive-level approval |

### Risk Rules

```python
# Dynamic threshold based on action risk
rule = asqav.create_risk_rule(
    name="High-risk finance",
    action_pattern="finance.*",
    risk_level="critical",
    threshold_override=3,
    entity_weights={"D": 2.0},  # Compliance counts double
)
```

### Key Lifecycle

```python
# Rotate shares (same public key)
asqav.refresh_keypair(keypair.id)

# Recover lost share
asqav.recover_share(keypair.id, entity_id="ent_lost", contributing_entity_ids=["ent_1", "ent_2"])

# Temporary delegation
asqav.create_delegation(config.id, delegator_entity_id="ent_alice", delegate_entity_id="ent_bob", expires_in=3600)
```

### Threshold

```python
# Config
config = asqav.create_threshold_config(agent_id, threshold, total_shares)
config = asqav.get_threshold_config(agent_id)
config = asqav.update_threshold_config(config_id, threshold=3)

# Entities
entity = asqav.add_entity(config_id, entity_class="B", label="operator")
entities = asqav.list_entities(config_id)
asqav.remove_entity(entity_id)

# Keypairs
keypair = asqav.generate_keypair(config_id)
keypair = asqav.get_keypair(keypair_id)
result = asqav.threshold_sign(keypair_id, message_hex="...")
asqav.refresh_keypair(keypair_id)
asqav.recover_share(keypair_id, entity_id, contributing_entity_ids)

# Sessions
session = asqav.request_action(agent_id, action_type, params)
result = asqav.approve_action(session_id, entity_id)
status = asqav.get_action_status(session_id)
sessions = asqav.list_sessions(status="pending")

# Risk Rules
rule = asqav.create_risk_rule(name, action_pattern, risk_level, ...)
rules = asqav.list_risk_rules()
asqav.update_risk_rule(rule_id, threshold_override=3)
asqav.delete_risk_rule(rule_id)

# Delegations
delegation = asqav.create_delegation(config_id, delegator_id, delegate_id)
delegations = asqav.list_delegations()
asqav.revoke_delegation(delegation_id)
```

## Requirements

- Python 3.10+

## Get your API key

Sign up at [asqav.com](https://asqav.com)

## License

MIT
