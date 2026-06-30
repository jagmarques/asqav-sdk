"""10-minute quickstart: govern() -> agent.sign() -> verify.

Run:
    ASQAV_API_KEY=sk_... python quickstart.py
"""

import os

import asqav

# One call: init + Agent.create
agent = asqav.govern(
    api_key=os.environ.get("ASQAV_API_KEY"),
    agent_name="quickstart-agent",
)

print(f"Agent created: {agent.agent_id} ({agent.name})")

# Sign an action
sig = agent.sign("api:call", {"model": "gpt-4", "tokens": 512})

print(f"Signed: {sig.signature_id}")
print(f"Verify: {sig.verification_url}")
