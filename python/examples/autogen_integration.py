"""
Autogen Multi-Agent Integration Example

This example demonstrates how to use ASQAV with Microsoft Autogen's multi-agent
conversations to provide governance, audit trails, and compliance for AI agents.

Requirements:
    pip install asqav pyautogen

Usage:
    python examples/autogen_integration.py
"""

import os
from autogen import ConversableAgent, UserProxyAgent
import asqav

# Initialize ASQAV
# Get your API key from https://asqav.com
ASQAV_API_KEY = os.getenv("ASQAV_API_KEY", "sk_your_api_key_here")

if ASQAV_API_KEY.startswith("sk_your"):
    print("⚠️  Warning: Please set ASQAV_API_KEY environment variable")
    print("   Get your key at: https://asqav.com")
    DEMO_MODE = True
else:
    asqav.init(api_key=ASQAV_API_KEY)
    DEMO_MODE = False

# Create an ASQAV agent for audit tracking
if not DEMO_MODE:
    audit_agent = asqav.Agent.create("autogen-governance-agent")


def create_governed_agent(name: str, system_message: str, asqav_action_type: str = "agent:message"):
    """
    Create an Autogen agent with ASQAV governance.
    
    Args:
        name: Agent name
        system_message: Agent's system message
        asqav_action_type: Type of action for audit logging
    
    Returns:
        ConversableAgent with ASQAV integration
    """
    agent = ConversableAgent(
        name=name,
        system_message=system_message,
        llm_config={
            "config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]
        }
    )
    
    original_generate_reply = agent.generate_reply
    
    def governed_generate_reply(messages=None, **kwargs):
        """Generate reply with ASQAV audit logging."""
        response = original_generate_reply(messages=messages, **kwargs)
        
        if not DEMO_MODE and response:
            action_data = {
                "agent": name,
                "message_count": len(messages) if messages else 0,
                "response_length": len(response) if isinstance(response, str) else 0
            }
            
            try:
                signature = audit_agent.sign(asqav_action_type, action_data)
                print(f"✅ ASQAV: Action signed for agent '{name}'")
            except Exception as e:
                print(f"⚠️  ASQAV signing failed: {e}")
        
        return response
    
    agent.generate_reply = governed_generate_reply
    return agent


def main():
    """Demonstrate multi-agent conversation with ASQAV governance."""
    
    print("🤖 ASQAV + Autogen Multi-Agent Example")
    print("=" * 50)
    
    researcher = create_governed_agent(
        name="Researcher",
        system_message="You are a research assistant. Find and summarize information."
    )
    
    reviewer = create_governed_agent(
        name="Reviewer", 
        system_message="You are a critical reviewer. Check facts and identify gaps."
    )
    
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3
    )
    
    print("\n📝 Starting multi-agent conversation...")
    print("-" * 50)
    
    user_proxy.initiate_chat(
        researcher,
        message="Research the benefits and risks of AI governance frameworks."
    )
    
    print("\n" + "=" * 50)
    if DEMO_MODE:
        print("✅ Demo completed (no ASQAV signing - API key not set)")
    else:
        print("✅ Multi-agent conversation completed with ASQAV governance")
        print("   View audit trail at: https://asqav.com/dashboard")


if __name__ == "__main__":
    main()
