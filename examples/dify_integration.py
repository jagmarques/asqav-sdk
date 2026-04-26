"""
Dify Plugin Integration Example

This example demonstrates how to use Asqav with Dify workflows to provide
governance, audit trails, and ML-DSA signing for AI agents in Dify.

The Dify plugin is available at: https://github.com/jagmarques/dify-plugin-asqav

Requirements:
    pip install asqav

Usage:
    ASQAV_API_KEY=sk_... python examples/dify_integration.py

What This Example Covers:
    1. How to install the Asqav plugin in Dify
    2. How to configure signing in a Dify workflow
    3. What the audit trail looks like
    4. How to verify agent actions programmatically
"""

import os

import asqav

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Get your API key from https://asqav.com
ASQAV_API_KEY = os.environ.get("ASQAV_API_KEY", "sk_your_api_key_here")

if ASQAV_API_KEY.startswith("sk_your"):
    print("⚠️  Warning: Please set ASQAV_API_KEY environment variable")
    print("   Get your key at: https://asqav.com")
    DEMO_MODE = True
else:
    asqav.init(api_key=ASQAV_API_KEY)
    DEMO_MODE = False

if not DEMO_MODE:
    agent = asqav.Agent.create("dify-governance-agent")


def install_dify_plugin_guide():
    """Step-by-step guide for installing the Asqav plugin in Dify."""
    print("=" * 70)
    print("STEP 1: Install the Asqav Dify Plugin")
    print("=" * 70)
    print("""
1. Open your Dify dashboard (self-hosted or cloud)
2. Navigate to "Plugins" → "Marketplace" → "Custom"
3. Click "Install from GitHub"
4. Enter the plugin repository URL:
   https://github.com/jagmarques/dify-plugin-asqav
5. Click "Install" and wait for the installation to complete
6. The plugin will appear as "Asqav Agent Governance" in your plugin list

Alternative: Install via Dify Plugin CLI
   $ dify-plugin install jagmarques/dify-plugin-asqav
""")


def configure_workflow_signing():
    """Step-by-step guide for configuring signing in a Dify workflow."""
    print("=" * 70)
    print("STEP 2: Configure Signing in a Dify Workflow")
    print("=" * 70)
    print("""
1. Open an existing workflow or create a new one
2. Add the "Asqav Sign" node from the plugin panel to your workflow
3. Configure the node:

   Required Settings:
   ├── API Key: Your Asqav API key (sk_...)
   ├── Agent ID: Unique identifier for this agent (e.g., "customer-support-bot")
   └── Action Type: The action being performed (e.g., "respond_to_customer")

   Optional Settings:
   ├── Store Raw Data: Whether to store the full prompt/response
   ├── Retention Days: How long to keep raw data (default: 30)
   └── Custom Metadata: Additional JSON metadata for the action

4. Connect the Asqav node after your LLM node:

   [Start] → [Input] → [LLM] → [Asqav Sign] → [End]
                              ↑
                    Signs and audits the LLM output

5. Save and publish the workflow
""")


def simulate_signed_workflow():
    """Simulate a Dify workflow execution with Asqav signing."""
    print("=" * 70)
    print("STEP 3: Simulated Workflow Execution")
    print("=" * 70)

    # This simulates what the Dify plugin does automatically
    workflow_steps = [
        {
            "agent_id": "customer-support-bot",
            "action": "classify_intent",
            "input_hash": "sha256:a3f5c8...",
            "output_hash": "sha256:b7e2d1...",
            "model": "gpt-4",
        },
        {
            "agent_id": "customer-support-bot",
            "action": "generate_response",
            "input_hash": "sha256:c9a1b4...",
            "output_hash": "sha256:d2f8e5...",
            "model": "gpt-4",
        },
        {
            "agent_id": "customer-support-bot",
            "action": "verify_compliance",
            "input_hash": "sha256:e5b3c7...",
            "output_hash": "sha256:f1a9d4...",
            "model": "gpt-4",
        },
    ]

    print("\nExecuting workflow with Asqav governance...\n")

    for i, step in enumerate(workflow_steps, 1):
        print(f"  Step {i}: {step['action']}")
        print(f"    Agent: {step['agent_id']}")
        print(f"    Input Hash:  {step['input_hash']}")
        print(f"    Output Hash: {step['output_hash']}")
        print(f"    Model: {step['model']}")
        print(f"    Status: ✅ Signed and recorded\n")

    print(f"  Total actions signed: {len(workflow_steps)}")
    print(f"  Audit trail: https://asqav.com/dashboard/#/sessions\n")


def show_audit_trail():
    """Display what the Asqav audit trail looks like."""
    print("=" * 70)
    print("STEP 4: Understanding the Audit Trail")
    print("=" * 70)
    print("""
The Asqav audit trail provides a complete, tamper-evident record of all
agent actions. Here's what each entry contains:

┌─────────────────────────────────────────────────────────────────────┐
│  AUDIT TRAIL ENTRY                                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Timestamp:    2026-04-26T14:32:01Z (UTC)                          │
│  Agent ID:     customer-support-bot                                 │
│  Action:       generate_response                                    │
│  Input Hash:   sha256:a3f5c8e2d1... (blinded for privacy)          │
│  Output Hash:  sha256:b7e2d1f4a9... (blinded for privacy)          │
│  Model:        gpt-4                                                │
│  Signature:    MbYJKGpOeNTu1hQvLmC... (base64-encoded ML-DSA)     │
│  Policy Check: PASSED (no PII detected)                            │
│  Retention:    30 days                                              │
└─────────────────────────────────────────────────────────────────────┘

Key Features:
  • ML-DSA (FIPS 204) signatures prevent future forgery
  • SHA-256 hashing protects sensitive data while maintaining auditability
  • Policy engine checks for PII, toxic content, and compliance violations
  • Retention policies ensure data is kept only as long as needed
""")


def verify_action_programmatically():
    """Show how to verify an agent action using the Asqav SDK."""
    print("=" * 70)
    print("STEP 5: Verify Actions Programmatically")
    print("=" * 70)
    print("""
After actions are signed in Dify, you can verify them using the SDK:

    import asqav

    # List sessions for an agent
    sessions = asqav.list_sessions(agent_id="customer-support-bot")

    # Verify a specific signature
    is_valid = asqav.verify_signature(
        signature="MbYJKGpOeNTu1hQvLmC...",
        data_hash="sha256:b7e2d1f4a9..."
    )

    # Export audit data to JSON
    asqav.export_audit_json(
        agent_id="customer-support-bot",
        output_path="./audit.json"
    )

    # Verify an output against a known hash
    result = asqav.verify_output(
        output="The actual agent response...",
        expected_hash="sha256:b7e2d1f4a9..."
    )

    print(f"Verification result: {result.status}")  # PASSED / FAILED
""")


def main():
    """Run the complete Dify integration example."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Asqav + Dify Integration Guide" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    install_dify_plugin_guide()
    configure_workflow_signing()
    simulate_signed_workflow()
    show_audit_trail()
    verify_action_programmatically()

    print("=" * 70)
    print("RESOURCES")
    print("=" * 70)
    print("""
  • Asqav Documentation:   https://docs.asqav.com
  • Dify Plugin Repo:      https://github.com/jagmarques/dify-plugin-asqav
  • SDK Repository:        https://github.com/jagmarques/asqav-sdk
  • Dashboard:             https://asqav.com/dashboard
  • Support:               support@asqav.com
""")


if __name__ == "__main__":
    main()
