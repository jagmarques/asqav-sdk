/**
 * 10-minute quickstart: govern() -> agent.sign() -> verify.
 * Run: ASQAV_API_KEY=sk_... npx tsx quickstart.ts
 */

import { govern } from "@asqav/sdk";

async function main() {
  // One call: init + Agent.create
  const agent = await govern({
    apiKey: process.env.ASQAV_API_KEY,
    agentName: "quickstart-agent",
  });

  console.log(`Agent created: ${agent.agentId} (${agent.name})`);

  // Sign an action
  const sig = await agent.sign({
    actionType: "api:call",
    context: { model: "gpt-4", tokens: 512 },
  });

  console.log(`Signed: ${sig.signatureId}`);
  console.log(`Verify: ${sig.verificationUrl}`);
}

main().catch(console.error);
