/**
 * Vercel AI SDK + Asqav cookbook recipe
 *
 * Wraps a Vercel AI SDK tool so every invocation produces an ML-DSA-signed
 * audit record on Asqav before the side effect runs. If the side effect
 * throws, the signed intent still exists in the audit trail.
 *
 * Install:
 *   npm install @asqav/sdk ai @ai-sdk/openai zod
 *
 * Env:
 *   export ASQAV_API_KEY=sk_...   # from https://asqav.com
 *   export OPENAI_API_KEY=sk-...
 *
 * Run:
 *   tsx examples/vercel-ai-sdk.ts
 */

import { tool, generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";
import { init, Agent } from "@asqav/sdk";

// Stripe stub. Replace with the real Stripe call in production.
async function stripeRefundStub(args: {
  chargeId: string;
  amountCents: number;
  reason: string;
}): Promise<{ refundId: string; status: string }> {
  return {
    refundId: `re_test_${Date.now()}`,
    status: "succeeded",
  };
}

async function main(): Promise<void> {
  init({ apiKey: process.env.ASQAV_API_KEY });

  const agent = await Agent.create({
    name: "support-bot",
    capabilities: ["refunds:issue"],
  });

  await agent.startSession();

  const refund = tool({
    description:
      "Refund a Stripe charge. Use only when the customer explicitly requests one.",
    inputSchema: z.object({
      chargeId: z.string().describe("The Stripe charge id, like ch_..."),
      amountCents: z.number().int().positive().describe("Amount to refund in cents"),
      reason: z.string().describe("Short reason for the refund"),
    }),
    execute: async ({ chargeId, amountCents, reason }) => {
      // Sign the intent FIRST. If the side effect throws, the audit
      // record still proves what the agent decided to do.
      //
      // Opt into the IETF Compliance Receipts profile so the cloud
      // emits a §5.7-conformant chain link, content-addressed
      // policy_digest, and applies fail-closed anchoring under the
      // raised retention floor.
      const sig = await agent.sign({
        actionType: "stripe:refund",
        context: { chargeId, amountCents, reason, model: "gpt-4o" },
        complianceMode: true,
        receiptType: "protectmcp:decision",
        riskClass: "high",
        sandboxState: "disabled",
        iterationId: `refund-${chargeId}`,
        policyDecision: "permit",
      });

      const result = await stripeRefundStub({ chargeId, amountCents, reason });

      return {
        ...result,
        asqavSignatureId: sig.signatureId,
        asqavVerificationUrl: sig.verificationUrl,
      };
    },
  });

  const { text, toolResults } = await generateText({
    model: openai("gpt-4o"),
    tools: { refund },
    prompt:
      "Customer asked for a refund of 1299 cents on charge ch_test_abc because the item arrived broken. Issue the refund.",
  });

  console.log("Model reply:", text);
  console.log("Tool results:", JSON.stringify(toolResults, null, 2));

  await agent.endSession();
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
