/**
 * LangChain.js + Asqav cookbook recipe
 *
 * Wraps a LangChain tool so every call produces an ML-DSA-signed audit
 * record on Asqav before the side effect runs. The signed intent persists
 * even if the underlying API call throws.
 *
 * Install:
 *   npm install @asqav/sdk @langchain/core @langchain/openai zod
 *
 * Env:
 *   export ASQAV_API_KEY=sk_...   # from https://asqav.com
 *   export OPENAI_API_KEY=sk-...
 *
 * Run:
 *   tsx examples/langchain-js.ts
 */

import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { init, Agent } from "@asqav/sdk";
import { enableLangchainGovernance } from "@asqav/sdk/extras/langchain";

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

  // Default-on: sign chain/tool/LLM lifecycle receipts for every run below
  // with no per-invoke { callbacks: [...] }. The explicit agent.sign inside
  // the refund tool still records the high-value intent receipt.
  await enableLangchainGovernance({ agentName: "support-bot-governance" });

  const agent = await Agent.create({
    name: "support-bot",
    capabilities: ["refunds:issue"],
  });

  await agent.startSession();

  const refundSchema = z.object({
    chargeId: z.string().describe("The Stripe charge id, like ch_..."),
    amountCents: z.number().int().positive().describe("Amount to refund in cents"),
    reason: z.string().describe("Short reason for the refund"),
  });

  const refund = tool(
    async (args: z.infer<typeof refundSchema>) => {
      // Sign the intent FIRST: if the refund throws, the receipt still proves what the
      // agent decided. complianceMode adds fail-closed anchoring + raised retention.
      const sig = await agent.sign({
        actionType: "stripe:refund",
        context: {
          chargeId: args.chargeId,
          amountCents: args.amountCents,
          reason: args.reason,
          model: "gpt-4o",
        },
        complianceMode: true,
        receiptType: "protectmcp:decision",
        riskClass: "high",
        sandboxState: "disabled",
        iterationId: `refund-${args.chargeId}`,
        policyDecision: "permit",
      });

      const result = await stripeRefundStub(args);

      return JSON.stringify({
        ...result,
        asqavSignatureId: sig.signatureId,
        asqavVerificationUrl: sig.verificationUrl,
      });
    },
    {
      name: "refund",
      description:
        "Refund a Stripe charge. Use only when the customer explicitly requests one.",
      schema: refundSchema,
    },
  );

  const model = new ChatOpenAI({ model: "gpt-4o", temperature: 0 });
  const modelWithTools = model.bindTools([refund]);

  const response = await modelWithTools.invoke([
    {
      role: "user",
      content:
        "Customer asked for a refund of 1299 cents on charge ch_test_abc because the item arrived broken. Issue the refund.",
    },
  ]);

  console.log("Model tool calls:", JSON.stringify(response.tool_calls, null, 2));

  // Execute any tool calls the model proposed.
  for (const call of response.tool_calls ?? []) {
    if (call.name === "refund") {
      const result = await refund.invoke(call);
      console.log("Tool result:", result);
    }
  }

  await agent.endSession();
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
