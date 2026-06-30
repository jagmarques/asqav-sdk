/** Runnable example: opt-in context schema validation in agent.sign().
 * ASQAV_API_KEY=sk_... npx ts-node examples/schema-receipts.ts */

import {
  Agent,
  ValidationError,
  type ContextSchema,
  init,
} from "@asqav/sdk";

// Schema: field -> {type: string|number|boolean|object|array, required?: bool}
const PAYMENT_SCHEMA: ContextSchema = {
  userId:   { type: "string",  required: true },
  amount:   { type: "number",  required: true },
  currency: { type: "string",  required: true },
  metadata: { type: "object" },                // optional
};

async function main(): Promise<void> {
  const apiKey = process.env["ASQAV_API_KEY"] ?? "";
  if (!apiKey) {
    console.error(
      "Set ASQAV_API_KEY before running.\nGet your key at https://asqav.com"
    );
    process.exit(1);
  }

  init({ apiKey });
  const agent = await Agent.create({ name: "schema-demo" });

  // --- Valid context: passes validation, keys are sorted before signing ---
  console.log("Signing with a valid, schema-validated context...");
  const sig = await agent.sign({
    actionType: "payment:initiate",
    // Deliberately unsorted: SDK normalises to alphabetical key order.
    context: {
      userId:   "user_abc123",
      currency: "EUR",
      amount:   49.95,
      metadata: { source: "api" },
    },
    contextSchema: PAYMENT_SCHEMA,
  });
  console.log(`  signatureId : ${sig.signatureId}`);
  console.log(`  verify      : ${sig.verificationUrl}`);

  // --- Invalid context: throws BEFORE the network call ---
  console.log("\nAttempting to sign with a missing required field (userId)...");
  try {
    await agent.sign({
      actionType: "payment:initiate",
      context: { amount: 100.0, currency: "USD" }, // userId missing
      contextSchema: PAYMENT_SCHEMA,
    });
  } catch (err) {
    if (err instanceof ValidationError) {
      console.log(`  Caught (expected): ${err.message}`);
      console.log(`  Docs: ${err.docsUrl}`);
    }
  }

  // --- Wrong type: throws BEFORE the network call ---
  console.log("\nAttempting to sign with the wrong type for 'amount'...");
  try {
    await agent.sign({
      actionType: "payment:initiate",
      context: { userId: "u1", amount: "one-hundred", currency: "USD" },
      contextSchema: PAYMENT_SCHEMA,
    });
  } catch (err) {
    if (err instanceof ValidationError) {
      console.log(`  Caught (expected): ${err.message}`);
    }
  }

  // --- Custom callable validator (e.g. wrap Ajv for full JSON Schema) ---
  console.log("\nSigning with a custom callable validator...");
  const sig2 = await agent.sign({
    actionType: "payment:initiate",
    context: { userId: "user_xyz", amount: 10.0, currency: "GBP" },
    contextSchema: (ctx) => {
      if (typeof ctx["amount"] === "number" && ctx["amount"] <= 0) {
        throw new Error("amount must be positive");
      }
    },
  });
  console.log(`  signatureId : ${sig2.signatureId}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
