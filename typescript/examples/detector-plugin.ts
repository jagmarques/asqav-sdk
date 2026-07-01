/** Runnable example: pluggable DLP/policy detector on the sign path.
 * A registered detector runs inside agent.sign(), after normalization and before
 * the network call; its verdict is recorded in the signed receipt and a deny throws
 * DetectorBlockedError. ASQAV_API_KEY=sk_... npx ts-node examples/detector-plugin.ts */

import {
  Agent,
  DetectorBlockedError,
  init,
  registerDetector,
  type DetectorResult,
} from "@asqav/sdk";

const SSN_RE = /\b\d{3}-\d{2}-\d{4}\b/;

// Tiny custom detector so this runs with no extra install. The reference
// PresidioDetector (PII) and OpaDetector (policy) live in @asqav/sdk/extras.
const ssnDetector = {
  name: "ssn-regex",
  inspect(_actionType: string, context: Record<string, unknown>): DetectorResult {
    if (SSN_RE.test(JSON.stringify(context))) {
      return { allow: false, confidence: 1.0, labels: ["US_SSN"], detector: "", reason: "SSN in context" };
    }
    return { allow: true, confidence: 1.0, labels: [], detector: "", reason: "" };
  },
};

async function main(): Promise<void> {
  const apiKey = process.env["ASQAV_API_KEY"] ?? "";
  if (!apiKey) {
    console.error("Set ASQAV_API_KEY before running.\nGet your key at https://asqav.com");
    process.exit(1);
  }

  init({ apiKey });
  const agent = await Agent.create({ name: "detector-demo" });

  // failOpen defaults to false: a detector error blocks rather than bypasses.
  registerDetector(ssnDetector);

  // Clean context: allowed, and the receipt records the detector verdict.
  console.log("Signing a clean action (detector allows)...");
  const sig = await agent.sign({
    actionType: "email:send",
    context: { to: "ops@acme.com", body: "quarterly report" },
  });
  console.log(`  signatureId : ${sig.signatureId}`);
  console.log(`  verify      : ${sig.verificationUrl}`);

  // Sensitive context: the detector denies, sign() blocks before the network.
  console.log("\nSigning an action with an SSN (detector denies)...");
  try {
    await agent.sign({
      actionType: "email:send",
      context: { to: "ops@acme.com", body: "ssn 123-45-6789" },
    });
  } catch (err) {
    if (err instanceof DetectorBlockedError) {
      console.log(`  Blocked by  : ${err.detectorName}`);
      console.log(`  Labels      : ${err.result.labels.join(", ")}`);
      console.log(`  Docs        : ${err.docsUrl}`);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
