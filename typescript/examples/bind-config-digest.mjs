/** Bind your agent's declared configuration into the signed receipt.
 *
 * Hash whatever declares what the agent is allowed to be (an agent manifest,
 * an AGENTS.md, a container image digest, an SBOM) with SHA-256, put the digest
 * in the sign context, and pin it with a context schema. The digest then lands
 * inside the cryptographically signed receipt, so anyone can later confirm which
 * declared configuration produced a given action at the public verify endpoint.
 *
 * This uses the existing schema-validated sign context. No new wire field.
 *
 * ASQAV_API_KEY=sk_... node examples/bind-config-digest.mjs
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { init, Agent } from "@asqav/sdk";

// The declaration you want bound to every receipt this agent signs. In a real
// deployment this is your AGENTS.md, a manifest, or an image digest on disk.
const MANIFEST_URL = new URL("./agent-manifest.example.json", import.meta.url);

// Pin the binding keys so a receipt that claims a config always carries both
// the digest and what kind of artifact it hashes. Extra context keys pass through.
const CONFIG_BINDING_SCHEMA = {
  config_digest: { type: "string", required: true },
  config_kind: { type: "string", required: true },
};

function configDigest(url) {
  return "sha256:" + createHash("sha256").update(readFileSync(url)).digest("hex");
}

async function main() {
  const apiKey = process.env["ASQAV_API_KEY"] ?? "";
  if (!apiKey) {
    console.error("Set ASQAV_API_KEY before running.\nGet your key at https://asqav.com");
    process.exit(1);
  }

  init({ apiKey });
  const agent = await Agent.create({ name: "config-binding-demo" });

  const digest = configDigest(MANIFEST_URL);
  console.log(`declaration : agent-manifest.example.json`);
  console.log(`config_digest: ${digest}`);

  // Bind the digest into the action the agent is signing. Anyone verifying
  // the receipt sees exactly which declared config authorized this action.
  const sig = await agent.sign({
    actionType: "payment:refund",
    context: {
      amount: 49.95,
      currency: "EUR",
      config_digest: digest,
      config_kind: "agent-manifest",
      config_ref: "agent-manifest.example.json",
    },
    contextSchema: CONFIG_BINDING_SCHEMA,
  });

  console.log(`signatureId : ${sig.signatureId}`);
  console.log(`verify      : ${sig.verificationUrl}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
