/** Bind the declared config (manifest, AGENTS.md, image digest, SBOM) into the
 * schema-pinned sign context as a SHA-256 digest, with no new wire field. */
// Run: ASQAV_API_KEY=sk_... node examples/bind-config-digest.mjs

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { init, Agent } from "@asqav/sdk";

// Declaration bound to every receipt this agent signs; in production use
// your AGENTS.md, a manifest or an image digest on disk
const MANIFEST_URL = new URL("./agent-manifest.example.json", import.meta.url);

// Pin the binding keys so a receipt that claims a config always carries both
// the digest and the artifact kind. Extra context keys pass through
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

  // Verifiers see which declared config authorized this action
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
