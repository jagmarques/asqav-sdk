import { afterEach, expect, it, vi } from "vitest";
import { Agent, _resetForTests, init, type SignOptions } from "../src/index.js";

afterEach(() => {
  vi.restoreAllMocks();
  _resetForTests();
});

it.each<[Partial<SignOptions>, string]>([
  [{ configManifestDigest: "wrong" }, "config_manifest_digest_not_sha256_wire_form"],
  [{ iso42001: ["A".repeat(129)] }, "iso_42001_entry_invalid"],
  [{ rfc3161Timestamp: "@@@@" }, "rfc3161_timestamp_not_base64"],
])("validates signing extensions before HTTP: %j", async (options, token) => {
  init({ apiKey: "asq_test_fixture", baseUrl: "https://api.example.com/api/v1" });
  const post = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({
    signature: "fixture", signature_id: "sig_fixture", action_id: "act_fixture",
    timestamp: "2026-05-25T00:00:00Z", verification_url: "https://example.com/verify",
  }), { status: 201, headers: { "Content-Type": "application/json" } }));
  const agent = Agent.attach({
    agent_id: "agt_fixture", name: "fixture", public_key: "fixture", key_id: "kid_fixture",
    algorithm: "ml-dsa-65", capabilities: [], created_at: "2026-05-25T00:00:00Z",
  });
  await expect(agent.sign({ actionType: "api:call", context: {}, ...options })).rejects.toThrow(token);
  expect(post).not.toHaveBeenCalled();
});
