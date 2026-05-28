/**
 * SDK parity for the `witnessPolicy` wire field.
 *
 * The cloud accepts an optional `witness_policy` object on the signed
 * Compliance Receipt with shape
 * `{ required: int, witnesses: [<subset of "rfc3161", "opentimestamps">] }`.
 * The SDK projects `witnessPolicy` to the snake_case `witness_policy` wire
 * key verbatim and validates the shape client-side: `required` in
 * `[1, witnesses.length]`, `witnesses` a non-empty subset of the two
 * shipped witnesses, and `rekor` rejected.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Agent, _resetForTests, init } from "../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fakeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_witness",
    name: "witness",
    public_key: "pk",
    key_id: "kid",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-05-25T00:00:00Z",
  });
}

function okBody(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    signature: "sig",
    signature_id: "sig_test",
    action_id: "act_test",
    timestamp: "2026-05-25T00:00:00Z",
    verification_url: "https://verify",
    ...extra,
  };
}

function readBody(spy: ReturnType<typeof vi.spyOn>): Record<string, unknown> {
  const init = spy.mock.calls[0][1] as RequestInit;
  return JSON.parse((init?.body as string) ?? "{}");
}

beforeEach(() => {
  _resetForTests();
  init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("witnessPolicy wire forwarding", () => {
  it("projects witnessPolicy to snake_case witness_policy", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "deploy:release",
      context: { k: "v" },
      complianceMode: true,
      witnessPolicy: { required: 1, witnesses: ["rfc3161", "opentimestamps"] },
    });
    const body = readBody(spy);
    expect(body.witness_policy).toEqual({
      required: 1,
      witnesses: ["rfc3161", "opentimestamps"],
    });
  });

  it("forwards a single-witness policy", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "deploy:release",
      context: { k: "v" },
      complianceMode: true,
      witnessPolicy: { required: 1, witnesses: ["opentimestamps"] },
    });
    const body = readBody(spy);
    expect(body.witness_policy).toEqual({ required: 1, witnesses: ["opentimestamps"] });
  });

  it("omits witness_policy when not supplied", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      actionType: "deploy:release",
      context: { k: "v" },
      complianceMode: true,
    });
    const body = readBody(spy);
    expect(body.witness_policy).toBeUndefined();
  });
});

describe("witnessPolicy shape guards", () => {
  it("rejects rekor as a witness", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "deploy:release",
        context: { k: "v" },
        complianceMode: true,
        // rekor is not a shipped witness.
        witnessPolicy: { required: 1, witnesses: ["rekor"] as never },
      }),
    ).rejects.toThrow(/witness_policy_unknown_witness/);
  });

  it("rejects rekor mixed with a valid witness", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "deploy:release",
        context: { k: "v" },
        complianceMode: true,
        witnessPolicy: { required: 1, witnesses: ["rfc3161", "rekor"] as never },
      }),
    ).rejects.toThrow(/witness_policy_unknown_witness/);
  });

  it("rejects required greater than witnesses length", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "deploy:release",
        context: { k: "v" },
        complianceMode: true,
        witnessPolicy: { required: 3, witnesses: ["rfc3161", "opentimestamps"] },
      }),
    ).rejects.toThrow(/witness_policy_required_out_of_range/);
  });

  it("rejects required of zero", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "deploy:release",
        context: { k: "v" },
        complianceMode: true,
        witnessPolicy: { required: 0, witnesses: ["rfc3161"] },
      }),
    ).rejects.toThrow(/witness_policy_required_out_of_range/);
  });

  it("rejects an empty witnesses list", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "deploy:release",
        context: { k: "v" },
        complianceMode: true,
        witnessPolicy: { required: 1, witnesses: [] },
      }),
    ).rejects.toThrow(/witness_policy_witnesses_must_be_non_empty_list/);
  });

  it("rejects duplicate witnesses", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "deploy:release",
        context: { k: "v" },
        complianceMode: true,
        witnessPolicy: { required: 1, witnesses: ["rfc3161", "rfc3161"] },
      }),
    ).rejects.toThrow(/witness_policy_duplicate_witness/);
  });

  it("rejects non-integer required", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "deploy:release",
        context: { k: "v" },
        complianceMode: true,
        witnessPolicy: { required: 1.5, witnesses: ["rfc3161"] },
      }),
    ).rejects.toThrow(/witness_policy_required_must_be_int/);
  });
});
