/**
 * Destructive SQL bypass fix: augment-match policy coverage.
 *
 * A data:write:sql: action with a destructive verb (DELETE, DROP, TRUNCATE,
 * ALTER) must also match a data:delete:* block policy. The write-namespace
 * match must be preserved (regression guard). Reads and benign writes are not
 * affected.
 *
 * Each test here fails against pre-fix code and passes after the fix.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Agent, _resetForTests, init } from "../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function makeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_1",
    name: "test",
    public_key: "pk",
    key_id: "kid",
    algorithm: "ml-dsa-65",
    capabilities: [],
    created_at: "2026-01-01T00:00:00Z",
  });
}

const activeAgent = { revoked: false, suspended: false };

const deletePolicies = [
  { is_active: true, action_pattern: "data:delete:*", action: "block", name: "no-deletions" },
];

const writeSqlPolicies = [
  { is_active: true, action_pattern: "data:write:sql:*", action: "block", name: "no-sql-writes" },
];

describe("destructive SQL write bypass fix", () => {
  beforeEach(() => {
    _resetForTests();
    init({ apiKey: "sk_test", baseUrl: "https://api.example.test/api/v1" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("data:delete:* policy blocks data:write:sql:DELETE FROM users", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(activeAgent))
      .mockResolvedValueOnce(jsonResponse(deletePolicies));

    const r = await makeAgent().preflight("data:write:sql:DELETE FROM users");

    expect(r.cleared).toBe(false);
    expect(r.policyAllowed).toBe(false);
    expect(r.reasons.some((s) => s.includes("no-deletions"))).toBe(true);
  });

  it("data:delete:* policy blocks data:write:sql:DELETE_FROM_users (underscore form)", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(activeAgent))
      .mockResolvedValueOnce(jsonResponse(deletePolicies));

    const r = await makeAgent().preflight("data:write:sql:DELETE_FROM_users");

    expect(r.cleared).toBe(false);
    expect(r.policyAllowed).toBe(false);
    expect(r.reasons.some((s) => s.includes("no-deletions"))).toBe(true);
  });

  it("data:delete:* policy blocks data:write:sql:DROP TABLE t", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(activeAgent))
      .mockResolvedValueOnce(jsonResponse(deletePolicies));

    const r = await makeAgent().preflight("data:write:sql:DROP TABLE t");

    expect(r.cleared).toBe(false);
    expect(r.policyAllowed).toBe(false);
  });

  it("data:delete:* policy blocks data:write:sql:TRUNCATE TABLE t", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(activeAgent))
      .mockResolvedValueOnce(jsonResponse(deletePolicies));

    const r = await makeAgent().preflight("data:write:sql:TRUNCATE TABLE t");

    expect(r.cleared).toBe(false);
    expect(r.policyAllowed).toBe(false);
  });

  it("regression guard: data:write:sql:* still blocks a destructive write action", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(activeAgent))
      .mockResolvedValueOnce(jsonResponse(writeSqlPolicies));

    const r = await makeAgent().preflight("data:write:sql:DELETE FROM users");

    expect(r.cleared).toBe(false);
    expect(r.policyAllowed).toBe(false);
    expect(r.reasons.some((s) => s.includes("no-sql-writes"))).toBe(true);
  });

  it("negative: data:delete:* does not block a benign INSERT write", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(activeAgent))
      .mockResolvedValueOnce(jsonResponse(deletePolicies));

    const r = await makeAgent().preflight("data:write:sql:INSERT INTO t VALUES (1)");

    expect(r.cleared).toBe(true);
    expect(r.policyAllowed).toBe(true);
  });

  it("negative: data:delete:* does not block a read mentioning delete", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(activeAgent))
      .mockResolvedValueOnce(jsonResponse(deletePolicies));

    const r = await makeAgent().preflight("data:read:sql:SELECT delete_flag FROM t");

    expect(r.cleared).toBe(true);
    expect(r.policyAllowed).toBe(true);
  });

  it("negative: deleted_at in a write suffix does not trigger destructive detection", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(activeAgent))
      .mockResolvedValueOnce(jsonResponse(deletePolicies));

    const r = await makeAgent().preflight("data:write:sql:UPDATE t SET deleted_at=NOW()");

    expect(r.cleared).toBe(true);
    expect(r.policyAllowed).toBe(true);
  });

  // H3: case-variant and whitespace must not bypass the destructive augment.
  it("uppercase DATA:WRITE:SQL:DELETE is blocked by a lowercase data:delete:* policy", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(activeAgent))
      .mockResolvedValueOnce(jsonResponse(deletePolicies));

    const r = await makeAgent().preflight("DATA:WRITE:SQL:DELETE FROM users");

    expect(r.cleared).toBe(false);
    expect(r.policyAllowed).toBe(false);
    expect(r.reasons.some((s) => s.includes("no-deletions"))).toBe(true);
  });

  it("leading/trailing whitespace DELETE write is blocked by data:delete:*", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(activeAgent))
      .mockResolvedValueOnce(jsonResponse(deletePolicies));

    const r = await makeAgent().preflight("  data:write:sql:DELETE FROM users  ");

    expect(r.cleared).toBe(false);
    expect(r.policyAllowed).toBe(false);
    expect(r.reasons.some((s) => s.includes("no-deletions"))).toBe(true);
  });

  it("a lowercase block pattern blocks an uppercase write action", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(activeAgent))
      .mockResolvedValueOnce(jsonResponse(writeSqlPolicies));

    const r = await makeAgent().preflight("DATA:WRITE:SQL:DELETE FROM users");

    expect(r.cleared).toBe(false);
    expect(r.policyAllowed).toBe(false);
    expect(r.reasons.some((s) => s.includes("no-sql-writes"))).toBe(true);
  });

  it("an uppercase block pattern blocks a lowercase write action", async () => {
    const upperWritePolicy = [
      { is_active: true, action_pattern: "DATA:WRITE:SQL:*", action: "block", name: "no-sql-writes" },
    ];
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(activeAgent))
      .mockResolvedValueOnce(jsonResponse(upperWritePolicy));

    const r = await makeAgent().preflight("data:write:sql:INSERT INTO t VALUES (1)");

    expect(r.cleared).toBe(false);
    expect(r.policyAllowed).toBe(false);
  });

  // M1: extended destructive verbs each match the delete augment.
  it.each(["GRANT", "REVOKE", "REPLACE", "COPY", "UPSERT"])(
    "data:delete:* policy blocks data:write:sql:%s (extended destructive verb)",
    async (verb) => {
      vi.spyOn(globalThis, "fetch")
        .mockResolvedValueOnce(jsonResponse(activeAgent))
        .mockResolvedValueOnce(jsonResponse(deletePolicies));

      const r = await makeAgent().preflight(`data:write:sql:${verb} something`);

      expect(r.cleared).toBe(false);
      expect(r.policyAllowed).toBe(false);
      expect(r.reasons.some((s) => s.includes("no-deletions"))).toBe(true);
    },
  );
});
