/**
 * SDK parity for the code-authorship receipt.
 *
 * Mirrors the deployed cloud SignRequest validators (routes/agents.py
 * `_validate_code_authorship_extensions` + AuthoredBy) for
 * `receiptType='protectmcp:lifecycle:code_authorship'`. The SDK forwards the
 * eight extension fields verbatim (camelCase -> snake_case), validates the
 * changeDigest shape, the changeClass closed vocabulary, the authored_by
 * model-requires-attestation_source rule, the field-fence to the
 * code-authorship type, and requires complianceMode so the fields are never
 * silently dropped from the signed bytes. Honest-scope: every field is
 * producer-asserted / recorded-not-verified.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  Agent,
  AsqavError,
  CODE_AUTHORSHIP_DOCS_URL,
  RECEIPT_TYPE_NAMESPACE,
  _resetForTests,
  init,
} from "../src/index.js";

const CODE_TYPE = "protectmcp:lifecycle:code_authorship";
const GOOD_DIGEST = "sha256:" + "a".repeat(64);
const GOOD_AUTHORED_BY = {
  humanId: "human:alice@example.com",
  modelId: "claude-opus-4-8",
  modelVersion: "2026-01",
  attestationSource: "GitHub Copilot session export",
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function fakeAgent(): Agent {
  return Agent.attach({
    agent_id: "agt_code",
    name: "code",
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
    timestamp: "2026-06-01T19:00:00Z",
    verification_url: "https://verify",
    ...extra,
  };
}

function readBody(spy: ReturnType<typeof vi.spyOn>): Record<string, unknown> {
  const init = spy.mock.calls[0][1] as RequestInit;
  return JSON.parse((init?.body as string) ?? "{}");
}

function goodCodeOptions(): Record<string, unknown> {
  return {
    actionType: "code:author",
    context: { k: "v" },
    complianceMode: true,
    receiptType: CODE_TYPE,
    policyDecision: "none",
    repoRef: "github.com/acme/repo",
    commitSha: "c0ffee0000000000000000000000000000000000",
    baseSha: "ba5e0000000000000000000000000000000000000",
    changeDigest: GOOD_DIGEST,
    changeRef: "github.com/acme/repo/pull/42",
    changeApprovalRef: "JIRA-ENG-9001",
    changeClass: "write",
    authoredBy: GOOD_AUTHORED_BY,
  };
}

beforeEach(() => {
  _resetForTests();
  init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("code-authorship receipt namespace", () => {
  it("includes the code_authorship type in the namespace", () => {
    expect(RECEIPT_TYPE_NAMESPACE).toContain(CODE_TYPE);
  });
});

describe("code-authorship wire forwarding (valid)", () => {
  it("forwards all eight extension fields as snake_case", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign(goodCodeOptions() as any);
    const body = readBody(spy);
    expect(body.repo_ref).toBe("github.com/acme/repo");
    expect(body.commit_sha).toBe("c0ffee0000000000000000000000000000000000");
    expect(body.base_sha).toBe("ba5e0000000000000000000000000000000000000");
    expect(body.change_digest).toBe(GOOD_DIGEST);
    expect(body.change_ref).toBe("github.com/acme/repo/pull/42");
    expect(body.change_approval_ref).toBe("JIRA-ENG-9001");
    expect(body.change_class).toBe("write");
    expect(body.authored_by).toEqual({
      human_id: "human:alice@example.com",
      model_id: "claude-opus-4-8",
      model_version: "2026-01",
      attestation_source: "GitHub Copilot session export",
    });
  });

  it("accepts a human-only author without attestation_source", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(okBody()));
    await fakeAgent().sign({
      ...goodCodeOptions(),
      authoredBy: { humanId: "human:alice@example.com" },
    } as any);
    const body = readBody(spy);
    expect((body.authored_by as Record<string, unknown>).human_id).toBe("human:alice@example.com");
  });
});

describe("code-authorship rejections", () => {
  it("rejects a bad change_digest shape", async () => {
    await expect(
      fakeAgent().sign({ ...goodCodeOptions(), changeDigest: "not-a-sha256" } as any),
    ).rejects.toThrow(/change_digest_not_sha256_wire_form/);
  });

  it("rejects a change_class outside the closed vocabulary", async () => {
    await expect(
      fakeAgent().sign({ ...goodCodeOptions(), changeClass: "merge" } as any),
    ).rejects.toThrow(/code_authorship_change_class_invalid/);
  });

  it("rejects a model author without attestation_source", async () => {
    await expect(
      fakeAgent().sign({
        ...goodCodeOptions(),
        authoredBy: { modelId: "claude-opus-4-8", attestationSource: "" },
      } as any),
    ).rejects.toThrow(/authored_by_model_requires_attestation_source/);
  });

  it("rejects code fields on a non-code-authorship receipt type", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "lifecycle:change",
        complianceMode: true,
        receiptType: "protectmcp:lifecycle",
        changeDigest: GOOD_DIGEST,
      } as any),
    ).rejects.toThrow(/code_authorship_fields_require_code_authorship_receipt/);
  });

  it("rejects a code-authorship receipt missing repo_ref/commit_sha", async () => {
    await expect(
      fakeAgent().sign({
        actionType: "code:author",
        complianceMode: true,
        receiptType: CODE_TYPE,
        policyDecision: "none",
      } as any),
    ).rejects.toThrow(/code_authorship_missing_required_field/);
  });

  it("rejects a code-authorship receipt without compliance_mode", async () => {
    await expect(
      fakeAgent().sign({
        ...goodCodeOptions(),
        complianceMode: false,
      } as any),
    ).rejects.toThrow(/code_authorship_requires_compliance_mode/);
  });

  it("rejects a code-authorship receipt with a non-none policy_decision", async () => {
    await expect(
      fakeAgent().sign({
        ...goodCodeOptions(),
        policyDecision: "permit",
      } as any),
    ).rejects.toThrow(/code_authorship_requires_no_policy_decision/);
  });

  it("attaches the docs_url to the validation error (rule 3.9)", async () => {
    try {
      await fakeAgent().sign({ ...goodCodeOptions(), changeDigest: "bad" } as any);
      throw new Error("expected rejection");
    } catch (err) {
      expect(err).toBeInstanceOf(AsqavError);
      expect((err as AsqavError).docsUrl).toBe(CODE_AUTHORSHIP_DOCS_URL);
    }
  });
});
