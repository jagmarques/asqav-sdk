/**
 * SDK half of the authoritative code-authorship path: the client digest is ADVISORY and the
 * server's recomputed one is signed. Only `github_sha_pull` is authoritative.
 */

import { createHash } from "node:crypto";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  AUTHORITATIVE_CAPTURE_LAYER,
  CODE_AUTHORSHIP_ASSET_CLASS,
  CODE_AUTHORSHIP_PATH,
  CODE_AUTHORSHIP_PREDICATE_TYPE,
  CODE_AUTHORSHIP_WRITE_SCOPE,
  INTOTO_STATEMENT_TYPE,
  OBSERVATION_ONLY_CAPTURE_LAYERS,
  _resetForTests,
  computeAdvisoryDigest,
  init,
  parseCodeAuthorshipResponse,
  subjectMatchesServer,
  submitCodeAuthorship,
  verifyCodeAuthorshipEnvelope,
} from "../src/index.js";

const SERVER_HEX = "f".repeat(64);
const ADVISORY = "sha256:" + "1".repeat(64);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function serverEnvelope(opts: {
  serverHex: string;
  captureLayer: string;
  advisory?: string;
  digestMatch?: boolean;
  predicateType?: string;
  statementType?: string;
  includeSubjectDigest?: boolean;
}): Record<string, unknown> {
  const subject =
    opts.includeSubjectDigest === false
      ? []
      : [{ name: "owner/repo@" + "c".repeat(40), digest: { sha256: opts.serverHex } }];
  return {
    _type: opts.statementType ?? INTOTO_STATEMENT_TYPE,
    subject,
    predicateType: opts.predicateType ?? CODE_AUTHORSHIP_PREDICATE_TYPE,
    predicate: {
      capture_layer: opts.captureLayer,
      asset_class: CODE_AUTHORSHIP_ASSET_CLASS,
      advisory_client_digest: opts.advisory ?? ADVISORY,
      digest_match: opts.digestMatch ?? true,
    },
  };
}

function serverResponse(opts: {
  serverHex: string;
  captureLayer?: string;
  digestMatch?: boolean;
}): Record<string, unknown> {
  return {
    envelope: serverEnvelope({
      serverHex: opts.serverHex,
      captureLayer: opts.captureLayer ?? "github_sha_pull",
    }),
    receipt: { signature_id: "sig_ca", algorithm: "ML-DSA-65" },
    kid: "key_ca",
    jwks_url: "https://api.asqav.com/.well-known/jwks.json",
    server_digest: "sha256:" + opts.serverHex,
    digest_match: opts.digestMatch ?? true,
  };
}

beforeEach(() => {
  _resetForTests();
  init({ apiKey: "asq_test_key", baseUrl: "https://api.example.com/api/v1" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("code-authorship constants + scope", () => {
  it("exposes the code_authorship:write scope", () => {
    expect(CODE_AUTHORSHIP_WRITE_SCOPE).toBe("code_authorship:write");
  });

  it("matches the contract predicate type, asset class, and capture layers", () => {
    expect(CODE_AUTHORSHIP_PREDICATE_TYPE).toBe(
      "https://asqav.com/attestation/code-authorship/v1",
    );
    expect(CODE_AUTHORSHIP_ASSET_CLASS).toBe("code");
    expect(AUTHORITATIVE_CAPTURE_LAYER).toBe("github_sha_pull");
    expect([...OBSERVATION_ONLY_CAPTURE_LAYERS].sort()).toEqual([
      "in_process_sdk",
      "passive_telemetry",
    ]);
  });
});

describe("computeAdvisoryDigest", () => {
  it("hashes supplied diff text into the sha256 wire form", () => {
    const diff = "diff --git a/x b/x\n+hello\n";
    const expected = "sha256:" + createHash("sha256").update(diff, "utf8").digest("hex");
    expect(computeAdvisoryDigest("head", diff)).toBe(expected);
  });

  it("falls back to the head sha when no diff is supplied", () => {
    const head = "a".repeat(40);
    const expected = "sha256:" + createHash("sha256").update(head, "utf8").digest("hex");
    expect(computeAdvisoryDigest(head)).toBe(expected);
  });

  it("always emits a sha256:<64 hex> value", () => {
    const digest = computeAdvisoryDigest("head", "anything");
    expect(digest.startsWith("sha256:")).toBe(true);
    const hex = digest.split(":")[1];
    expect(hex).toHaveLength(64);
    expect(/^[0-9a-f]{64}$/.test(hex)).toBe(true);
  });
});

describe("submitCodeAuthorship", () => {
  it("posts the advisory digest to /code-authorship and reads the server subject", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(serverResponse({ serverHex: SERVER_HEX })));

    const result = await submitCodeAuthorship({
      repo: "owner/repo",
      commitSha: "c".repeat(40),
      baseSha: "b".repeat(40),
      changeDigest: ADVISORY,
      changeClass: "write",
      author: "human:alice@example.com",
      anchor: "https://github.com/owner/repo/pull/7",
    });

    const url = spy.mock.calls[0][0] as string;
    expect(url.endsWith(CODE_AUTHORSHIP_PATH)).toBe(true);
    const body = JSON.parse((spy.mock.calls[0][1] as RequestInit)?.body as string);
    expect(body.repo).toBe("owner/repo");
    expect(body.commit_sha).toBe("c".repeat(40));
    expect(body.base_sha).toBe("b".repeat(40));
    // The client digest is advisory. It travels so the server can report a match.
    expect(body.change_digest).toBe(ADVISORY);
    expect(body.change_class).toBe("write");
    expect(body.author).toBe("human:alice@example.com");
    expect(body.anchor).toBe("https://github.com/owner/repo/pull/7");

    // The authoritative subject digest is the SERVER value, not the client's.
    expect(result.subjectDigest).toBe(SERVER_HEX);
    expect(result.serverDigest).toBe("sha256:" + SERVER_HEX);
    expect(result.captureLayer).toBe("github_sha_pull");
    expect(result.kid).toBe("key_ca");
    expect(subjectMatchesServer(result)).toBe(true);
  });

  it("omits absent optional fields", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(serverResponse({ serverHex: "e".repeat(64) })));

    await submitCodeAuthorship({ repo: "owner/repo", commitSha: "c".repeat(40) });

    const body = JSON.parse((spy.mock.calls[0][1] as RequestInit)?.body as string);
    expect(body).toEqual({ repo: "owner/repo", commit_sha: "c".repeat(40) });
  });
});

describe("parseCodeAuthorshipResponse digest_match semantics", () => {
  it("reports advisory-vs-server agreement while binding the server subject", () => {
    const result = parseCodeAuthorshipResponse({
      envelope: serverEnvelope({
        serverHex: "2".repeat(64),
        captureLayer: "github_sha_pull",
        advisory: ADVISORY,
        digestMatch: false,
      }),
      receipt: {},
      kid: "k",
      jwks_url: "u",
      server_digest: "sha256:" + "2".repeat(64),
      digest_match: false,
    });
    // Advisory digest disagreed, but the bound subject is still the server digest.
    expect(result.digestMatch).toBe(false);
    expect(result.advisoryClientDigest).toBe(ADVISORY);
    expect(result.subjectDigest).toBe("2".repeat(64));
    expect(subjectMatchesServer(result)).toBe(true);
  });
});

describe("verifyCodeAuthorshipEnvelope capture-layer rule", () => {
  it("treats a github_sha_pull envelope as authoritative and passing", () => {
    const envelope = serverEnvelope({ serverHex: "a".repeat(64), captureLayer: "github_sha_pull" });
    const verification = verifyCodeAuthorshipEnvelope(envelope);
    expect(verification.verdict).toBe("PASS");
    expect(verification.authoritative).toBe(true);
    expect(verification.observationOnly).toBe(false);
    expect(verification.captureLayer).toBe("github_sha_pull");
    expect(verification.subjectDigest).toBe("a".repeat(64));
  });

  it.each([...OBSERVATION_ONLY_CAPTURE_LAYERS])(
    "treats capture_layer=%s as observation only, never authoritative",
    (layer) => {
      const envelope = serverEnvelope({ serverHex: "a".repeat(64), captureLayer: layer });
      const verification = verifyCodeAuthorshipEnvelope(envelope);
      expect(verification.verdict).toBe("REJECT");
      expect(verification.authoritative).toBe(false);
      expect(verification.observationOnly).toBe(true);
      expect(verification.reasons).toContain("observation_capture_layer_not_authoritative");
    },
  );

  it("rejects an envelope missing the subject digest", () => {
    const envelope = serverEnvelope({
      serverHex: "a".repeat(64),
      captureLayer: "github_sha_pull",
      includeSubjectDigest: false,
    });
    const verification = verifyCodeAuthorshipEnvelope(envelope);
    expect(verification.verdict).toBe("REJECT");
    expect(verification.reasons).toContain("code_authorship_missing_subject_digest");
  });

  it("rejects a wrong predicateType", () => {
    const envelope = serverEnvelope({
      serverHex: "a".repeat(64),
      captureLayer: "github_sha_pull",
      predicateType: "https://example.com/wrong/v1",
    });
    const verification = verifyCodeAuthorshipEnvelope(envelope);
    expect(verification.verdict).toBe("REJECT");
    expect(verification.reasons).toContain("code_authorship_wrong_predicate_type");
  });

  it("treats an unknown capture layer as not authoritative", () => {
    const envelope = serverEnvelope({ serverHex: "a".repeat(64), captureLayer: "network_proxy" });
    const verification = verifyCodeAuthorshipEnvelope(envelope);
    expect(verification.verdict).toBe("REJECT");
    expect(verification.authoritative).toBe(false);
    expect(verification.observationOnly).toBe(false);
    expect(verification.reasons).toContain("code_authorship_capture_layer_not_github_sha_pull");
  });

  it("rejects a non-object envelope cleanly", () => {
    const verification = verifyCodeAuthorshipEnvelope(null);
    expect(verification.verdict).toBe("REJECT");
    expect(verification.reasons).toContain("envelope_not_an_object");
  });
});
