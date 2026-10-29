import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { expect, it } from "vitest";

import { ASSERTED, MISSING, SIGNATURE_VERIFIED } from "./oversightAttestation.js";
import { runOne } from "./runner.js";

const CORPUS = resolve(__dirname, "../../..", "verifier", "conformance-vectors");

function load(dir: string, name: string): Record<string, unknown> {
  return JSON.parse(readFileSync(resolve(CORPUS, dir, name), "utf8")) as Record<string, unknown>;
}

function caseDirs(): string[] {
  const manifest = JSON.parse(readFileSync(resolve(CORPUS, "manifest.json"), "utf8")) as {
    dir: string;
  }[];
  const dirs = manifest
    .map((entry) => entry.dir)
    .filter((dir) => typeof load(dir, "expected.json")["oversight_attestation"] === "string");
  expect(dirs).toHaveLength(3);
  return dirs;
}

it("oversightAttestation reports three states through the runner", () => {
  const reports = caseDirs().map((dir) => {
    const expected = load(dir, "expected.json");
    const outcome = runOne(
      resolve(CORPUS, dir),
      expected.format as string,
      expected.outcome as string,
      (expected.reason_code as string) ?? "",
      (expected.failure_class as string) ?? "",
    );
    expect(outcome.ok).toBeTruthy();
    expect(outcome.oversightAttestation).toBe(expected["oversight_attestation"]);
    return outcome.oversightAttestation;
  });
  expect(new Set(reports).size).toBe(3);
  expect(new Set(reports)).toEqual(new Set([MISSING, ASSERTED, SIGNATURE_VERIFIED]));
});

function signatureVerifiedDir(): string {
  const dir = caseDirs().find(
    (name) => load(name, "expected.json")["oversight_attestation"] === SIGNATURE_VERIFIED,
  );
  expect(dir).toBeDefined();
  return dir!;
}

function runPlanted(
  receipt: Record<string, unknown>,
  jwks: Record<string, unknown>,
  expected: Record<string, unknown> = load(signatureVerifiedDir(), "expected.json"),
) {
  const work = mkdtempSync(join(tmpdir(), "asqav-918-"));
  try {
    writeFileSync(join(work, "receipt.json"), JSON.stringify(receipt));
    writeFileSync(join(work, "jwks.json"), JSON.stringify(jwks));
    writeFileSync(join(work, "expected.json"), JSON.stringify(expected));
    return runOne(
      work,
      expected.format as string,
      expected.outcome as string,
      (expected.reason_code as string) ?? "",
      (expected.failure_class as string) ?? "",
    );
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

// NEGCTL: flip one byte of the reviewer's attestation signature; the runner
// reports asserted, so the declared signature-verified state no longer matches
it("NEGCTL: an attestation signature that does not verify is asserted", () => {
  const dir = signatureVerifiedDir();
  const bad = structuredClone(load(dir, "receipt.json"));
  const payload = bad.payload as Record<string, Record<string, Record<string, Record<string, unknown>>>>;
  const signature = payload.reviewer!.attestation!.signature!;
  const raw = Buffer.from(signature.sig as string, "base64url");
  raw[0] = raw[0]! ^ 1;
  signature.sig = raw.toString("base64url");
  const outcome = runPlanted(bad, load(dir, "jwks.json"));
  expect(outcome.oversightAttestation).toBe(ASSERTED);
  expect(outcome.ok).toBe(false);
});

// NEGCTL: carry 058's attestation, signature intact, into a ruling that judges
// another action; the signature binds the ruling, so the claim is asserted
it("NEGCTL: an attestation moved to another ruling is asserted", () => {
  const dir = signatureVerifiedDir();
  const moved = structuredClone(load(dir, "receipt.json"));
  const payload = moved.payload as Record<string, unknown>;
  payload.judged_action_refs = [...(payload.judged_action_refs as string[]), payload.policy_digest as string];
  const outcome = runPlanted(moved, load(dir, "jwks.json"));
  expect(outcome.oversightAttestation).toBe(ASSERTED);
  expect(outcome.ok).toBe(false);
});

// NEGCTL: change the reviewer's role, signature intact; the signature binds the
// reviewer's identity beside the principal, so the claim is asserted
it("NEGCTL: an attestation under another reviewer role is asserted", () => {
  const dir = signatureVerifiedDir();
  const moved = structuredClone(load(dir, "receipt.json"));
  const reviewer = (moved.payload as Record<string, Record<string, unknown>>).reviewer!;
  reviewer.role = reviewer.authority;
  const outcome = runPlanted(moved, load(dir, "jwks.json"));
  expect(outcome.oversightAttestation).toBe(ASSERTED);
  expect(outcome.ok).toBe(false);
});

// NEGCTL: bind the receipt signer's key to the reviewer's principal; the
// receipt still verifies, the attestation signature does not: asserted
it("NEGCTL: a reviewer bound to another key is asserted", () => {
  const dir = signatureVerifiedDir();
  const jwks = load(dir, "jwks.json");
  const [signer, reviewer] = jwks.keys as Record<string, unknown>[];
  reviewer!.pub = signer!.pub;
  reviewer!.public_key = signer!.public_key;
  const outcome = runPlanted(load(dir, "receipt.json"), jwks);
  expect(outcome.actualFailureClass).toBe(load(dir, "expected.json").failure_class);
  expect(outcome.oversightAttestation).toBe(ASSERTED);
  expect(outcome.ok).toBe(false);
});

// NEGCTL: revoke the reviewer's key before the ruling was issued; the
// signature still checks, but a revoked key backs no claim: asserted
it("NEGCTL: a revoked reviewer key is asserted", () => {
  const dir = signatureVerifiedDir();
  const jwks = load(dir, "jwks.json");
  const reviewer = (jwks.keys as Record<string, unknown>[])[1]!;
  reviewer.status = "revoked";
  reviewer.revoked_at = "2026-01-01T00:00:00+00:00";
  const outcome = runPlanted(load(dir, "receipt.json"), jwks);
  expect(outcome.actualFailureClass).toBe(load(dir, "expected.json").failure_class);
  expect(outcome.oversightAttestation).toBe(ASSERTED);
  expect(outcome.ok).toBe(false);
});

// NEGCTL: declare the state as a list, not a string; the runner never reads
// that declaration as met, so the vector fails though the report is right
it("NEGCTL: a state declared as a non-string never matches", () => {
  const dir = signatureVerifiedDir();
  const expected = { ...load(dir, "expected.json"), oversight_attestation: [SIGNATURE_VERIFIED] };
  const outcome = runPlanted(load(dir, "receipt.json"), load(dir, "jwks.json"), expected);
  expect(outcome.oversightAttestation).toBe(SIGNATURE_VERIFIED);
  expect(outcome.ok).toBe(false);
});
