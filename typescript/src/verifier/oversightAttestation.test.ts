import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { expect, it } from "vitest";

import { FAILURE_INVALID, VERDICT_UNVERIFIED } from "./core.js";
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

// NEGCTL: flip the signature-verified receipt's signature; the runner's report
// calls it asserted and the declared signature-verified state no longer matches
it("NEGCTL: a receipt whose signature does not verify is asserted", () => {
  const dir = caseDirs().find(
    (name) => load(name, "expected.json")["oversight_attestation"] === SIGNATURE_VERIFIED,
  );
  expect(dir).toBeDefined();
  const receipt = load(dir!, "receipt.json");
  const expected = load(dir!, "expected.json");
  const bad = structuredClone(receipt);
  const signature = bad.signature as Record<string, unknown>;
  const raw = Buffer.from(signature.sig as string, "base64url");
  raw[0] = raw[0]! ^ 1;
  signature.sig = raw.toString("base64url");
  const work = mkdtempSync(join(tmpdir(), "asqav-918-"));
  try {
    writeFileSync(join(work, "receipt.json"), JSON.stringify(bad));
    writeFileSync(join(work, "jwks.json"), readFileSync(resolve(CORPUS, dir!, "jwks.json")));
    writeFileSync(join(work, "expected.json"), JSON.stringify(expected));
    const outcome = runOne(
      work,
      expected.format as string,
      VERDICT_UNVERIFIED,
      (expected.reason_code as string) ?? "",
      FAILURE_INVALID,
    );
    expect(outcome.actualFailureClass).toBe(FAILURE_INVALID);
    expect(outcome.oversightAttestation).toBe(ASSERTED);
    expect(outcome.oversightAttestation).not.toBe(SIGNATURE_VERIFIED);
    expect(outcome.ok).toBe(false);
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
});
