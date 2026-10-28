import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, it } from "vitest";

import { ADAPTERS, verify } from "./index.js";
import {
  ASSERTED,
  MISSING,
  SIGNATURE_VERIFIED,
  oversightAttestation,
} from "./oversightAttestation.js";

const CORPUS = resolve(__dirname, "../../..", "verifier", "conformance-vectors");

function load(dir: string, name: string): Record<string, unknown> {
  return JSON.parse(readFileSync(resolve(CORPUS, dir, name), "utf8")) as Record<string, unknown>;
}

function caseDirs(): string[] {
  const manifest = JSON.parse(readFileSync(resolve(CORPUS, "manifest.json"), "utf8")) as {
    dir: string;
  }[];
  const dirs = manifest.map((entry) => entry.dir).filter((dir) => dir.includes("oversight-attestation-"));
  expect(dirs).toHaveLength(3);
  return dirs;
}

function signatureOk(receipt: Record<string, unknown>, jwks: Record<string, unknown>): boolean {
  const result = verify(receipt, ADAPTERS, jwks);
  return result.axes.find((axis) => axis.axis === "signature")?.result === "PASS";
}

it("oversight_attestation reports three states", () => {
  const reports = caseDirs().map((dir) => {
    const receipt = load(dir, "receipt.json");
    const expected = load(dir, "expected.json").oversight_attestation;
    const report = oversightAttestation(receipt.payload, signatureOk(receipt, load(dir, "jwks.json")));
    expect(report).toBe(expected);
    return report;
  });
  expect(new Set(reports).size).toBe(3);
  expect(new Set(reports)).toEqual(new Set([MISSING, ASSERTED, SIGNATURE_VERIFIED]));
});

// NEGCTL: flip the signature-verified vector's receipt signature; the same
// shipped report calls the claim asserted, never signature-verified
it("NEGCTL: a receipt whose signature does not verify is asserted", () => {
  const dir = caseDirs().find(
    (name) => load(name, "expected.json").oversight_attestation === SIGNATURE_VERIFIED,
  );
  expect(dir).toBeDefined();
  const receipt = load(dir!, "receipt.json");
  const jwks = load(dir!, "jwks.json");
  expect(signatureOk(receipt, jwks)).toBeTruthy();
  const bad = structuredClone(receipt);
  const signature = bad.signature as Record<string, unknown>;
  const raw = Buffer.from(signature.sig as string, "base64url");
  raw[0] = raw[0]! ^ 1;
  signature.sig = raw.toString("base64url");
  expect(signatureOk(bad, jwks)).toBe(false);
  const report = oversightAttestation(bad.payload, signatureOk(bad, jwks));
  expect(report).toBe(ASSERTED);
  expect(report).not.toBe(SIGNATURE_VERIFIED);
});
