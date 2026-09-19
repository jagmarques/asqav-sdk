// Independent RFC 9964 read of the flagship ML-DSA-65 vector (criterion 534)
// jose + WebCrypto only for key load and verify; repo JCS only for bytes
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { calculateJwkThumbprint, importJWK } from "jose";

import { asqavJcs } from "../src/verifier/canonical.js";
import { b64decode } from "../src/verifier/vrShim.js";

const KAT = resolve(__dirname, "..", "..", "verifier", "conformance-vectors", "asqav-06-mldsa65-payload-prod");
const PUBLISHED_JWKS = resolve(__dirname, "..", "..", "verifier", "docs", "fixtures", "published-jwks.json");

// WebCrypto learns ML-DSA-65 at Node 24.7.0; the matrix typescript job stays
// on 20/22, so skip there. The rfc9964 CI job on 24 is the enforcing runner
function nodeSupportsMldsa65(): boolean {
  const [major, minor] = process.versions.node.split(".").map(Number);
  return major > 24 || (major === 24 && minor >= 7);
}

function loadJson(dir: string, file: string): Record<string, unknown> {
  return JSON.parse(readFileSync(join(dir, file), "utf-8")) as Record<string, unknown>;
}

// Hand-rolled RFC 7638 over the docs-published copy of the key, no jose
// involved, so the expected digest cannot be a restatement of the edit
function publishedThumbprint(): string {
  const fixture = JSON.parse(readFileSync(PUBLISHED_JWKS, "utf-8")) as { keys: Record<string, string>[] };
  const entry = fixture.keys.find((k) => k.kid === "mxYqaLBR_T76ThNw0Kiekw");
  if (!entry) throw new Error("published fixture lacks the flagship key");
  const raw = Buffer.from(entry.public_key, "base64");
  if (raw.length !== 1952) throw new Error(`published pub is ${raw.length} bytes, want 1952`);
  const pub = raw.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  const canonical = `{"alg":"ML-DSA-65","kty":"AKP","pub":"${pub}"}`;
  return createHash("sha256").update(canonical, "utf8").digest("base64url");
}

describe.skipIf(!nodeSupportsMldsa65())("rfc9964 independent read of asqav-06-mldsa65-payload-prod", () => {
  it("verifies via jose importJWK plus WebCrypto ML-DSA-65", async () => {
    const receipt = loadJson(KAT, "receipt.json");
    const jwk = (loadJson(KAT, "jwks.json") as unknown as { keys: Record<string, string>[] }).keys[0];
    // importJWK can return raw bytes for symmetric keys; ML-DSA-65 is
    // asymmetric, so require the WebCrypto key form the verify call needs
    const rawKey = await importJWK(jwk, "ML-DSA-65");
    if (rawKey instanceof Uint8Array) throw new Error("importJWK returned raw bytes, want a CryptoKey");
    const key: Parameters<typeof crypto.subtle.verify>[1] = rawKey;
    const payload = receipt.payload as Record<string, unknown>;
    const msg = asqavJcs(payload);
    const sig = b64decode((receipt.signature as { sig: string }).sig);
    const ok = await crypto.subtle.verify({ name: "ML-DSA-65" }, key, sig, msg);

    const flipped = new Uint8Array(sig);
    flipped[flipped.length - 1] ^= 0x01;
    const okFlipped = await crypto.subtle.verify({ name: "ML-DSA-65" }, key, flipped, msg);

    const tampered = { ...payload, decision: payload.decision === "allow" ? "deny" : "allow" };
    const okTampered = await crypto.subtle.verify({ name: "ML-DSA-65" }, key, sig, asqavJcs(tampered));

    const thumb = await calculateJwkThumbprint(jwk, "sha256");
    const thumbOk = thumb === publishedThumbprint();

    console.log(`VERIFY (jose importJWK + WebCrypto ML-DSA-65): ${ok}`);
    console.log(`CONTROL flipped-sig:      ${okFlipped}`);
    console.log(`CONTROL tampered-payload: ${okTampered}`);
    console.log(`thumbprint == published key_thumbprint: ${thumbOk}`);

    expect(ok).toBe(true);
    expect(okFlipped).toBe(false);
    expect(okTampered).toBe(false);
    expect(thumbOk).toBe(true);
  });
});
