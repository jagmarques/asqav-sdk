/** A reused buffer never reads an earlier verification result. */

import { generateKeyPairSync, sign } from "node:crypto";
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";
import { describe, expect, it } from "vitest";

import { verifySignature } from "../src/verifier/crypto.js";

function signed(message: Uint8Array): { pk: Uint8Array; sig: Uint8Array } {
  const { publicKey, privateKey } = generateKeyPairSync("ed25519");
  const x = publicKey.export({ format: "jwk" }).x as string;
  return {
    pk: new Uint8Array(Buffer.from(x, "base64url")),
    sig: new Uint8Array(sign(null, message, privateKey)),
  };
}

const enc = (text: string) => new TextEncoder().encode(text);

describe("verifySignature with reused buffers", () => {
  it("verifies again after the message is rewritten in place", () => {
    const { pk, sig } = signed(enc("receipt A"));
    const msg = enc("receipt A");
    expect(verifySignature("Ed25519", pk, msg, sig).result).toBe("PASS");
    msg.set(enc("receipt B"));
    expect(verifySignature("Ed25519", pk, msg, sig).result).toBe("FAIL");
  });

  it("verifies again after the signature is rewritten in place, and from fresh arrays", () => {
    const { pk, sig } = signed(enc("receipt A"));
    const held = { pk, msg: enc("receipt A"), sig: new Uint8Array(sig) };
    expect(verifySignature("Ed25519", held.pk, held.msg, held.sig).result).toBe("PASS");
    held.sig[0] ^= 1;
    expect(verifySignature("Ed25519", held.pk, held.msg, held.sig).result).toBe("FAIL");
    const forged = new Uint8Array(sig);
    forged[0] ^= 1;
    expect(verifySignature("Ed25519", pk, enc("receipt A"), forged).result).toBe("FAIL");
  });

  it("gives the same answer for the same input", () => {
    const { pk, sig } = signed(enc("receipt A"));
    const first = verifySignature("Ed25519", pk, enc("receipt A"), sig);
    expect(verifySignature("Ed25519", pk, enc("receipt A"), sig)).toEqual(first);
    expect(first.result).toBe("PASS");
  });

  it("answers a non-Uint8Array input the same whether or not the cache is warm", () => {
    const { pk, sig } = signed(enc("receipt E"));
    const msg = enc("receipt E");
    verifySignature("ES256", enc("x"), enc("x"), enc("x"));
    const inputs = [Array.from(pk), new Uint8ClampedArray(pk), new Uint16Array(pk)] as unknown as Uint8Array[];
    const cold = inputs.map((key) => verifySignature("Ed25519", key, msg, sig));
    expect(verifySignature("Ed25519", pk, msg, sig).result).toBe("PASS");
    expect(inputs.map((key) => verifySignature("Ed25519", key, msg, sig))).toEqual(cold);
  });

  it("never serves a verdict reached on another input type to a Uint8Array", () => {
    const { publicKey, secretKey } = ml_dsa65.keygen(new Uint8Array(32).fill(7));
    const msg = enc("receipt F");
    const sig = ml_dsa65.sign(msg, secretKey);
    verifySignature("Ed25519", enc("x"), enc("x"), enc("x"));
    verifySignature("ML-DSA-65", publicKey, new Uint16Array(msg) as unknown as Uint8Array, sig);
    expect(verifySignature("ML-DSA-65", publicKey, msg, sig).result).toBe("PASS");
  });
});
