// audit-pack verify --offline over the real pack from the Python suite
// fetch and sockets refused, API base pointed at a dead port
import net from "node:net";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";
import { signedBundlePayload, verifyAuditPackOffline, verifyPackEntry } from "../src/auditPackOffline.js";
import { runCli } from "../src/cli.js";
import { asqavJcs } from "../src/verifier/canonical.js";
import { sha256Hex } from "../src/verifier/crypto.js";

const PACK_PATH = resolve(__dirname, "..", "..", "python", "tests", "fixtures", "audit_pack_offline_2026_10_11.json");
const PACK_TEXT = readFileSync(PACK_PATH, "utf-8");
const [FLAT, FIRST, SECOND] = [0, 1, 2];

type Obj = Record<string, any>;

const ORG = "11111111-2222-4333-8444-555555555555";
const AGENT = "agt_minted";
const b64 = (b: Uint8Array): string => Buffer.from(b).toString("base64");

function signEntry(secretKey: Uint8Array, payload: Obj): Obj {
  const sig = b64(ml_dsa65.sign(asqavJcs(payload), secretKey));
  return {
    signature_id: `sig_${payload.seq}_${payload.decision}`,
    agent_id: AGENT,
    payload,
    signature: { alg: "ML-DSA-65", kid: ORG, sig },
    anchors: [{ type: "rfc3161", value: Buffer.from("token").toString("base64") }],
  };
}

function seal(pack: Obj, secretKey: Uint8Array, publicKey: Uint8Array): Obj {
  pack.receipt_count = pack.receipts.length;
  const msg = asqavJcs(signedBundlePayload(pack));
  return Object.assign(pack, {
    bundle_digest: `sha256:${sha256Hex(msg)}`,
    bundle_signature: b64(ml_dsa65.sign(msg, secretKey)),
    bundle_signature_algorithm: "ML-DSA-65",
    bundle_public_key: b64(publicKey),
  });
}

// A three-receipt chain sealed with its key, plus the holder's key set
function minted(): { pack: Obj; jwks: Obj; keys: { secretKey: Uint8Array; publicKey: Uint8Array } } {
  const keys = ml_dsa65.keygen();
  const receipts: Obj[] = [];
  let prev = "0".repeat(64);
  for (const seq of [1, 2, 3]) {
    const payload = {
      type: "protectmcp:decision", v: 1, seq, agent_id: AGENT,
      issued_at: `2026-06-19T00:00:0${seq}.000000Z`, issuer_id: ORG, org_id: ORG,
      action_ref: `sha256:${"a".repeat(64)}`, payload_digest: { hash: "b".repeat(64), size: 1 },
      policy_digest: `sha256:${"c".repeat(64)}`, previousReceiptHash: prev,
      decision: "allow", expires_at: "2099-01-01T00:00:00.000000Z",
    };
    receipts.push(signEntry(keys.secretKey, payload));
    prev = sha256Hex(asqavJcs(payload));
  }
  const key = {
    kid: "minted-key-01", issuer_id: ORG, org_id: ORG, agent_id: AGENT,
    alg: "ML-DSA-65", status: "active", public_key: b64(keys.publicKey),
  };
  const pack: Obj = {
    organization_id: ORG, start: "2026-06-19T00:00:00", end: "2026-06-20T00:00:00",
    agent_id: null, only_compliance: true, algorithm_registry_version: "2026-05-04.v1",
    receipts, revocation_manifest: [], regime_mapping: {}, jwks: { keys: [key] },
  };
  return { pack: seal(pack, keys.secretKey, keys.publicKey), jwks: { keys: [{ ...key }] }, keys };
}

function editedPack(edit: (pack: Obj) => void): string {
  const pack = JSON.parse(PACK_TEXT) as Obj;
  edit(pack);
  return JSON.stringify(pack);
}

describe("audit-pack verify --offline", () => {
  let stdout: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.stubGlobal("fetch", () => Promise.reject(new Error("network refused by test")));
    vi.spyOn(net.Socket.prototype, "connect").mockImplementation(() => {
      throw new Error("network refused by test");
    });
    process.env.ASQAV_API_URL = "http://127.0.0.1:9";
    delete process.env.ASQAV_API_KEY;
    vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
      throw new Error(`__EXIT__${code ?? 0}`);
    }) as never);
    stdout = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    vi.spyOn(process.stderr, "write").mockImplementation(() => true);
  });

  afterEach(() => {
    delete process.env.ASQAV_API_URL;
    vi.unstubAllGlobals();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  async function runOffline(packText: string, ...extra: string[]): Promise<[number, Obj]> {
    const dir = mkdtempSync(join(tmpdir(), "asqav-pack-"));
    writeFileSync(join(dir, "pack.json"), packText);
    let code = -1;
    try {
      await runCli(["audit-pack", "verify", "--offline", join(dir, "pack.json"), "--output", "json", ...extra]);
    } catch (e) {
      code = Number(String((e as Error).message).replace("__EXIT__", ""));
    }
    const text = stdout.mock.calls.map((c: unknown[]) => String(c[0])).join("");
    return [code, JSON.parse(text) as Obj];
  }

  it("verifies the real pack offline: bundle ok, receipts not verifiable offline, none invalid", async () => {
    const dir = mkdtempSync(join(tmpdir(), "asqav-jwks-"));
    writeFileSync(join(dir, "jwks.json"), JSON.stringify(JSON.parse(PACK_TEXT).jwks));
    const [code, report] = await runOffline(PACK_TEXT, "--jwks", join(dir, "jwks.json"));
    expect(code).toBe(2);
    expect(report.status).toBe("not_verifiable_offline");
    expect(report.trust.anchored).toBe(true);
    expect(report.bundle.status).toBe("verified");
    expect(report.summary).toEqual({ total: 3, verified: 0, not_verifiable_offline: 3, invalid: 0 });
    expect(report.receipts[FLAT].reasons[0]).toContain("non-compliance-mode receipt");
    expect(report.receipts[FIRST].signature).toBe("signature valid");
    expect(report.receipts[SECOND].chain).toBe("chain link rederives from predecessor");
    expect(report.receipts[SECOND].reasons[0]).toMatch(/^anchors: .*pending/);
  });

  it("fails a tampered receipt", async () => {
    const [code, report] = await runOffline(editedPack((p) => (p.receipts[SECOND].payload.decision = "deny")));
    expect(code).toBe(1);
    expect(report.receipts[SECOND].status).toBe("invalid");
    expect(report.receipts[SECOND].reasons.some((r: string) => r.startsWith("signature:"))).toBe(true);
  });

  it("fails the untouched successor of a tampered chain link", () => {
    const report = verifyAuditPackOffline(editedPack((p) => (p.receipts[FIRST].payload.nonce = "0".repeat(24))));
    expect(report.exit_code).toBe(1);
    const second = report.receipts[SECOND];
    expect(second.signature).toBe("signature valid");
    expect(second.status).toBe("invalid");
    expect(second.reasons.some((r) => r.startsWith("chain:"))).toBe(true);
  });

  it("fails a tampered bundle signature", async () => {
    const [code, report] = await runOffline(
      editedPack((p) => (p.bundle_signature = (p.bundle_signature[0] === "B" ? "C" : "B") + p.bundle_signature.slice(1))),
    );
    expect(code).toBe(1);
    expect(report.bundle).toMatchObject({ status: "invalid", reason: "bundle signature: signature mismatch" });
  });

  it("fails an edited pack member through the bundle digest", () => {
    const report = verifyAuditPackOffline(editedPack((p) => (p.regime_mapping = { eu_ai_act: [] })));
    expect(report.bundle.reason).toBe("bundle_digest does not match the pack contents");
  });

  it("labels an RFC3161 anchor it cannot check as not verifiable offline, not invalid", () => {
    const pack = JSON.parse(PACK_TEXT) as Obj;
    const entry = pack.receipts[SECOND];
    entry.anchors = [{ type: "rfc3161", value: Buffer.from("token").toString("base64") }];
    const result = verifyPackEntry(entry, pack.jwks, pack.receipts[FIRST].payload);
    expect(result.status).toBe("not_verifiable_offline");
    expect(result.reasons[0]).toContain("rfc3161");
  });

  it("reports a lapsed expiry as a note, not a failure", () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(Date.now() + 3650 * 86400 * 1000);
    const report = verifyAuditPackOffline(PACK_TEXT);
    expect(report.summary.invalid).toBe(0);
    for (const entry of report.receipts.slice(FIRST)) {
      expect(entry.reasons.some((r) => r.startsWith("expiry"))).toBe(false);
      expect(entry.notes[0]).toContain("signed before it, so not a failure");
    }
  });

  it("fails a tampered flat receipt", () => {
    const report = verifyAuditPackOffline(
      editedPack((p) => {
        const msg = Buffer.from(p.receipts[FLAT].message_b64, "base64").toString("utf8").replace("permit", "deny");
        p.receipts[FLAT].message_b64 = Buffer.from(msg).toString("base64");
      }),
    );
    expect(report.receipts[FLAT].status).toBe("invalid");
  });

  it("says the pack only vouches for itself without --jwks", async () => {
    const [code, report] = await runOffline(PACK_TEXT);
    expect(code).toBe(2);
    expect(report.status).toBe("self_consistent");
    expect(report.bundle.key_thumbprint).toMatch(/^sha256:b4777311/);
  });

  it("refuses a --jwks that is not an object", async () => {
    const dir = mkdtempSync(join(tmpdir(), "asqav-bad-"));
    writeFileSync(join(dir, "bad.json"), "[]");
    let code = -1;
    try {
      await runCli(["audit-pack", "verify", "--offline", PACK_PATH, "--jwks", join(dir, "bad.json")]);
    } catch (e) {
      code = Number(String((e as Error).message).replace("__EXIT__", ""));
    }
    expect(code).toBe(1);
  });

  it("fails a stripped bundle signature, an unsupported algorithm and a mismatched one", () => {
    const jwks = JSON.stringify(JSON.parse(PACK_TEXT).jwks);
    const stripped = verifyAuditPackOffline(editedPack((p) => delete p.bundle_signature), jwks);
    expect(stripped.bundle.reason).toBe("pack carries no bundle signature");
    expect(stripped.exit_code).toBe(1);
    const none = verifyAuditPackOffline(editedPack((p) => (p.bundle_signature_algorithm = "none")));
    expect(none.bundle.reason).toBe("unsupported bundle signature algorithm 'none'");
    expect(none.exit_code).toBe(1);
    const ed = verifyAuditPackOffline(editedPack((p) => (p.bundle_signature_algorithm = "Ed25519")));
    expect(ed.bundle.reason).toContain("differs from its key's alg ML-DSA-65");
  });

  it("fails a bundle key outside the key set", () => {
    const report = verifyAuditPackOffline(PACK_TEXT, JSON.stringify({ keys: [] }));
    expect(report.bundle.reason).toBe("bundle key is not in the key set");
    expect(report.exit_code).toBe(1);
  });

  it("fails a bundle key of another org", () => {
    const m = minted();
    m.pack.organization_id = "org-elsewhere";
    seal(m.pack, m.keys.secretKey, m.keys.publicKey);
    const report = verifyAuditPackOffline(JSON.stringify(m.pack), JSON.stringify(m.jwks));
    expect(report.bundle.reason).toBe("bundle signing key belongs to another organization");
  });

  it.each([
    ["revoked", "revoked_at"],
    ["suspended", "suspended_at"],
  ])("leaves a pack whose key was %s after the export not verifiable, not invalid", (status, member) => {
    const m = minted();
    Object.assign(m.jwks.keys[0], { status, [member]: "2027-01-01T00:00:00Z" });
    const report = verifyAuditPackOffline(JSON.stringify(m.pack), JSON.stringify(m.jwks));
    expect(report.bundle).toMatchObject({
      status: "not_verifiable_offline",
      reason: `bundle key ${status} at 2027-01-01T00:00:00Z; nothing proves the bundle was signed before it`,
    });
    expect(report.summary.invalid).toBe(0);
    expect(report.exit_code).toBe(2);
  });

  it("fails a bundle signed by a compromised key", () => {
    const m = minted();
    Object.assign(m.jwks.keys[0], {
      status: "revoked",
      revocation_reason: "compromised",
      revoked_at: "2027-01-01T00:00:00Z",
    });
    const report = verifyAuditPackOffline(JSON.stringify(m.pack), JSON.stringify(m.jwks));
    expect(report.bundle).toMatchObject({ status: "invalid", reason: "bundle signing key compromised" });
    expect(report.exit_code).toBe(1);
  });

  it("leaves a bundle key with no known status not verifiable", () => {
    const m = minted();
    m.jwks.keys[0].status = "unknown";
    const report = verifyAuditPackOffline(JSON.stringify(m.pack), JSON.stringify(m.jwks));
    expect(report.bundle).toMatchObject({
      status: "not_verifiable_offline",
      reason: "bundle signing key status unknown",
    });
    expect(report.exit_code).toBe(2);
  });

  it("fails a bundle whose signed window starts after the key was revoked", () => {
    const m = minted();
    Object.assign(m.jwks.keys[0], { status: "revoked", revoked_at: "2026-06-18T00:00:00Z" });
    const report = verifyAuditPackOffline(JSON.stringify(m.pack), JSON.stringify(m.jwks));
    expect(report.bundle.reason).toBe("bundle key revoked at 2026-06-18T00:00:00Z, before the signed window start");
    expect(report.exit_code).toBe(1);
  });

  it("never verifies a bundle key the pack snapshot lists as revoked", () => {
    const m = minted();
    m.pack.jwks.keys[0].status = "revoked";
    seal(m.pack, m.keys.secretKey, m.keys.publicKey);
    expect(verifyAuditPackOffline(JSON.stringify(m.pack)).bundle.status).toBe("not_verifiable_offline");
  });

  it("does not trust TSA entries carried in the pack", () => {
    const m = minted();
    m.pack.tsa_trust = [b64(new Uint8Array(32))];
    seal(m.pack, m.keys.secretKey, m.keys.publicKey);
    const report = verifyAuditPackOffline(JSON.stringify(m.pack), JSON.stringify(m.jwks));
    expect(report.trust.tsa_carried_not_trusted).toBe(1);
    expect(report.summary.verified).toBe(0);
  });

  it("fails a receipt missing from the middle of a chain", () => {
    const m = minted();
    m.pack.receipts.splice(1, 1);
    seal(m.pack, m.keys.secretKey, m.keys.publicKey);
    const report = verifyAuditPackOffline(JSON.stringify(m.pack), JSON.stringify(m.jwks));
    expect(report.receipts[1].status).toBe("invalid");
    expect(report.receipts[1].reasons[0]).toBe("chain: gap, the predecessor of seq 3 is missing from the pack");
  });

  it("fails a forked chain and a duplicate entry", () => {
    const m = minted();
    const forked = { ...m.pack.receipts[1].payload, decision: "deny" };
    m.pack.receipts.splice(1, 0, signEntry(m.keys.secretKey, forked));
    m.pack.receipts.push(JSON.parse(JSON.stringify(m.pack.receipts[3])));
    seal(m.pack, m.keys.secretKey, m.keys.publicKey);
    const report = verifyAuditPackOffline(JSON.stringify(m.pack), JSON.stringify(m.jwks));
    for (const entry of report.receipts.slice(1, 3)) {
      expect(entry.status).toBe("invalid");
      expect(entry.reasons).toContain("chain: fork, seq 2 appears twice for agent agt_minted");
    }
    expect(report.receipts[4].reasons).toContain("duplicate entry: signature_id sig_3_allow");
    expect(report.exit_code).toBe(1);
  });
});

describe("mapping resolution", () => {
  it("fails a pack whose mapping reference matches no entry", () => {
    const m = minted();
    m.pack.regime_mapping = { dora: ["sig_ghost"] };
    seal(m.pack, m.keys.secretKey, m.keys.publicKey);
    const report = verifyAuditPackOffline(JSON.stringify(m.pack), JSON.stringify(m.jwks));
    expect(report.status).toBe("invalid");
    expect(report.exit_code).toBe(1);
    expect(report.bundle.reason).toContain("sig_ghost");
  });

  it("fails a pack whose receipt_count disagrees with the receipts", () => {
    const m = minted();
    seal(m.pack, m.keys.secretKey, m.keys.publicKey);
    const tampered = JSON.parse(JSON.stringify(m.pack));
    tampered.receipt_count = tampered.receipts.length + 1;
    const report = verifyAuditPackOffline(JSON.stringify(tampered), JSON.stringify(m.jwks));
    expect(report.status).toBe("invalid");
    expect(report.bundle.reason).toContain("receipt_count");
  });
});
