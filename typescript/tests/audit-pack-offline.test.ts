// audit-pack verify --offline over the real pack from the Python suite
// fetch and sockets refused, API base pointed at a dead port
import net from "node:net";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { verifyAuditPackOffline, verifyPackEntry } from "../src/auditPackOffline.js";
import { runCli } from "../src/cli.js";

const PACK_PATH = resolve(__dirname, "..", "..", "python", "tests", "fixtures", "audit_pack_offline_2026_10_11.json");
const PACK_TEXT = readFileSync(PACK_PATH, "utf-8");
const [FLAT, FIRST, SECOND] = [0, 1, 2];

type Obj = Record<string, any>;

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

  async function runOffline(packText: string): Promise<[number, Obj]> {
    const dir = mkdtempSync(join(tmpdir(), "asqav-pack-"));
    writeFileSync(join(dir, "pack.json"), packText);
    let code = -1;
    try {
      await runCli(["audit-pack", "verify", "--offline", join(dir, "pack.json"), "--output", "json"]);
    } catch (e) {
      code = Number(String((e as Error).message).replace("__EXIT__", ""));
    }
    const text = stdout.mock.calls.map((c: unknown[]) => String(c[0])).join("");
    return [code, JSON.parse(text) as Obj];
  }

  it("verifies the real pack offline: bundle ok, receipts not verifiable offline, none invalid", async () => {
    const [code, report] = await runOffline(PACK_TEXT);
    expect(code).toBe(2);
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
    expect(report.bundle).toEqual({ status: "invalid", reason: "bundle signature: signature mismatch" });
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
});
