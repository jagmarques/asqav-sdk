/**
 * The CommonJS build defines RawFloat once, so entries mixed in one process
 * give the verdicts ESM gives. Builds with the package.json build flags.
 */

import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const ROOT = resolve(__dirname, "..");
const VECTORS = resolve(ROOT, "..", "verifier");
const AERF = join(VECTORS, "conformance-vectors", "aerf-up-01-genesis-happy-path");
const DEPTH = JSON.parse(readFileSync(join(VECTORS, "axis-parity-cases.json"), "utf-8")).nesting_depth as {
  envelope: Record<string, unknown>;
  jwks: Record<string, unknown>;
  cases: Array<{ name: string; wrappers: number; leaf: string; expect: { structure: string } }>;
};

// The two published entries a caller mixes, as either module format loads them
interface Entries {
  sdk: {
    verifyReceiptOffline: (r: unknown, jwks: unknown) => {
      verdict: string;
      axes: Array<{ axis: string; result: string }>;
    };
  };
  ver: {
    parseJsonPreservingFloats: (text: string) => unknown;
    RawFloat: new (value: number) => object;
  };
}

let outDir = "";

beforeAll(() => {
  const cache = join(ROOT, "node_modules", ".cache");
  mkdirSync(cache, { recursive: true });
  outDir = mkdtempSync(join(cache, "asqav-rawfloat-"));
  const pkg = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf-8")) as { scripts: { build: string } };
  const [tool, ...args] = pkg.scripts.build.split(/\s+/);
  // Same entries and flags as `npm run build`, minus the slow type declarations
  const flags = args.filter((a) => a !== "--dts" && a !== "--clean");
  const run = spawnSync(join(ROOT, "node_modules", ".bin", tool), [...flags, "--out-dir", outDir], {
    cwd: ROOT,
    encoding: "utf-8",
  });
  if (run.status !== 0) throw new Error(`build failed: ${run.stdout}${run.stderr}`);
}, 120_000);

afterAll(() => {
  if (outDir) rmSync(outDir, { recursive: true, force: true });
});

function filesUnder(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? filesUnder(join(dir, e.name)) : [join(dir, e.name)],
  );
}

function rawFloatDefinitions(ext: string): number {
  const pattern = /(?:var|let|const)\s+RawFloat\s*=\s*class\b|\bclass\s+RawFloat\b/g;
  return filesUnder(outDir)
    .filter((f) => f.endsWith(ext))
    .reduce((n, f) => n + (readFileSync(f, "utf-8").match(pattern)?.length ?? 0), 0);
}

async function load(format: "cjs" | "esm"): Promise<Entries> {
  if (format === "cjs") {
    const req = createRequire(join(outDir, "loader.js"));
    return { sdk: req("./index.js"), ver: req("./verifier/index.js") };
  }
  const url = (file: string): string => pathToFileURL(join(outDir, file)).href;
  return { sdk: await import(url("index.mjs")), ver: await import(url("verifier/index.mjs")) };
}

const nest = (wrappers: number, leaf: string): string =>
  '{"n":'.repeat(wrappers) + `{"leaf":${leaf}}` + "}".repeat(wrappers);

describe("the build emits RawFloat once", () => {
  it("defines one class in the CommonJS files", () => {
    expect(rawFloatDefinitions(".js")).toBe(1);
  });

  it("defines one class in the ESM files", () => {
    expect(rawFloatDefinitions(".mjs")).toBe(1);
  });
});

describe.each(["cjs", "esm"] as const)("%s entries mixed in one process", (format) => {
  let entries: Entries;
  beforeAll(async () => {
    entries = await load(format);
  });

  it("parses a decimal to the RawFloat the verifier entry exports", () => {
    const parsed = entries.ver.parseJsonPreservingFloats('{"leaf":1.5}') as Record<string, unknown>;
    expect(parsed.leaf).toBeInstanceOf(entries.ver.RawFloat);
  });

  it("verifies aerf-up-01 parsed by the verifier entry", () => {
    const parse = entries.ver.parseJsonPreservingFloats;
    const receipt = parse(readFileSync(join(AERF, "receipt.json"), "utf-8"));
    const keys = parse(readFileSync(join(AERF, "keys.json"), "utf-8"));
    const result = entries.sdk.verifyReceiptOffline(receipt, keys);
    expect(result.verdict, JSON.stringify(result.axes)).toBe("verified");
  });

  for (const c of DEPTH.cases) {
    it(`applies the shared depth cap: ${c.name}`, () => {
      const base = JSON.stringify(DEPTH.envelope).slice(0, -1);
      const text = `${base},"junk":${nest(c.wrappers, c.leaf)}}`;
      const receipt = entries.ver.parseJsonPreservingFloats(text);
      const result = entries.sdk.verifyReceiptOffline(receipt, DEPTH.jwks);
      expect(result.axes.find((a) => a.axis === "structure")?.result).toBe(c.expect.structure);
    });
  }
});
