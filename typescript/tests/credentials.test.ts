import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EventEmitter } from "node:events";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { PassThrough } from "node:stream";
import { promptHidden, runCli } from "../src/cli.js";
import { _resetForTests } from "../src/index.js";
import {
  credentialsPath,
  loadCredentials,
  resolveApiBase,
  resolveApiKey,
  saveCredentials,
  validatedPath,
} from "../src/credentials.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const AGENT_CREATED = {
  agent_id: "agt_demo",
  name: "n",
  public_key: "pk",
  key_id: "kid",
  algorithm: "ml-dsa-65",
  capabilities: [],
  created_at: "2026-01-01T00:00:00Z",
};

const SIGN_RESPONSE = {
  signature: "sig_b64",
  signature_id: "sig_demo",
  action_id: "act_1",
  timestamp: "2026-01-01T00:00:00Z",
  verification_url: "https://verify.example/sig_demo",
  algorithm: "ml-dsa-65",
};

describe("credentials layer + onboarding CLI (TypeScript)", () => {
  let exitSpy: ReturnType<typeof vi.spyOn>;
  let stdoutSpy: ReturnType<typeof vi.spyOn>;
  let stderrSpy: ReturnType<typeof vi.spyOn>;
  const tempDirs: string[] = [];
  const savedEnv: Record<string, string | undefined> = {};
  const ENV_KEYS = ["ASQAV_CREDENTIALS_PATH", "ASQAV_API_KEY", "ASQAV_API_BASE", "HOME"];

  beforeEach(() => {
    _resetForTests();
    for (const k of ENV_KEYS) savedEnv[k] = process.env[k];

    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "asqav-creds-"));
    tempDirs.push(tmp);
    // Isolate the file-backed chain so no test reads a real ~/.asqav/credentials.
    process.env.ASQAV_CREDENTIALS_PATH = path.join(tmp, "credentials");
    delete process.env.ASQAV_API_KEY;
    delete process.env.ASQAV_API_BASE;

    exitSpy = vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
      throw new Error(`__EXIT__${code ?? 0}`);
    }) as never);
    stdoutSpy = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    stderrSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
  });

  afterEach(() => {
    for (const k of ENV_KEYS) {
      if (savedEnv[k] === undefined) delete process.env[k];
      else process.env[k] = savedEnv[k];
    }
    vi.restoreAllMocks();
    for (const dir of tempDirs.splice(0)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  function output(): string {
    const out = stdoutSpy.mock.calls.map((c: unknown[]) => String(c[0])).join("");
    const err = stderrSpy.mock.calls.map((c: unknown[]) => String(c[0])).join("");
    return out + err;
  }

  // Switch to the default ~/.asqav path under a temp HOME (drop the env override),
  // mirroring the Python `home` fixture.
  function useDefaultPath(): string {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), "asqav-home-"));
    tempDirs.push(home);
    delete process.env.ASQAV_CREDENTIALS_PATH;
    delete process.env.ASQAV_API_KEY;
    delete process.env.ASQAV_API_BASE;
    process.env.HOME = home;
    return home;
  }

  // Run a callback in an empty temp cwd (for `init` framework detection).
  async function inTempCwd(fn: (cwd: string) => Promise<void>): Promise<void> {
    const cwd = fs.mkdtempSync(path.join(os.tmpdir(), "asqav-cwd-"));
    tempDirs.push(cwd);
    const orig = process.cwd();
    process.chdir(cwd);
    try {
      await fn(cwd);
    } finally {
      process.chdir(orig);
    }
  }

  // === credentials module ===

  it("default path lives under the home dir", () => {
    const home = useDefaultPath();
    expect(credentialsPath()).toBe(path.join(home, ".asqav", "credentials"));
  });

  it("load on a missing file returns {}", () => {
    useDefaultPath();
    expect(loadCredentials()).toEqual({});
  });

  it("resolveApiKey on a missing file returns null", () => {
    useDefaultPath();
    expect(resolveApiKey()).toBeNull();
  });

  it("save/load roundtrip", () => {
    useDefaultPath();
    saveCredentials("sk_file_123", "https://example.test/api/v1");
    expect(loadCredentials()).toEqual({
      api_key: "sk_file_123",
      api_base: "https://example.test/api/v1",
    });
  });

  it("save without api_base omits the key", () => {
    useDefaultPath();
    saveCredentials("sk_only");
    expect(loadCredentials()).toEqual({ api_key: "sk_only" });
  });

  it("saved file mode is 0600", () => {
    useDefaultPath();
    const file = saveCredentials("sk_secret");
    expect(fs.statSync(file).mode & 0o777).toBe(0o600);
  });

  it("credentials dir mode is 0700", () => {
    const home = useDefaultPath();
    saveCredentials("sk_secret");
    expect(fs.statSync(path.join(home, ".asqav")).mode & 0o777).toBe(0o700);
  });

  it("corrupt file returns {} and resolves no key", () => {
    const home = useDefaultPath();
    const dir = path.join(home, ".asqav");
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, "credentials"), "{not valid json");
    expect(loadCredentials()).toEqual({});
    expect(resolveApiKey()).toBeNull();
  });

  it("resolve precedence: arg beats env beats file", () => {
    useDefaultPath();
    saveCredentials("sk_file");
    process.env.ASQAV_API_KEY = "sk_env";
    expect(resolveApiKey("sk_arg")).toBe("sk_arg");
    expect(resolveApiKey()).toBe("sk_env");
    delete process.env.ASQAV_API_KEY;
    expect(resolveApiKey()).toBe("sk_file");
  });

  it("resolveApiBase precedence and default", () => {
    useDefaultPath();
    expect(resolveApiBase()).toBe("https://api.asqav.com/api/v1");
    saveCredentials("sk", "https://file.test/api");
    expect(resolveApiBase()).toBe("https://file.test/api");
    process.env.ASQAV_API_BASE = "https://env.test/api";
    expect(resolveApiBase()).toBe("https://env.test/api");
    expect(resolveApiBase("https://arg.test/api")).toBe("https://arg.test/api");
  });

  it("ASQAV_CREDENTIALS_PATH overrides the location", () => {
    const home = useDefaultPath();
    const override = path.join(home, "custom", "creds.json");
    process.env.ASQAV_CREDENTIALS_PATH = override;
    expect(credentialsPath()).toBe(override);
    saveCredentials("sk_override");
    expect(fs.existsSync(override)).toBe(true);
    expect(resolveApiKey()).toBe("sk_override");
  });

  // === path-injection regression (Trustabl #375) ===

  it("rejects a traversal ASQAV_CREDENTIALS_PATH and writes nothing outside", () => {
    const home = useDefaultPath();
    // String concat (not path.join) so the literal ".." survives into the env value.
    const traversal = `${home}/../escaped_creds.json`;
    process.env.ASQAV_CREDENTIALS_PATH = traversal;
    expect(() => credentialsPath()).toThrow(/path traversal/);
    expect(() => saveCredentials("sk_evil")).toThrow(/path traversal/);
    expect(fs.existsSync(path.resolve(traversal))).toBe(false);
  });

  it("load on a traversal path returns {} and leaks nothing", () => {
    const home = useDefaultPath();
    const secret = path.resolve(`${home}/../secret.json`);
    fs.writeFileSync(secret, JSON.stringify({ api_key: "sk_leaked" }));
    process.env.ASQAV_CREDENTIALS_PATH = `${home}/../secret.json`;
    expect(loadCredentials()).toEqual({});
    expect(resolveApiKey()).toBeNull();
  });

  it("validatedPath rejects null bytes", () => {
    expect(() => validatedPath("creds\x00.json", "X")).toThrow(/null bytes/);
  });

  it("validatedPath rejects traversal components", () => {
    expect(() => validatedPath("/a/../b", "X")).toThrow(/path traversal/);
    expect(() => validatedPath("../creds", "X")).toThrow(/path traversal/);
  });

  it("validatedPath expands ~ and allows legit absolute paths", () => {
    const home = useDefaultPath();
    expect(validatedPath("~/creds.json", "X")).toBe(path.join(home, "creds.json"));
    const abs = path.join(home, "custom", "creds.json");
    expect(validatedPath(abs, "X")).toBe(abs);
  });

  // === login command ===

  it("login saves credentials after validation", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse({ agents: [] }));
    await runCli(["login", "--api-key", "sk_test_123"]);
    expect(output()).toContain("Saved API key");
    const saved = JSON.parse(fs.readFileSync(process.env.ASQAV_CREDENTIALS_PATH!, "utf-8"));
    expect(saved.api_key).toBe("sk_test_123");
  });

  it("login does not save on validation failure", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      jsonResponse({ error: "Invalid API key" }, 401),
    );
    try {
      await runCli(["login", "--api-key", "sk_bad"]);
    } catch {
      // exit
    }
    expect(output()).toContain("validation failed");
    expect(exitSpy).toHaveBeenCalledWith(1);
    expect(fs.existsSync(process.env.ASQAV_CREDENTIALS_PATH!)).toBe(false);
  });

  it("login refuses overwrite without --force", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({ agents: [] }));
    await runCli(["login", "--api-key", "sk_one"]);
    try {
      await runCli(["login", "--api-key", "sk_two"]);
    } catch {
      // exit
    }
    expect(output()).toContain("already exists");
    const saved = JSON.parse(fs.readFileSync(process.env.ASQAV_CREDENTIALS_PATH!, "utf-8"));
    expect(saved.api_key).toBe("sk_one");
  });

  it("login --force overwrites", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({ agents: [] }));
    await runCli(["login", "--api-key", "sk_one"]);
    await runCli(["login", "--api-key", "sk_two", "--force"]);
    const saved = JSON.parse(fs.readFileSync(process.env.ASQAV_CREDENTIALS_PATH!, "utf-8"));
    expect(saved.api_key).toBe("sk_two");
  });

  // === whoami / status commands ===

  it("whoami reports the env source", async () => {
    process.env.ASQAV_API_KEY = "sk_env";
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      jsonResponse({ agents: [{ agent_id: "a1" }, { agent_id: "a2" }] }),
    );
    await runCli(["whoami"]);
    expect(output()).toContain("Key source: env");
    expect(output()).toContain("Key valid");
    expect(output()).toContain("2 agent(s)");
  });

  it("whoami reports the file source", async () => {
    fs.writeFileSync(
      process.env.ASQAV_CREDENTIALS_PATH!,
      JSON.stringify({ api_key: "sk_file" }),
    );
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse([]));
    await runCli(["whoami"]);
    expect(output()).toContain("Key source: file");
  });

  it("whoami reports the arg source", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse([]));
    await runCli(["whoami", "--api-key", "sk_arg"]);
    expect(output()).toContain("Key source: arg");
  });

  it("whoami with no key exits nonzero", async () => {
    try {
      await runCli(["whoami"]);
    } catch {
      // exit
    }
    expect(output()).toContain("asqav login");
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it("whoami with a rejected key exits nonzero", async () => {
    process.env.ASQAV_API_KEY = "sk_bad";
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      jsonResponse({ error: "Invalid API key" }, 401),
    );
    try {
      await runCli(["whoami"]);
    } catch {
      // exit
    }
    expect(output()).toContain("rejected");
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it("status alias reports the source", async () => {
    process.env.ASQAV_API_KEY = "sk_env";
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse([]));
    await runCli(["status"]);
    expect(output()).toContain("Key source: env");
  });

  // === init command ===

  it("init prints a snippet without writing", async () => {
    await inTempCwd(async (cwd) => {
      await runCli(["init"]);
      expect(output()).toContain("Detected framework: typescript");
      expect(output()).toContain("govern");
      expect(output()).toContain("agent.sign");
      expect(fs.readdirSync(cwd)).toEqual([]);
    });
  });

  it("init detects the framework from package.json", async () => {
    await inTempCwd(async (cwd) => {
      fs.writeFileSync(
        path.join(cwd, "package.json"),
        JSON.stringify({ dependencies: { openai: "^4.0.0" } }),
      );
      await runCli(["init"]);
      expect(output()).toContain("Detected framework: openai");
      expect(output()).toContain("api:openai:chat");
    });
  });

  it("init --write creates asqav_governance.ts", async () => {
    await inTempCwd(async (cwd) => {
      await runCli(["init", "--write"]);
      const written = path.join(cwd, "asqav_governance.ts");
      expect(fs.existsSync(written)).toBe(true);
      expect(fs.readFileSync(written, "utf-8")).toContain("govern");
    });
  });

  it("init --demo skips without a key", async () => {
    await inTempCwd(async () => {
      await runCli(["init", "--demo"]);
      expect(output()).toContain("skipping demo");
    });
  });

  it("init --demo signs with a resolved key", async () => {
    process.env.ASQAV_API_KEY = "sk_env";
    await inTempCwd(async () => {
      vi.spyOn(globalThis, "fetch")
        .mockResolvedValueOnce(jsonResponse(AGENT_CREATED))
        .mockResolvedValueOnce(jsonResponse(SIGN_RESPONSE));
      await runCli(["init", "--demo"]);
      expect(output()).toContain("sig_demo");
      expect(output()).toContain("asqav verify sig_demo");
    });
  });

  // === promptHidden ===

  it("promptHidden non-TTY reads a line from piped stdin", async () => {
    const stream = new PassThrough();
    const origStdin = process.stdin;
    Object.defineProperty(process, "stdin", { value: stream, configurable: true });
    try {
      const p = promptHidden("Key: ");
      stream.write("sk_piped_123\n");
      stream.end();
      await expect(p).resolves.toBe("sk_piped_123");
    } finally {
      Object.defineProperty(process, "stdin", { value: origStdin, configurable: true });
    }
  });

  it("promptHidden TTY uses raw mode and suppresses echo", async () => {
    const fake = new EventEmitter() as EventEmitter & {
      isTTY: boolean;
      setRawMode: ReturnType<typeof vi.fn>;
      resume: ReturnType<typeof vi.fn>;
      pause: ReturnType<typeof vi.fn>;
    };
    fake.isTTY = true;
    fake.setRawMode = vi.fn();
    fake.resume = vi.fn();
    fake.pause = vi.fn();

    const origStdin = process.stdin;
    Object.defineProperty(process, "stdin", { value: fake, configurable: true });
    try {
      const p = promptHidden("Key: ");
      fake.emit("data", Buffer.from("a"));
      fake.emit("data", Buffer.from("b"));
      fake.emit("data", Buffer.from("\r"));
      await expect(p).resolves.toBe("ab");
      expect(fake.setRawMode).toHaveBeenCalledWith(true);
      expect(fake.setRawMode).toHaveBeenCalledWith(false);
      expect(fake.pause).toHaveBeenCalled();
    } finally {
      Object.defineProperty(process, "stdin", { value: origStdin, configurable: true });
    }
  });

  it("promptHidden TTY handles backspace", async () => {
    const fake = new EventEmitter() as EventEmitter & {
      isTTY: boolean;
      setRawMode: ReturnType<typeof vi.fn>;
      resume: ReturnType<typeof vi.fn>;
      pause: ReturnType<typeof vi.fn>;
    };
    fake.isTTY = true;
    fake.setRawMode = vi.fn();
    fake.resume = vi.fn();
    fake.pause = vi.fn();

    const origStdin = process.stdin;
    Object.defineProperty(process, "stdin", { value: fake, configurable: true });
    try {
      const p = promptHidden("Key: ");
      fake.emit("data", Buffer.from("abc"));
      fake.emit("data", Buffer.from("\x7f"));
      fake.emit("data", Buffer.from("\r"));
      await expect(p).resolves.toBe("ab");
    } finally {
      Object.defineProperty(process, "stdin", { value: origStdin, configurable: true });
    }
  });
});
