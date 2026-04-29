import { describe, expect, it } from "vitest";

import { isAsqavCloudHost, resolveMode } from "../src/mode.js";

describe("isAsqavCloudHost", () => {
  const cases: Array<[string | null, boolean]> = [
    ["api.asqav.com", true],
    ["API.ASQAV.COM", true],
    ["staging.asqav.com", true],
    ["app.asqav.com", true],
    ["foo.bar.asqav.com", true],
    ["asqav.com", false],
    ["myasqav.com", false],
    ["asqav.com.evil.com", false],
    ["evilasqav.com", false],
    ["localhost", false],
    ["127.0.0.1", false],
    ["", false],
    [null, false],
  ];
  for (const [host, expected] of cases) {
    it(`${JSON.stringify(host)} -> ${expected}`, () => {
      expect(isAsqavCloudHost(host)).toBe(expected);
    });
  }
});

describe("resolveMode precedence", () => {
  it("explicit overrides env and url", () => {
    expect(resolveMode("https://api.asqav.com/api/v1", "full-payload", "hash-only")).toBe(
      "hash-only",
    );
    expect(resolveMode("http://localhost:8000", "hash-only", "full-payload")).toBe(
      "full-payload",
    );
  });

  it("env overrides auto detection", () => {
    expect(resolveMode("http://localhost:8000", "hash-only", "auto")).toBe("hash-only");
    expect(resolveMode("https://api.asqav.com/api/v1", "full-payload", "auto")).toBe(
      "full-payload",
    );
  });

  it("garbage env falls through to url-based auto", () => {
    expect(resolveMode("https://api.asqav.com/api/v1", "banana", "auto")).toBe("hash-only");
  });

  it("invalid explicit raises", () => {
    expect(() => resolveMode(null, null, "banana")).toThrow();
  });
});

describe("resolveMode auto", () => {
  const cases: Array<[string | null, "hash-only" | "full-payload"]> = [
    ["https://api.asqav.com/api/v1", "hash-only"],
    ["https://app.asqav.com", "hash-only"],
    ["https://staging.asqav.com/api/v1", "hash-only"],
    ["http://localhost:8000", "full-payload"],
    ["http://localhost:8000/api/v1", "full-payload"],
    ["https://internal.example.com", "full-payload"],
    ["https://myasqav.com/api/v1", "full-payload"],
    ["https://asqav.com.evil.com", "full-payload"],
    ["https://asqav.com", "full-payload"],
    ["api.asqav.com/api/v1", "hash-only"], // missing scheme
    [null, "full-payload"],
    ["", "full-payload"],
  ];
  for (const [url, expected] of cases) {
    it(`${JSON.stringify(url)} -> ${expected}`, () => {
      expect(resolveMode(url, null, "auto")).toBe(expected);
    });
  }
});
