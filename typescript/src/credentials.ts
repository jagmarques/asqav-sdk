/**
 * File-backed credential layer storing an API key in ~/.asqav/credentials. Resolution mirrors the Python
 * half: explicit argument, then environment, then file.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const DEFAULT_API_BASE = "https://api.asqav.com/api/v1";

export const CREDENTIALS_PATH = path.join(os.homedir(), ".asqav", "credentials");

function expandHome(raw: string): string {
  if (raw === "~") return os.homedir();
  if (raw.startsWith("~/") || raw.startsWith("~\\")) {
    return path.join(os.homedir(), raw.slice(2));
  }
  return raw;
}

/**
 * Sanitize a caller-supplied path before any file I/O: expands a leading "~" and rejects null bytes and
 * traversal components, so an attacker-influenced value cannot redirect I/O. Throws on a rejected path.
 */
export function validatedPath(raw: string, source: string): string {
  if (raw.includes("\0")) {
    throw new Error(`${source} must not contain null bytes`);
  }
  const expanded = expandHome(raw);
  if (expanded.split(/[\\/]+/).includes("..")) {
    throw new Error(`${source} must not contain path traversal ('..')`);
  }
  return expanded;
}

/**
 * Resolve the credentials file location (env override, else ~/.asqav/credentials). ASQAV_CREDENTIALS_PATH
 * is validated before use, so a traversal value is rejected rather than read or written.
 */
export function credentialsPath(): string {
  const override = process.env.ASQAV_CREDENTIALS_PATH;
  if (override) return validatedPath(override, "ASQAV_CREDENTIALS_PATH");
  return path.join(os.homedir(), ".asqav", "credentials");
}

/** Read the credentials file. Missing or corrupt file returns {} (never throws). */
export function loadCredentials(): Record<string, unknown> {
  try {
    const data = JSON.parse(fs.readFileSync(credentialsPath(), "utf-8"));
    if (data !== null && typeof data === "object" && !Array.isArray(data)) {
      return data as Record<string, unknown>;
    }
    return {};
  } catch {
    return {};
  }
}

/** Write the credentials file with mode 0600 under a mode 0700 ~/.asqav dir. */
export function saveCredentials(apiKey: string, apiBase?: string): string {
  const file = credentialsPath();
  const dir = path.dirname(file);
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  fs.chmodSync(dir, 0o700);

  const payload: Record<string, string> = { api_key: apiKey };
  if (apiBase) payload.api_base = apiBase;

  fs.writeFileSync(file, JSON.stringify(payload, null, 2), { mode: 0o600 });
  fs.chmodSync(file, 0o600);
  return file;
}

/** Resolve an API key: explicit arg, then ASQAV_API_KEY env, then credentials file. */
export function resolveApiKey(explicit?: string): string | null {
  if (explicit) return explicit;
  const envKey = process.env.ASQAV_API_KEY;
  if (envKey) return envKey;
  const fileKey = loadCredentials().api_key;
  if (typeof fileKey === "string" && fileKey) return fileKey;
  return null;
}

/** Resolve an API base: explicit arg, then ASQAV_API_BASE env, then file, then default. */
export function resolveApiBase(explicit?: string): string {
  if (explicit) return explicit;
  const envBase = process.env.ASQAV_API_BASE;
  if (envBase) return envBase;
  const fileBase = loadCredentials().api_base;
  if (typeof fileBase === "string" && fileBase) return fileBase;
  return DEFAULT_API_BASE;
}
