#!/usr/bin/env bash
# Copyright 2026 Asqav
# SPDX-License-Identifier: Apache-2.0
#
# Conformance corpus freeze, verification path B (criterion 420)
# Re-derives every manifest.lock.json pin through the sha256sum binary and
# wc, an independent code path from the hashlib pass in
# python/tests/test_corpus_lock.py. CI runs both; any drift exits nonzero
#
# Usage: bash verifier/check_corpus_lock.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_NAME="manifest.lock.json"

# SHA-256 of stdin via the system binary (the path-B hash engine)
if command -v sha256sum >/dev/null 2>&1; then
  SHA256_TOOL="sha256sum"
  sha256_stdin() { sha256sum | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  SHA256_TOOL="shasum -a 256"
  sha256_stdin() { shasum -a 256 | cut -d' ' -f1; }
else
  echo "FATAL: neither sha256sum nor shasum found; cannot re-derive the corpus pins" >&2
  exit 1
fi

# Emit one line per pinned file: "<sha256> <bytes> <path>", straight from the lock
lock_entries() {
  python3 - "$1" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1]))
for e in lock["files"]:
    print(e["sha256"], e["bytes"], e["path"])
PY
}

# Canonical bytes of the lock without its digest field, for the digest pin
lock_body_bytes() {
  python3 - "$1" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1]))
body = {k: v for k, v in lock.items() if k != "digest"}
sys.stdout.buffer.write(json.dumps(
    body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
).encode("utf-8"))
PY
}

check_lock() {
  local corpus_root="$1"
  local lock="$corpus_root/$LOCK_NAME"
  local failures=0

  [[ -f "$lock" ]] || { echo "FATAL: missing lock file: $lock" >&2; exit 1; }

  while read -r want_sha want_bytes rel_path; do
    local file="$corpus_root/$rel_path"
    if [[ ! -f "$file" ]]; then
      echo "DRIFT [$corpus_root] $rel_path: pinned file missing" >&2
      failures=$((failures + 1))
      continue
    fi
    local got_sha
    got_sha="$(sha256_stdin < "$file" | tr '[:upper:]' '[:lower:]')"
    if [[ "$got_sha" != "$want_sha" ]]; then
      echo "DRIFT [$corpus_root] $rel_path: sha256 $got_sha != pinned $want_sha" >&2
      failures=$((failures + 1))
    fi
    local got_bytes
    got_bytes="$(wc -c < "$file" | tr -d '[:space:]')"
    if [[ "$got_bytes" != "$want_bytes" ]]; then
      echo "DRIFT [$corpus_root] $rel_path: $got_bytes bytes != pinned $want_bytes" >&2
      failures=$((failures + 1))
    fi
  done < <(lock_entries "$lock")

  # A file on disk the lock does not pin is drift in the other direction
  while IFS= read -r unlisted; do
    echo "DRIFT [$corpus_root] $unlisted: on disk but not pinned in the lock" >&2
    failures=$((failures + 1))
  done < <(
    python3 - "$corpus_root" "$LOCK_NAME" <<'PY'
import json, os, sys
root, lock_name = sys.argv[1], sys.argv[2]
pinned = {e["path"] for e in json.load(open(os.path.join(root, lock_name)))["files"]}
for dirpath, _dirs, files in os.walk(root):
    for f in files:
        rel = os.path.relpath(os.path.join(dirpath, f), root)
        if f != lock_name and rel not in pinned:
            print(rel)
PY
  )

  local want_digest got_digest
  want_digest="$(python3 -c \
    "import json,sys; print(json.load(open(sys.argv[1]))['digest'])" "$lock")"
  got_digest="$(lock_body_bytes "$lock" | sha256_stdin | tr '[:upper:]' '[:lower:]')"
  if [[ "$got_digest" != "$want_digest" ]]; then
    echo "DRIFT [$corpus_root] manifest digest $got_digest != pinned $want_digest" >&2
    failures=$((failures + 1))
  fi

  if [[ "$failures" -gt 0 ]]; then
    echo "FAIL [$corpus_root] $failures drifted pin(s);" \
      "regenerate with verifier/freeze_corpus_lock.py" >&2
    return 1
  fi
  local count
  count="$(lock_entries "$lock" | wc -l | tr -d '[:space:]')"
  echo "ok [$corpus_root] $count file pins and the manifest digest re-derived via $SHA256_TOOL"
}

# Report both corpora before exiting, but any drift reddens the run
status=0
check_lock "$ROOT/conformance" || status=1
check_lock "$ROOT/verifier/conformance-vectors" || status=1
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi
echo "corpus lock: both corpora match the pins of their current rolling version"
