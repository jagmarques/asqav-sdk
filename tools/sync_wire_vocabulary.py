"""Verify pinned reference sources before checking or writing their fixed outputs."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "jagmarques/asqav-registry"
SOURCES = (
    "LICENSE", "NOTICE", "vocabulary/wire.json", "vocabulary/wire.schema.json",
    "vocabulary/validate.py", "tools/wire_vocabulary.py",
)
OUTPUTS = tuple("vocabulary/generated/" + name for name in ("wire.py", "wire.ts", "wire-cases.json"))
DIRECTORIES = {"vocabulary", "tools", "vocabulary/generated"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest member: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"invalid manifest constant: {value}")


def check_paths(vendor):
    for path in (vendor, *vendor.parents):
        if path.is_symlink():
            raise ValueError(f"symlink vendor ancestor: {path}")
    if not vendor.is_dir():
        raise ValueError("missing vendor directory")
    allowed = set(SOURCES) | set(OUTPUTS) | {"README.md", "upstream.json"}
    files = set()
    for path in vendor.rglob("*"):
        name = path.relative_to(vendor).as_posix()
        if path.is_symlink():
            raise ValueError(f"symlink vendor path: {name}")
        if path.is_dir():
            if name not in DIRECTORIES:
                raise ValueError(f"unexpected vendor directory: {name}")
        elif path.is_file() and name in allowed:
            files.add(name)
        else:
            raise ValueError(f"unexpected vendor path: {name}")
    missing = set(SOURCES) | {"README.md", "upstream.json"}
    if missing - files:
        raise ValueError("missing vendor files: " + ", ".join(sorted(missing - files)))


def read_manifest(vendor):
    manifest = json.loads(
        (vendor / "upstream.json").read_text(encoding="utf-8"),
        object_pairs_hook=unique_object, parse_constant=reject_constant,
    )
    if not isinstance(manifest, dict) or set(manifest) != {"repository", "commit", "files"}:
        raise ValueError("manifest must contain only repository, commit and files")
    if manifest["repository"] != REPOSITORY:
        raise ValueError("unexpected upstream repository")
    commit = manifest["commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("upstream commit must be a full lowercase hexadecimal object id")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(SOURCES):
        raise ValueError("manifest must pin exactly the six upstream source paths")
    for name, digest in files.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"invalid source digest: {name}")
    return manifest


def verified_snapshot(vendor):
    check_paths(vendor)
    manifest = read_manifest(vendor)
    contents = {}
    for name, digest in manifest["files"].items():
        raw = (vendor / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"pinned source mismatch: {name}")
        contents[name] = raw
    return manifest, contents


def verify_upstream(manifest, contents):
    for name in SOURCES:
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{manifest['commit']}/{name}"
        result = subprocess.run(
            ["curl", "--disable", "--fail", "--silent", "--show-error", "--proto", "=https",
             "--tlsv1.2", "--max-time", "20", "--write-out", "\n%{http_code}", url],
            capture_output=True, timeout=25, check=False,
        )
        body, _, status = result.stdout.rpartition(b"\n")
        if result.returncode != 0 or status != b"200":
            raise ValueError(f"upstream request failed: {name}")
        if body != contents[name]:
            raise ValueError(f"upstream source mismatch: {name}")
    print("OK: six immutable upstream sources match")


def run_reference(vendor, mode):
    manifest, contents = verified_snapshot(vendor)
    if mode == "--verify-upstream":
        verify_upstream(manifest, contents)
        return 0
    saved = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location(
            "asqav_wire_reference", vendor / "tools/wire_vocabulary.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main([mode])
    finally:
        sys.dont_write_bytecode = saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("--check", "--write", "--verify-upstream"):
        modes.add_argument(mode, action="store_const", const=mode, dest="mode")
    args = parser.parse_args(argv)
    try:
        return run_reference(ROOT / "vendor/wire-vocabulary", args.mode)
    except (OSError, UnicodeError, ValueError, ImportError, SyntaxError, subprocess.SubprocessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
