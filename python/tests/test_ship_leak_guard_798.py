"""Shipped SDKs leak no server internals (798 D3).

The SDKs are open source; this gate proves they carry no source maps, private
hostnames or live secrets outside documented examples, scanning python/src,
typescript/src and built dist/ when present.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
assert (ROOT / "typescript" / "package.json").exists(), f"repo root misresolved: {ROOT}"
PY_SRC = ROOT / "python" / "src" / "asqav"
TS_SRC = ROOT / "typescript" / "src"
TS_DIST = ROOT / "typescript" / "dist"

PRIVATE_HOST = re.compile(
    r"hetzner|coolify|staging|\.internal\b|0\.0\.0\.0"
    r"|10\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}"
    r"|169\.254\.\d{1,3}\.\d{1,3}|://[A-Za-z0-9.-]*\.(local|lan)\b",
    re.IGNORECASE,
)
SOURCE_MAP = re.compile(r"sourceMappingURL|\.map\.js", re.IGNORECASE)
LIVE_SECRET = re.compile(
    r"ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[bap]-"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY"
)
EXAMPLE_HINT = re.compile(
    r"example|mock|test|placeholder|your_|xxx|\.\.\.|docstring", re.IGNORECASE
)
SK_LIVE = re.compile(r"sk_live_[A-Za-z0-9]{16,}")


def _findings(text: str) -> list[str]:
    out: list[str] = []
    if SOURCE_MAP.search(text):
        out.append("source-map reference")
    if PRIVATE_HOST.search(text):
        out.append("private hostname")
    if LIVE_SECRET.search(text):
        out.append("live secret shape")
    for line in text.splitlines():
        live = [
            token
            for token in SK_LIVE.findall(line)
            if not EXAMPLE_HINT.search(token) and "..." not in line
        ]
        if live:
            out.append(f"non-example secret: {line.strip()[:60]}")
            break
    return out


def _scan_tree(root: Path, *exts: str) -> list[str]:
    hits: list[str] = []
    for ext in exts:
        for path in sorted(root.rglob(f"*{ext}")):
            if ".venv" in path.parts or "node_modules" in path.parts:
                continue
            for finding in _findings(path.read_text(encoding="utf-8", errors="replace")):
                hits.append(f"{path.relative_to(ROOT)}: {finding}")
    return hits


def test_shipped_sources_carry_no_server_internals() -> None:
    hits = _scan_tree(PY_SRC, ".py") + _scan_tree(TS_SRC, ".ts")
    assert hits == [], hits[:10]


def test_built_dist_has_no_source_maps() -> None:
    if not TS_DIST.is_dir():
        pytest.skip("no local dist build; publish builds it")
    maps = [p for p in TS_DIST.rglob("*.map")]
    assert maps == [], [str(p) for p in maps[:5]]
    # CI builds dist before pytest, so this scans the real shipped bytes.
    hits = _scan_tree(TS_DIST, ".js", ".cjs", ".mjs", ".d.ts", ".d.mts")
    assert hits == [], hits[:10]


def _planted(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_planted_source_map_fails(tmp_path: Path) -> None:
    planted = _planted(tmp_path, "a.js", "x=1;\n//# sourceMappingURL=a.js.map\n")
    assert _findings(planted.read_text())


def test_planted_private_hostname_fails(tmp_path: Path) -> None:
    planted = _planted(tmp_path, "b.py", 'HOST = "https://api-coolify-01.internal"\n')
    assert _findings(planted.read_text())


def test_planted_local_and_lan_hostnames_fail(tmp_path: Path) -> None:
    body = 'A = "http://asqav-hub.local:8000"\nB = "http://nas.lan:5000"\n'
    assert _findings(_planted(tmp_path, "b2.py", body).read_text())


def test_planted_live_token_fails(tmp_path: Path) -> None:
    planted = _planted(tmp_path, "c.py", 'token = "ghp_abcdefghijklmnopqrstuvwx"\n')
    assert _findings(planted.read_text())


def test_planted_sk_live_token_fails(tmp_path: Path) -> None:
    tok = "sk_live_" + "9f8K2mZx" + "QvBn4Lp7"
    assert _findings(_planted(tmp_path, "c2.py", f'token = "{tok}"\n').read_text())


def test_documented_example_secret_passes(tmp_path: Path) -> None:
    # Built from parts so no token-shaped literal lands in the tree; the hint
    # sits inside the token, proving detector and allowlist both engage.
    tok = "sk_live_" + "example" + "0" * 10
    body = f'api_key = "{tok}"  # example from the docs\n'
    assert SK_LIVE.search(body), "example must reach the detector"
    assert _findings(_planted(tmp_path, "d.py", body).read_text()) == []
