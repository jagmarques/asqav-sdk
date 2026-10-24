"""Build the sdist twice from clean git archives of HEAD and compare sha256."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
HATCHLING_SDIST = "import sys, hatchling.build as b; b.build_sdist(sys.argv[1])"


class BuilderUnavailable(RuntimeError):
    """Neither hatchling nor uv can build the sdist on this machine."""


class Result(NamedTuple):
    ok: bool
    digests: list[str]
    names: list[str]


def _git(root: Path, *args: str) -> bytes:
    run = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if run.returncode:
        raise RuntimeError(f"git {args[0]} exited {run.returncode}: {run.stderr.decode().strip()}")
    return run.stdout


def source_date_epoch(root: Path, rev: str = "HEAD") -> str:
    epoch = _git(root, "log", "-1", "--format=%ct", rev).decode().strip()
    if not epoch.isdigit():
        raise RuntimeError(f"no commit time for {rev}: {epoch!r}")
    return epoch


def export_tree(root: Path, dest: Path, rev: str = "HEAD") -> None:
    with tarfile.open(fileobj=io.BytesIO(_git(root, "archive", "--format=tar", rev))) as tree:
        if hasattr(tarfile, "data_filter"):
            tree.extractall(dest, filter="data")
        else:
            tree.extractall(dest)


def build_sdist(project: Path, out: Path, epoch: str) -> Path:
    if importlib.util.find_spec("hatchling") is not None:
        cmd = [sys.executable, "-c", HATCHLING_SDIST, str(out)]
    elif shutil.which("uv"):
        cmd = ["uv", "build", "--sdist", "--offline", "--out-dir", str(out), str(project)]
    else:
        raise BuilderUnavailable("neither hatchling nor uv is available to build the sdist")
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SOURCE_DATE_EPOCH": epoch}
    run = subprocess.run(cmd, cwd=project, env=env, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(f"sdist build exited {run.returncode}: {run.stderr.strip()[-600:]}")
    built = sorted(out.glob("*.tar.gz"))
    if len(built) != 1:
        raise RuntimeError(f"expected one sdist in {out}, found {[p.name for p in built]}")
    return built[0]


def check(
    root: Path = ROOT, *, rev: str = "HEAD", between: Callable[[Path], None] | None = None
) -> Result:
    """Two builds from fresh archives of rev; between(tree) edits the second."""
    epoch = source_date_epoch(root, rev)
    digests, names = [], []
    with tempfile.TemporaryDirectory(prefix="asqav-sdist-") as tmp:
        for n in (0, 1):
            tree = Path(tmp) / f"tree{n}"
            export_tree(root, tree, rev)
            if between is not None and n == 1:
                between(tree)
            sdist = build_sdist(tree / "python", Path(tmp) / f"out{n}", epoch)
            digests.append(hashlib.sha256(sdist.read_bytes()).hexdigest())
            names.append(sdist.name)
    return Result(digests[0] == digests[1], digests, names)


def main(
    argv: list[str] | None = None, *, between: Callable[[Path], None] | None = None
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--rev", default="HEAD")
    args = parser.parse_args(argv)
    try:
        result = check(args.root, rev=args.rev, between=between)
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, BuilderUnavailable) else 1
    if not result.ok:
        print(f"FAIL: sdist digests differ: {' '.join(result.digests)}", file=sys.stderr)
        return 1
    print(f"OK: {result.names[0]} sha256 {result.digests[0]} twice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
