"""Check license contents and metadata in a built wheel and source archive."""

import argparse
import hashlib
import lzma
import os
import re
import sys
import tarfile
import zipfile
import zlib
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import BinaryIO, cast

ROOT = Path(__file__).resolve().parents[1]
EXPRESSION = "LicenseRef-Elastic-License-2.0 AND Apache-2.0"
TERMS = {
    "LICENSE": "ad5d7597acff2920f256e16d0a47b0547e67201b72f67a1813b1ce253e36a58c",
    "licenses/Apache-2.0.txt": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}
LICENSE_FILES = ["LICENSE", "NOTICE", "licenses/Apache-2.0.txt"]
VERIFIER = "asqav/verifier/verify_receipt.py"
CORPUS_DIR = "asqav/conformance_vectors"
CORPUS_MIN_EXPECTED = 118
#: 118 shipped expected.json; dsse-attestation-ml-dsa-65 sits outside
#: manifest.json, so the manifest census stays 117.

#: The only repo paths granted Apache-2.0; widening the grant is irrevocable.
APACHE_SPDX_FILES = ("python/src/asqav/verifier/verify_receipt.py",)
SPDX_SCAN_DIRS = ("python/src", "typescript/src")
SPDX_MARKER = re.compile(rb"spdx-license-identifier", re.IGNORECASE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def references(root: Path) -> dict[str, bytes]:
    files = {name: (root / name).read_bytes() for name in LICENSE_FILES}
    for name, expected in TERMS.items():
        require(hashlib.sha256(files[name]).hexdigest() == expected, f"source terms drift: {name}")
    files[VERIFIER] = (root / "src" / VERIFIER).read_bytes()
    return files


def archive_members(path: Path) -> dict[str, bytes]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as wheel_archive:
            pairs = [
                (name, wheel_archive.read(name))
                for name in wheel_archive.namelist()
                if not name.endswith("/")
            ]
    else:
        require(path.name.endswith(".tar.gz"), "expected a wheel or .tar.gz source archive")
        with tarfile.open(path, "r:gz") as source_archive:
            pairs = [
                (entry.name, cast(BinaryIO, source_archive.extractfile(entry)).read())
                for entry in source_archive.getmembers()
                if entry.isfile()
            ]
    require(len(pairs) == len({name for name, _ in pairs}), "duplicate archive member")
    return dict(pairs)


def check_metadata(raw: bytes) -> str:
    metadata = BytesParser(policy=policy.default).parsebytes(raw)
    require(metadata.get_all("Name") == ["asqav"], "unexpected package name")
    versions = metadata.get_all("Version", [])
    require(len(versions) == 1 and bool(versions[0]), "missing or duplicate package version")
    formats = metadata.get_all("Metadata-Version", [])
    require(len(formats) == 1, "missing or duplicate metadata version")
    format_version = str(formats[0]).split(".")
    require(
        len(format_version) == 2 and all(part.isdigit() for part in format_version),
        "invalid metadata version",
    )
    require(
        tuple(map(int, format_version)) >= (2, 4), "license metadata requires version 2.4 or newer"
    )
    require(metadata.get_all("License-Expression") == [EXPRESSION], "license expression mismatch")
    paths = metadata.get_all("License-File", [])
    require(sorted(paths) == sorted(LICENSE_FILES), "missing, duplicate or unexpected License-File")
    return str(versions[0])


def check_archive(path: Path, expected: dict[str, bytes]) -> None:
    members = archive_members(path)
    wheel = path.suffix == ".whl"
    suffix = "/METADATA" if wheel else "/PKG-INFO"
    metadata_paths = [name for name in members if name.endswith(suffix)]
    require(len(metadata_paths) == 1, "expected exactly one package metadata file")
    metadata_path = metadata_paths[0]
    version = check_metadata(members[metadata_path])
    root = f"asqav-{version}.dist-info" if wheel else f"asqav-{version}"
    require(metadata_path == root + suffix, "metadata archive path mismatch")
    license_root = root + "/licenses/" if wheel else root + "/"
    for name in LICENSE_FILES:
        require(
            members.get(license_root + name) == expected[name],
            f"missing or changed license: {name}",
        )
    verifier_path = VERIFIER if wheel else root + "/src/" + VERIFIER
    require(
        members.get(verifier_path) == expected[VERIFIER], "standalone verifier or header changed"
    )
    check_corpus(members)


def check_corpus(members: dict[str, bytes]) -> None:
    """The archive must carry the conformance corpus it verifies against."""
    expected = [name for name in members if name.endswith("expected.json")]
    require(len(expected) >= CORPUS_MIN_EXPECTED, f"corpus too small: {len(expected)}")
    papers = {name: (ROOT.parent / "conformance" / name).read_bytes()
              for name in ("NOTICE", "LICENSE")}
    for name, want in papers.items():
        matches = [m for m in members if m.endswith(f"{CORPUS_DIR}/{name}")]
        require(len(matches) == 1, f"missing or changed license: {CORPUS_DIR}/{name}")
        require(members[matches[0]] == want, f"missing or changed license: {CORPUS_DIR}/{name}")


def find_spdx_files(root: Path) -> list[str]:
    """Repo paths under SPDX_SCAN_DIRS whose bytes carry an SPDX marker."""
    hits = []
    for scan in SPDX_SCAN_DIRS:
        require((root / scan).is_dir(), f"no {scan} directory under {root}")
        for top, dirs, names in os.walk(root / scan):
            dirs.sort()
            for name in sorted(names):
                path = Path(top) / name
                if path.is_file() and SPDX_MARKER.search(path.read_bytes()):
                    hits.append(path.relative_to(root).as_posix())
    return hits


def check_spdx_scope(root: Path) -> None:
    found = find_spdx_files(root)
    stray = [path for path in found if path not in APACHE_SPDX_FILES]
    require(not stray, f"SPDX header outside APACHE_SPDX_FILES: {', '.join(stray)}")
    missing = [path for path in APACHE_SPDX_FILES if path not in found]
    require(not missing, f"APACHE_SPDX_FILES entry lost its header: {', '.join(missing)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, nargs="?")
    parser.add_argument("sdist", type=Path, nargs="?")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    args = parser.parse_args(argv)
    archives = args.wheel is not None or args.sdist is not None
    try:
        check_spdx_scope(args.root)
        if archives:
            require(
                args.wheel is not None
                and args.sdist is not None
                and args.wheel.suffix == ".whl"
                and args.sdist.name.endswith(".tar.gz"),
                "supply one wheel followed by one .tar.gz source archive",
            )
            expected = references(ROOT)
            check_archive(args.wheel, expected)
            check_archive(args.sdist, expected)
    except (
        OSError,
        ValueError,
        EOFError,
        RuntimeError,
        tarfile.TarError,
        zipfile.BadZipFile,
        zlib.error,
        lzma.LZMAError,
    ) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"OK: SPDX headers only in {', '.join(APACHE_SPDX_FILES)}")
    if archives:
        print("OK: wheel and source archive license files, metadata and verifier bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
