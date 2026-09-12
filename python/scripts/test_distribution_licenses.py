"""Exercise license guards using actual built archives and altered copies."""

import contextlib
import io
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import check_distribution_licenses as checker


class DistributionLicenses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archives = [Path(value).resolve() for value in sys.argv[1:]]
        if len(cls.archives) != 2:
            raise ValueError("tests require an actual built wheel and source archive")
        cls.expected = checker.references(checker.ROOT)

    def changed_archive(self, original, change):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        members = checker.archive_members(original)
        change(members)
        path = Path(directory.name) / original.name
        if path.suffix == ".whl":
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in members.items():
                    archive.writestr(name, value)
        else:
            with tarfile.open(path, "w:gz") as archive:
                for name, value in members.items():
                    entry = tarfile.TarInfo(name)
                    entry.size = len(value)
                    archive.addfile(entry, io.BytesIO(value))
        return path

    def test_actual_built_pair(self):
        for path in self.archives:
            checker.check_archive(path, self.expected)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(checker.main([str(path) for path in self.archives]), 0)
        self.assertIn("OK:", output.getvalue())

    def test_missing_or_changed_license_bytes(self):
        for original in self.archives:
            names = checker.archive_members(original)
            for license_name in checker.LICENSE_FILES:
                suffix = (
                    "/licenses/" + license_name if original.suffix == ".whl" else "/" + license_name
                )
                member = next(name for name in names if name.endswith(suffix))
                for remove in (True, False):
                    with self.subTest(archive=original.name, member=member, remove=remove):

                        def change(members):
                            if remove:
                                members.pop(member)
                            else:
                                members[member] += b"altered terms"

                        broken = self.changed_archive(original, change)
                        with self.assertRaisesRegex(ValueError, "missing or changed license"):
                            checker.check_archive(broken, self.expected)

    def test_metadata_guards(self):
        for original in self.archives:
            members = checker.archive_members(original)
            name = next(key for key in members if key.endswith(("/METADATA", "/PKG-INFO")))
            raw = members[name]
            metadata_version = next(
                line for line in raw.splitlines(True) if line.startswith(b"Metadata-Version:")
            )
            version = next(line for line in raw.splitlines(True) if line.startswith(b"Version:"))
            cases = [
                (b"License-File: LICENSE\n", b"", "License-File"),
                (b"License-File: LICENSE\n", b"License-File: LICENSE\n" * 2, "License-File"),
                (b"License-File: LICENSE\n", b"License-File: ../LICENSE\n", "License-File"),
                (
                    checker.EXPRESSION.encode(),
                    b"Apache-2.0 OR LicenseRef-Elastic-License-2.0",
                    "expression",
                ),
                (metadata_version, b"Metadata-Version: 2.3\n", "2.4 or newer"),
                (metadata_version, b"Metadata-Version: wrong\n", "invalid metadata version"),
                (metadata_version, metadata_version * 2, "duplicate metadata version"),
                (b"Name: asqav\n", b"Name: another-package\n", "package name"),
                (version, version * 2, "duplicate package version"),
            ]
            for before, after, error in cases:
                with self.subTest(archive=original.name, error=error, after=after):
                    self.assertIn(before, raw)
                    broken = self.changed_archive(
                        original, lambda data: data.update({name: raw.replace(before, after, 1)})
                    )
                    with self.assertRaisesRegex(ValueError, error):
                        checker.check_archive(broken, self.expected)

    def test_missing_metadata_and_wrong_path(self):
        for original in self.archives:
            members = checker.archive_members(original)
            name = next(key for key in members if key.endswith(("/METADATA", "/PKG-INFO")))
            broken = self.changed_archive(original, lambda data: data.pop(name))
            with self.assertRaisesRegex(ValueError, "exactly one package metadata"):
                checker.check_archive(broken, self.expected)
            renamed = self.changed_archive(
                original, lambda data: data.update({"wrong/" + name: data.pop(name)})
            )
            with self.assertRaisesRegex(ValueError, "metadata archive path mismatch"):
                checker.check_archive(renamed, self.expected)

    def test_standalone_verifier_bytes_and_header(self):
        for original in self.archives:
            name = next(
                key for key in checker.archive_members(original) if key.endswith(checker.VERIFIER)
            )
            broken = self.changed_archive(
                original, lambda data: data.update({name: b"# missing original header\n"})
            )
            with self.assertRaisesRegex(ValueError, "verifier or header changed"):
                checker.check_archive(broken, self.expected)

    def test_duplicate_archive_member(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "duplicate.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for _ in range(2):
                entry = tarfile.TarInfo("duplicate")
                entry.size = 1
                archive.addfile(entry, io.BytesIO(b"x"))
        with self.assertRaisesRegex(ValueError, "duplicate archive member"):
            checker.archive_members(path)

    def test_source_terms_cannot_drift(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for name in checker.LICENSE_FILES:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.expected[name])
        verifier = root / "src" / checker.VERIFIER
        verifier.parent.mkdir(parents=True)
        verifier.write_bytes(self.expected[checker.VERIFIER])
        for name in checker.TERMS:
            with self.subTest(name=name):
                (root / name).write_bytes(b"not the existing terms")
                with self.assertRaisesRegex(ValueError, "source terms drift"):
                    checker.references(root)
                (root / name).write_bytes(self.expected[name])

    def test_cli_refuses_incomplete_or_invalid_inputs(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(checker.main([str(self.archives[1]), str(self.archives[0])]), 1)
            self.assertEqual(checker.main(["missing.whl", str(self.archives[1])]), 1)
        self.assertIn("FAIL:", output.getvalue())
        with self.assertRaisesRegex(ValueError, "expected a wheel"):
            checker.archive_members(Path("unsupported.zip"))

    def test_damaged_archives_report_concise_cli_errors(self):
        wheel, sdist = self.archives
        wheel_bytes, source_bytes = wheel.read_bytes(), sdist.read_bytes()
        damaged = {
            "truncated.tar.gz": source_bytes[: len(source_bytes) // 2],
            "deflate.tar.gz": source_bytes[:10] + b"\x07" + source_bytes[11:],
            "truncated.whl": wheel_bytes[: len(wheel_bytes) // 2],
        }
        with zipfile.ZipFile(wheel) as archive:
            first = archive.infolist()[0]
            offset = first.header_offset + 30 + len(first.filename.encode()) + len(first.extra)
            central = wheel_bytes.index(b"PK\x01\x02")
            for name, local_field, central_field, value in (
                ("unsupported", 8, 10, 99),
                ("encrypted", 6, 8, first.flag_bits | 1),
            ):
                raw = bytearray(wheel_bytes)
                struct.pack_into("<H", raw, first.header_offset + local_field, value)
                struct.pack_into("<H", raw, central + central_field, value)
                damaged[name + ".whl"] = bytes(raw)
            raw = bytearray(wheel_bytes)
            raw[offset] = 7
            damaged["deflate.whl"] = bytes(raw)
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_LZMA) as target:
                for item in archive.infolist():
                    target.writestr(item.filename, archive.read(item))
        raw = bytearray(stream.getvalue())
        name_size, extra_size = struct.unpack_from("<HH", raw, 26)
        raw[30 + name_size + extra_size + 4] = 255
        damaged["lzma.whl"] = bytes(raw)
        with tempfile.TemporaryDirectory() as directory:
            for name, data in damaged.items():
                with self.subTest(name=name):
                    path = Path(directory) / name
                    path.write_bytes(data)
                    arguments = [wheel, path] if name.endswith(".gz") else [path, sdist]
                    result = subprocess.run(
                        [sys.executable, "-B", str(Path(checker.__file__)), *map(str, arguments)],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, "")
                    self.assertTrue(result.stderr.startswith("FAIL: "), result.stderr)
                    self.assertEqual(len(result.stderr.splitlines()), 1, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
