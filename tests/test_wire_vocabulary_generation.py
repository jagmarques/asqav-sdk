"""Exercise pinned imports, fixed outputs and refused corruption in isolated copies."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sync_wire", ROOT / "tools/sync_wire_vocabulary.py")
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)
EXPECTED = {
    "wire.py": "debe3adc12910d1108143cd8422714d4b05969757768c4fa83ab4c815a253a84",
    "wire.ts": "09cee48e8af98e4310c2440a04c5ac06234dcd86e133713812a0baccc0b34474",
    "wire-cases.json": "54ae6b61acc6b260b93cd331bff97395c7307423f3dd1b9dec3e5d85e6eabf9c",
}


class WireImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="wire-import-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.vendor = self.root / "vendor/wire-vocabulary"
        self.reset()

    def reset(self):
        if self.vendor.exists():
            shutil.rmtree(self.vendor)
        shutil.copytree(ROOT / "vendor/wire-vocabulary", self.vendor)

    def run_mode(self, mode):
        output, error = io.StringIO(), io.StringIO()
        with patch.object(sync, "ROOT", self.root), redirect_stdout(output), redirect_stderr(error):
            code = sync.main([mode])
        return code, output.getvalue(), error.getvalue()

    def write_manifest(self, value):
        (self.vendor / "upstream.json").write_text(json.dumps(value), encoding="utf-8")

    def test_exact_outputs_and_repeat(self):
        generated = self.vendor / "vocabulary/generated"
        if generated.exists():
            shutil.rmtree(generated)
        self.assertEqual(self.run_mode("--check")[0], 1)
        self.assertEqual(self.run_mode("--write")[0], 0)
        actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in generated.iterdir()}
        self.assertEqual(actual, EXPECTED)
        self.assertEqual(self.run_mode("--write")[0], 0)
        self.assertEqual(self.run_mode("--check")[0], 0)
        self.assertEqual(actual, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in generated.iterdir()})
        self.assertFalse(list(self.vendor.rglob("__pycache__")))

    def test_each_output_drift_is_named_without_writes(self):
        self.assertEqual(self.run_mode("--write")[0], 0)
        for name in EXPECTED:
            path = self.vendor / "vocabulary/generated" / name
            original = path.read_bytes()
            for value in (None, b"corrupt"):
                with self.subTest(name=name, value=value):
                    if value is None:
                        path.unlink()
                    else:
                        path.write_bytes(value)
                    code, _, error = self.run_mode("--check")
                    self.assertEqual(code, 1)
                    self.assertIn("vocabulary/generated/" + name, error)
                    self.assertEqual(path.read_bytes() if path.exists() else None, value)
                    path.write_bytes(original)
        self.assertEqual(self.run_mode("--check")[0], 0)

    def test_manifest_shapes_are_refused(self):
        original = json.loads((self.vendor / "upstream.json").read_text())
        for field, value in [("repository", "other/repo"), ("commit", "main"), ("commit", 12),
                             ("files", []), ("url", "https://example.invalid"), ("commit", float("nan"))]:
            with self.subTest(field=field, value=value):
                self.write_manifest({**original, field: value})
                self.assertEqual(self.run_mode("--check")[0], 1)
        for value in ([], {"repository": original["repository"]}):
            self.write_manifest(value)
            self.assertEqual(self.run_mode("--check")[0], 1)
        for files in ({k: v for k, v in original["files"].items() if k != "NOTICE"},
                      {**original["files"], "extra.py": "0" * 64},
                      {**original["files"], "NOTICE": 9},
                      {**original["files"], "NOTICE": "bad"}):
            self.write_manifest({**original, "files": files})
            self.assertEqual(self.run_mode("--check")[0], 1)
        raw = json.dumps(original)
        (self.vendor / "upstream.json").write_text('{"commit":"main",' + raw[1:])
        self.assertIn("duplicate manifest member", self.run_mode("--check")[2])

    def test_paths_and_cached_code_are_refused(self):
        for case in ("missing", "extra", "cache", "output-parent", "output-file", "symlink", "source-link"):
            with self.subTest(case=case):
                self.reset()
                generated = self.vendor / "vocabulary/generated"
                if generated.exists():
                    shutil.rmtree(generated)
                if case == "missing":
                    (self.vendor / "NOTICE").unlink()
                elif case == "extra":
                    (self.vendor / "extra.json").write_text("{}")
                elif case == "cache":
                    (self.vendor / "vocabulary/__pycache__").mkdir()
                elif case == "output-parent":
                    generated.write_text("not a directory")
                elif case == "output-file":
                    (generated / "wire.py").mkdir(parents=True)
                elif case == "symlink":
                    generated.symlink_to(self.root, target_is_directory=True)
                else:
                    (self.vendor / "NOTICE").unlink()
                    (self.vendor / "NOTICE").symlink_to(ROOT / "vendor/wire-vocabulary/NOTICE")
                self.assertEqual(self.run_mode("--write")[0], 1)
        shutil.rmtree(self.vendor)
        self.assertIn("missing vendor directory", self.run_mode("--check")[2])
        self.vendor.symlink_to(ROOT / "vendor/wire-vocabulary", target_is_directory=True)
        self.assertIn("symlink vendor ancestor", self.run_mode("--check")[2])
        self.vendor.unlink()

    def test_hash_precedes_loading_and_network(self):
        for name in sync.SOURCES:
            for mode in ("--check", "--write", "--verify-upstream"):
                with self.subTest(name=name, mode=mode):
                    self.reset()
                    marker = self.root / "executed"
                    path = self.vendor / name
                    suffix = f"\n__import__('pathlib').Path({str(marker)!r}).write_text('executed')\n"
                    path.write_bytes(path.read_bytes() + suffix.encode())
                    with patch.object(sync.subprocess, "run", side_effect=AssertionError("unexpected HTTP")):
                        code, _, error = self.run_mode(mode)
                    self.assertEqual(code, 1)
                    self.assertIn("pinned source mismatch: " + name, error)
                    self.assertFalse(marker.exists())

    def test_upstream_transport_is_fixed_and_read_only(self):
        manifest, contents = sync.verified_snapshot(self.vendor)
        calls = []
        def transport(argv, **kwargs):
            calls.append(argv)
            self.assertEqual(argv[:2], ["curl", "--disable"])
            self.assertNotIn("--location", argv)
            self.assertNotIn("--insecure", argv)
            self.assertEqual(kwargs["timeout"], 25)
            name = argv[-1].removeprefix(f"https://raw.githubusercontent.com/{sync.REPOSITORY}/{manifest['commit']}/")
            self.assertIn(name, sync.SOURCES)
            return subprocess.CompletedProcess(argv, 0, contents[name] + b"\n200", b"")
        with patch.object(sync.subprocess, "run", side_effect=transport):
            self.assertEqual(self.run_mode("--verify-upstream")[0], 0)
        self.assertEqual(len(calls), 6)
        for result in (subprocess.CompletedProcess([], 0, b"changed\n200", b""),
                       subprocess.CompletedProcess([], 0, contents["LICENSE"] + b"\n302", b""),
                       subprocess.CompletedProcess([], 22, b"\n404", b"")):
            with patch.object(sync.subprocess, "run", return_value=result) as transport_call:
                self.assertEqual(self.run_mode("--verify-upstream")[0], 1)
                self.assertEqual(transport_call.call_count, 1)
        with patch.object(sync.subprocess, "run", side_effect=subprocess.TimeoutExpired("curl", 25)):
            self.assertEqual(self.run_mode("--verify-upstream")[0], 1)

    def test_real_cli_modes_and_missing_dependency(self):
        tools = self.root / "tools"
        tools.mkdir()
        script = tools / "sync_wire_vocabulary.py"
        shutil.copy2(ROOT / "tools/sync_wire_vocabulary.py", script)
        for args, expected in [(["--write"], 0), (["--check"], 0), ([], 2),
                               (["--write", "--check"], 2), (["--url", "https://example.invalid"], 2)]:
            result = subprocess.run([sys.executable, "-B", str(script), *args], cwd=self.root.parent, capture_output=True)
            self.assertEqual(result.returncode, expected, result.stderr)
        result = subprocess.run([sys.executable, "-B", "-S", str(script), "--check"], capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"jsonschema is required", result.stderr)
        (self.vendor / "NOTICE").write_text("changed")
        result = subprocess.run([sys.executable, "-B", "-O", str(script), "--write"], capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"pinned source mismatch", result.stderr)

    def test_bytecode_flag_restored_after_loader_failure(self):
        saved = sys.dont_write_bytecode
        with patch.object(sync.importlib.util, "module_from_spec", side_effect=ImportError("fixture")):
            self.assertEqual(self.run_mode("--check")[0], 1)
        self.assertEqual(sys.dont_write_bytecode, saved)


if __name__ == "__main__":
    unittest.main()
