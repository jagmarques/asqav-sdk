"""CODEMAP.md lists each tracked source file with purpose, symbols and tests.
CI fails when it is stale. NEGCTL cases run the generator on a tmp git repo."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("gen_codemap", ROOT / "tools" / "gen_codemap.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _repo(root, files):
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "add", "--", *files], cwd=root, check=True)
    return root


class CodemapTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def line_for(self, name):
        return next(x for x in gen.render(self.root).splitlines() if f"`{name}`" in x)

    def test_python_entry_has_purpose_symbols_and_named_test(self):
        _repo(self.root, {
            "python/src/asqav/billing.py": '"""Invoice totals."""\n\ndef total():\n    pass\n\n'
                                           "def _private():\n    pass\n\nclass Ledger:\n    pass\n",
            "python/tests/test_billing.py": "def test_total():\n    pass\n",
        })
        self.assertEqual(self.line_for("billing.py"),
                         "- `billing.py` - Invoice totals | symbols: total Ledger"
                         " | tests: python/tests/test_billing.py")

    def test_typescript_entry_reads_first_comment_and_exports(self):
        _repo(self.root, {
            "typescript/src/budget.ts": "// Spend caps per agent\nexport function capFor() {}\n"
                                        "export class Budget {}\n",
            "typescript/tests/budget.test.ts": "",
        })
        self.assertEqual(self.line_for("budget.ts"),
                         "- `budget.ts` - Spend caps per agent | symbols: capFor Budget"
                         " | tests: typescript/tests/budget.test.ts")

    def test_file_added_after_the_map_fails_check(self):
        """NEGCTL: a source file added after the map fails --check."""
        _repo(self.root, {"python/src/asqav/a.py": '"""A."""\n'})
        (self.root / gen.OUT).write_text(gen.render(self.root))
        self.assertEqual(gen.check(self.root), 0)
        _repo(self.root, {"python/src/asqav/b.py": '"""B."""\n'})
        self.assertEqual(gen.check(self.root), 1)

    def test_unparsable_python_names_the_file(self):
        _repo(self.root, {"python/src/asqav/bad.py": "def (:\n"})
        with self.assertRaisesRegex(SystemExit, "python/src/asqav/bad.py does not parse"):
            gen.render(self.root)

    def test_missing_map_fails_check(self):
        _repo(self.root, {"python/src/asqav/a.py": '"""A."""\n'})
        self.assertEqual(gen.check(self.root), 1)

    def test_committed_map_is_fresh(self):
        self.assertEqual(gen.check(ROOT), 0)

    def test_script_symbols_exclude_nonexports_and_nested_declarations(self):
        """NEGCTL: private and nested declarations must stay out of symbols."""
        source = ("export function publicFn() {}\nexport class PublicClass {}\n"
                  "export const publicArrow = () => {};\n"
                  "export default async function defaultFn() {}\n"
                  "function privateFn() {}\nconst privateArrow = () => {};\n"
                  "  const inner = () => {};\n  function nestedFn() {}\n"
                  "  export function nestedExport() {}\n")
        _repo(self.root, {f"typescript/src/sample{ext}": source
                          for ext in (".ts", ".js", ".mjs")})
        for ext in (".ts", ".js", ".mjs"):
            with self.subTest(ext=ext):
                self.assertEqual(self.line_for("sample" + ext),
                                 f"- `sample{ext}` | symbols: publicFn PublicClass"
                                 " publicArrow defaultFn")

    def test_private_modules_link_tests_after_underscore_normalization(self):
        modules = {
            "_jcs.py": "test_jcs.py",
            "_mode.py": "test_mode_resolver.py",
            "_schema.py": "test_schema_validation.py",
            "extras/_base.py": "test_base_adapter.py",
        }
        files = {"python/src/asqav/" + module: '"""Private module."""\n'
                 for module in modules}
        files.update({"python/tests/" + test: "" for test in modules.values()})
        files["python/tests/test_microscope.py"] = ""
        _repo(self.root, files)
        for module, test in modules.items():
            with self.subTest(module=module):
                self.assertEqual(self.line_for(Path(module).name),
                                 f"- `{Path(module).name}` - Private module"
                                 f" | tests: python/tests/{test}")
        self.assertEqual(gen.tests_for("python/src/asqav/_scope.py", [
            "python/tests/test_microscope.py", "python/tests/test_scopes.py",
            "python/tests/test_scope.py", "python/tests/test_scope_tokens.py",
        ]), ["python/tests/test_scope.py", "python/tests/test_scope_tokens.py"])

    def test_private_generic_and_short_stems_do_not_link_tests(self):
        """NEGCTL: normalize before generic-name and minimum-length checks."""
        for stem in ("__init__", "__main__", "_main", "_utils", "_common",
                     "_types", "_index", "_conftest", "_io", "___"):
            with self.subTest(stem=stem):
                name = stem.lstrip("_")
                self.assertEqual(gen.tests_for(f"python/src/asqav/{stem}.py", [
                    f"python/tests/test_{name}.py",
                    f"python/tests/test_{name}_extra.py",
                ]), [])

    def test_named_tests_exclude_substring_matches(self):
        """NEGCTL: unrelated scope names must not acquire a test link."""
        unrelated = ["tests/test_offline_api_declares_anchor_scope.py",
                     "tests/test_scopes.py", "tests/test_microscope.py"]
        named = ["tests/scope.py", "tests/test_scope.py",
                 "tests/test_scope_validation.py", "tests/scope_extra.py",
                 "typescript/tests/scope.test.ts", "typescript/tests/scope.spec.ts"]
        for test in named:
            with self.subTest(test=test):
                self.assertEqual(gen.tests_for("python/src/asqav/scope.py",
                                               unrelated + [test]), [test])
        self.assertEqual(gen.tests_for("python/src/asqav/scope.py", unrelated), [])


if __name__ == "__main__":
    unittest.main()
