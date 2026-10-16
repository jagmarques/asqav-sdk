"""Registry drift: a tagged SDK release must be published and latest on its registry."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("registry", ROOT / "tools/check_registry_versions.py")
registry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registry)
V = registry.parse


def state(latest, *versions):
    return {"latest": latest, "versions": list(versions) or [latest]}


class DriftTests(unittest.TestCase):
    def test_in_sync(self):
        tags = {"python": V("0.10.13"), "typescript": V("0.10.13")}
        reg = {"python": state("0.10.13", "0.10.12", "0.10.13"), "typescript": state("0.10.13")}
        self.assertEqual(registry.drift(tags, reg, tags), [])

    def test_a_tag_missing_from_npm_is_caught(self):
        # Two TypeScript tags past the version npm served as latest
        tags = {"python": V("0.10.12"), "typescript": V("0.10.12")}
        reg = {"python": state("0.10.12"), "typescript": state("0.10.10", "0.10.9", "0.10.10")}
        self.assertEqual(registry.drift(tags, reg, tags),
                         ["typescript: tag 0.10.12 is not on npm (latest 0.10.10)"])

    def test_control_numeric_order_not_string_order(self):
        # A string compare ranks "0.10.9" above "0.10.10" and would pass this lag
        self.assertGreater("0.10.9", "0.10.10")
        tags = {"python": V("0.10.10"), "typescript": None}
        reg = {"python": state("0.10.9", "0.10.9", "0.10.10"), "typescript": state("0.10.10")}
        self.assertEqual(registry.drift(tags, reg, {}),
                         ["python: PyPI latest 0.10.9 is behind tag 0.10.10"])

    def test_latest_moved_back_is_caught(self):
        tags = {"python": None, "typescript": V("0.10.13")}
        reg = {"python": state("0.10.13"), "typescript": state("0.10.12", "0.10.12", "0.10.13")}
        self.assertEqual(registry.drift(tags, reg, {}),
                         ["typescript: npm latest 0.10.12 is behind tag 0.10.13"])

    def test_repo_behind_registry_is_caught(self):
        tags = {"python": V("0.10.13"), "typescript": V("0.10.13")}
        reg = {"python": state("0.10.13"), "typescript": state("0.10.13")}
        repo = {"python": V("0.10.12"), "typescript": V("0.10.13")}
        self.assertEqual(registry.drift(tags, reg, repo),
                         ["python: repo version 0.10.12 is behind PyPI 0.10.13"])

    def test_a_bumped_but_untagged_repo_is_not_drift(self):
        tags = {"python": V("0.10.12"), "typescript": V("0.10.12")}
        reg = {"python": state("0.10.12"), "typescript": state("0.10.12")}
        repo = {"python": V("0.10.13"), "typescript": V("0.10.13")}
        self.assertEqual(registry.drift(tags, reg, repo), [])


class InputTests(unittest.TestCase):
    def test_newest_tags_are_numeric_per_language(self):
        tags = ["py-v0.10.9", "py-v0.10.10", "ts-v0.10.2", "archive/pr-528", "ts-v0.9.30"]
        self.assertEqual(registry.newest_tags(tags),
                         {"python": (0, 10, 10), "typescript": (0, 10, 2)})

    def test_non_release_strings_do_not_parse(self):
        for text in ("", None, "0.10", "0.10.12rc1", "v0.10.12", "0.10.12 "):
            self.assertIsNone(V(text), text)

    def test_both_halves_of_the_repo_carry_one_version(self):
        versions = registry.repo_versions()
        self.assertIsNotNone(versions["python"])
        self.assertEqual(versions["python"], versions["typescript"])


if __name__ == "__main__":
    unittest.main()
