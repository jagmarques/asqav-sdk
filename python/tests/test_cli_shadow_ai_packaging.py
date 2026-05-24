"""Packaging smoke test: the shadow-ai template set ships in the wheel.

`asqav shadow-ai init` resolves its assets via `importlib.resources` and
copies three files into the target directory: `docker-compose.yml`,
`.env.template`, and `README.md`. Hatchling's default file selector
excludes dotfiles, so `.env.template` is easy to drop on a repackaging
sweep. This test asserts all three resources are importable from the
installed package, catching the regression at test time rather than at
`asqav shadow-ai init` time.
"""

from __future__ import annotations

import os
import sys
from importlib.resources import files

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.cli import _SHADOW_AI_TEMPLATE_FILES


def test_all_shadow_ai_template_files_are_packaged() -> None:
    """Every name in `_SHADOW_AI_TEMPLATE_FILES` resolves from the package."""
    template_dir = files("asqav").joinpath("templates", "shadow-ai")
    for name in _SHADOW_AI_TEMPLATE_FILES:
        resource = template_dir.joinpath(name)
        assert resource.is_file(), (
            f"Template asset `{name}` is not present in the installed package. "
            "Check `python/pyproject.toml` include / force-include rules."
        )


def test_env_template_carries_required_keys() -> None:
    """The packaged `.env.template` carries the keys the CLI validator checks."""
    template_dir = files("asqav").joinpath("templates", "shadow-ai")
    body = template_dir.joinpath(".env.template").read_text(encoding="utf-8")
    for key in ("ASQAV_API_KEY", "ASQAV_AGENT_ID", "POSTGRES_PASSWORD"):
        assert key in body, f"`.env.template` is missing the `{key}` key."
