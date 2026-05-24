"""Tests for the `asqav shadow-ai` CLI subcommand group."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typer.testing import CliRunner

from asqav.cli import app

runner = CliRunner()


def test_init_creates_files_in_target_dir(tmp_path: Path) -> None:
    """`shadow-ai init --dir X` writes the three template files into X."""
    target = tmp_path / "stack"
    result = runner.invoke(app, ["shadow-ai", "init", "--dir", str(target)])
    assert result.exit_code == 0, result.output
    assert (target / "docker-compose.yml").exists()
    assert (target / ".env.template").exists()
    assert (target / "README.md").exists()
    body = (target / ".env.template").read_text(encoding="utf-8")
    assert "ASQAV_API_KEY" in body
    assert "ASQAV_AGENT_ID" in body
    assert "POSTGRES_PASSWORD" in body


def test_init_refuses_to_overwrite(tmp_path: Path) -> None:
    """A second `init` against the same dir exits non-zero unless --force is passed."""
    target = tmp_path / "stack"
    first = runner.invoke(app, ["shadow-ai", "init", "--dir", str(target)])
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, ["shadow-ai", "init", "--dir", str(target)])
    assert second.exit_code != 0
    assert "Refusing to overwrite" in second.output

    forced = runner.invoke(app, ["shadow-ai", "init", "--dir", str(target), "--force"])
    assert forced.exit_code == 0, forced.output


def test_status_handles_missing_compose_gracefully(tmp_path: Path) -> None:
    """`shadow-ai status` against a directory with no docker-compose.yml fails clearly."""
    missing = tmp_path / "does-not-exist"
    result = runner.invoke(app, ["shadow-ai", "status", "--dir", str(missing)])
    assert result.exit_code != 0
    assert "No docker-compose.yml" in result.output
    assert "asqav shadow-ai init" in result.output


def test_up_validates_env_present(tmp_path: Path) -> None:
    """`shadow-ai up` without a .env file errors with a clear message."""
    target = tmp_path / "stack"
    init = runner.invoke(app, ["shadow-ai", "init", "--dir", str(target)])
    assert init.exit_code == 0, init.output

    result = runner.invoke(app, ["shadow-ai", "up", "--dir", str(target)])
    assert result.exit_code != 0
    assert ".env" in result.output


def test_up_validates_required_env_vars(tmp_path: Path) -> None:
    """`shadow-ai up` rejects a .env that is missing required variables."""
    target = tmp_path / "stack"
    init = runner.invoke(app, ["shadow-ai", "init", "--dir", str(target)])
    assert init.exit_code == 0, init.output

    (target / ".env").write_text(
        "ASQAV_API_KEY=\nASQAV_AGENT_ID=\nPOSTGRES_PASSWORD=\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["shadow-ai", "up", "--dir", str(target)])
    assert result.exit_code != 0
    assert "ASQAV_API_KEY" in result.output


def test_init_with_upstream_override(tmp_path: Path) -> None:
    """`--upstream` rewrites OPENAI_UPSTREAM in the generated .env.template."""
    target = tmp_path / "stack"
    result = runner.invoke(
        app,
        ["shadow-ai", "init", "--dir", str(target), "--upstream", "https://proxy.example.com"],
    )
    assert result.exit_code == 0, result.output
    body = (target / ".env.template").read_text(encoding="utf-8")
    assert "OPENAI_UPSTREAM=https://proxy.example.com" in body
