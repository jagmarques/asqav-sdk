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


def test_init_calls_ensure_policy_when_api_key_set(
    tmp_path: Path, monkeypatch
) -> None:
    """`init` triggers _ensure_shadow_ai_policy when ASQAV_API_KEY is in env."""
    from asqav import cli as cli_mod

    calls: list[bool] = []
    monkeypatch.setattr(cli_mod, "_ensure_shadow_ai_policy", lambda: calls.append(True) or True)
    monkeypatch.setenv("ASQAV_API_KEY", "sk_test_dummy")

    target = tmp_path / "stack"
    result = runner.invoke(app, ["shadow-ai", "init", "--dir", str(target)])
    assert result.exit_code == 0, result.output
    assert calls == [True]


def test_init_prints_hint_when_api_key_absent(tmp_path: Path, monkeypatch) -> None:
    """`init` skips ensure_policy and prints the hint when ASQAV_API_KEY is unset."""
    from asqav import cli as cli_mod

    calls: list[bool] = []
    monkeypatch.setattr(cli_mod, "_ensure_shadow_ai_policy", lambda: calls.append(True) or True)
    monkeypatch.delenv("ASQAV_API_KEY", raising=False)

    target = tmp_path / "stack"
    result = runner.invoke(app, ["shadow-ai", "init", "--dir", str(target)])
    assert result.exit_code == 0, result.output
    assert calls == []
    assert "ensure-policy" in result.output


def test_ensure_policy_noop_when_present(monkeypatch) -> None:
    """`ensure-policy` exits 0 with an 'already present' line when a matching policy exists."""
    from asqav import cli as cli_mod
    from asqav import client as client_mod

    monkeypatch.setattr(cli_mod, "_init_sdk", lambda: None)
    monkeypatch.setattr(
        client_mod,
        "_get",
        lambda path: [
            {"id": "pol_existing_abc", "action_pattern": "*", "action": "monitor", "is_active": True}
        ],
    )

    def _refuse_post(path, data):
        raise AssertionError("ensure-policy must not POST when a matching policy already exists")

    monkeypatch.setattr(client_mod, "_post", _refuse_post)

    result = runner.invoke(app, ["shadow-ai", "ensure-policy"])
    assert result.exit_code == 0, result.output
    assert "already present" in result.output
    assert "pol_existing_abc" in result.output


def test_ensure_policy_creates_when_absent(monkeypatch) -> None:
    """`ensure-policy` POSTs a new monitor policy when none matches."""
    from asqav import cli as cli_mod
    from asqav import client as client_mod

    monkeypatch.setattr(cli_mod, "_init_sdk", lambda: None)
    monkeypatch.setattr(client_mod, "_get", lambda path: [])

    posted: list[dict] = []

    def _capture_post(path, data):
        posted.append({"path": path, "data": data})
        return {"id": "pol_new_xyz", "action_pattern": "*", "action": "monitor", "is_active": True}

    monkeypatch.setattr(client_mod, "_post", _capture_post)

    result = runner.invoke(app, ["shadow-ai", "ensure-policy"])
    assert result.exit_code == 0, result.output
    assert "Created shadow-AI monitor policy" in result.output
    assert "pol_new_xyz" in result.output
    assert len(posted) == 1
    assert posted[0]["path"] == "/policies"
    assert posted[0]["data"]["action_pattern"] == "*"
    assert posted[0]["data"]["action"] == "monitor"
    assert posted[0]["data"]["is_active"] is True


def test_ensure_policy_exits_nonzero_when_list_fails(monkeypatch) -> None:
    """`ensure-policy` exits 1 with a diagnostic when listing policies raises."""
    from asqav import cli as cli_mod
    from asqav import client as client_mod

    monkeypatch.setattr(cli_mod, "_init_sdk", lambda: None)

    def _boom(path):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(client_mod, "_get", _boom)

    result = runner.invoke(app, ["shadow-ai", "ensure-policy"])
    assert result.exit_code == 1
    assert "Could not list policies" in result.output


def test_ensure_policy_exits_nonzero_when_create_fails(monkeypatch) -> None:
    """`ensure-policy` exits 1 with a diagnostic when POSTing the new policy raises."""
    from asqav import cli as cli_mod
    from asqav import client as client_mod

    monkeypatch.setattr(cli_mod, "_init_sdk", lambda: None)
    monkeypatch.setattr(client_mod, "_get", lambda path: [])

    def _boom(path, data):
        raise RuntimeError("server returned 403")

    monkeypatch.setattr(client_mod, "_post", _boom)

    result = runner.invoke(app, ["shadow-ai", "ensure-policy"])
    assert result.exit_code == 1
    assert "Could not create shadow-ai monitor policy" in result.output
