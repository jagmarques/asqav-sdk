"""Tests for the file-backed credential layer and the onboarding CLI commands."""

from __future__ import annotations

import json
import os
import stat
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typer.testing import CliRunner  # noqa: E402

from asqav import credentials as creds  # noqa: E402
from asqav.cli import app  # noqa: E402

runner = CliRunner()


    # Use the default ~/.asqav path under a temp HOME (drop the env override).
@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.delenv("ASQAV_CREDENTIALS_PATH", raising=False)
    monkeypatch.delenv("ASQAV_API_KEY", raising=False)
    monkeypatch.delenv("ASQAV_API_BASE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


# === credentials module ===


def test_default_path_under_home(home) -> None:
    assert creds.credentials_path() == home / ".asqav" / "credentials"


def test_load_missing_file_returns_empty(home) -> None:
    assert creds.load_credentials() == {}


def test_resolve_api_key_missing_returns_none(home) -> None:
    assert creds.resolve_api_key() is None


def test_save_load_roundtrip(home) -> None:
    creds.save_credentials("sk_file_123", "https://example.test/api/v1")
    assert creds.load_credentials() == {
        "api_key": "sk_file_123",
        "api_base": "https://example.test/api/v1",
    }


def test_save_without_api_base_omits_key(home) -> None:
    creds.save_credentials("sk_only")
    assert creds.load_credentials() == {"api_key": "sk_only"}


def test_saved_file_mode_is_0600(home) -> None:
    path = creds.save_credentials("sk_secret")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_credentials_dir_mode_is_0700(home) -> None:
    creds.save_credentials("sk_secret")
    assert stat.S_IMODE(os.stat(home / ".asqav").st_mode) == 0o700


def test_corrupt_file_returns_empty(home) -> None:
    path = home / ".asqav" / "credentials"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert creds.load_credentials() == {}
    assert creds.resolve_api_key() is None


def test_resolve_precedence_arg_beats_env_beats_file(home, monkeypatch) -> None:
    creds.save_credentials("sk_file")
    monkeypatch.setenv("ASQAV_API_KEY", "sk_env")
    assert creds.resolve_api_key("sk_arg") == "sk_arg"
    assert creds.resolve_api_key() == "sk_env"
    monkeypatch.delenv("ASQAV_API_KEY")
    assert creds.resolve_api_key() == "sk_file"


def test_resolve_api_base_precedence_and_default(home, monkeypatch) -> None:
    assert creds.resolve_api_base() == "https://api.asqav.com/api/v1"
    creds.save_credentials("sk", "https://file.test/api")
    assert creds.resolve_api_base() == "https://file.test/api"
    monkeypatch.setenv("ASQAV_API_BASE", "https://env.test/api")
    assert creds.resolve_api_base() == "https://env.test/api"
    assert creds.resolve_api_base("https://arg.test/api") == "https://arg.test/api"


def test_credentials_path_env_override(home, monkeypatch) -> None:
    override = home / "custom" / "creds.json"
    monkeypatch.setenv("ASQAV_CREDENTIALS_PATH", str(override))
    assert creds.credentials_path() == override
    creds.save_credentials("sk_override")
    assert override.exists()
    assert creds.resolve_api_key() == "sk_override"


# === path-injection regression (Trustabl #375) ===


def test_credentials_path_rejects_traversal(home, monkeypatch) -> None:
    monkeypatch.setenv("ASQAV_CREDENTIALS_PATH", str(home / ".." / "escaped.json"))
    with pytest.raises(ValueError, match="path traversal"):
        creds.credentials_path()


def test_save_credentials_traversal_does_not_write_outside(home, monkeypatch) -> None:
    target = home.parent / "escaped_creds.json"
    monkeypatch.setenv("ASQAV_CREDENTIALS_PATH", str(home / ".." / "escaped_creds.json"))
    with pytest.raises(ValueError):
        creds.save_credentials("sk_evil")
    assert not target.exists()


def test_load_credentials_traversal_does_not_leak(home, monkeypatch) -> None:
    secret = home.parent / "secret.json"
    secret.write_text(json.dumps({"api_key": "sk_leaked"}), encoding="utf-8")
    monkeypatch.setenv("ASQAV_CREDENTIALS_PATH", str(home / ".." / "secret.json"))
    assert creds.load_credentials() == {}
    assert creds.resolve_api_key() is None


def test_validated_fs_path_rejects_null_byte(home) -> None:
    with pytest.raises(ValueError, match="null bytes"):
        creds._validated_fs_path(str(home / "creds") + "\x00.json", "ASQAV_CREDENTIALS_PATH")


def test_validated_fs_path_rejects_traversal(home) -> None:
    with pytest.raises(ValueError, match="path traversal"):
        creds._validated_fs_path(str(home / ".." / "x"), "ASQAV_CREDENTIALS_PATH")


def test_validated_fs_path_expands_user_and_allows_legit(home) -> None:
    assert creds._validated_fs_path("~/creds.json", "X") == home / "creds.json"
    legit = home / "custom" / "creds.json"
    assert creds._validated_fs_path(str(legit), "X") == legit


# === login command ===


@patch("asqav.client._get")
@patch("asqav.init")
def test_login_saves_credentials(mock_init: MagicMock, mock_get: MagicMock, tmp_path) -> None:
    mock_get.return_value = {"agents": []}
    result = runner.invoke(app, ["login", "--api-key", "sk_test_123"])
    assert result.exit_code == 0, result.output
    assert "Saved API key" in result.output
    saved = json.loads((tmp_path / "credentials").read_text())
    assert saved["api_key"] == "sk_test_123"


@patch("asqav.client._get")
@patch("asqav.init")
def test_login_does_not_save_on_validation_failure(
    mock_init: MagicMock, mock_get: MagicMock, tmp_path
) -> None:
    from asqav.client import AuthenticationError

    mock_get.side_effect = AuthenticationError("Invalid API key")
    result = runner.invoke(app, ["login", "--api-key", "sk_bad"])
    assert result.exit_code == 1
    assert "validation failed" in result.output
    assert not (tmp_path / "credentials").exists()


@patch("asqav.client._get")
@patch("asqav.init")
def test_login_refuses_overwrite_without_force(
    mock_init: MagicMock, mock_get: MagicMock, tmp_path
) -> None:
    mock_get.return_value = {"agents": []}
    assert runner.invoke(app, ["login", "--api-key", "sk_one"]).exit_code == 0
    second = runner.invoke(app, ["login", "--api-key", "sk_two"])
    assert second.exit_code == 1
    assert "already exists" in second.output
    saved = json.loads((tmp_path / "credentials").read_text())
    assert saved["api_key"] == "sk_one"


@patch("asqav.client._get")
@patch("asqav.init")
def test_login_force_overwrites(
    mock_init: MagicMock, mock_get: MagicMock, tmp_path
) -> None:
    mock_get.return_value = {"agents": []}
    runner.invoke(app, ["login", "--api-key", "sk_one"])
    result = runner.invoke(app, ["login", "--api-key", "sk_two", "--force"])
    assert result.exit_code == 0
    saved = json.loads((tmp_path / "credentials").read_text())
    assert saved["api_key"] == "sk_two"


# === whoami / status commands ===


@patch("asqav.client._get")
@patch("asqav.init")
def test_whoami_reports_env_source(
    mock_init: MagicMock, mock_get: MagicMock, monkeypatch
) -> None:
    monkeypatch.setenv("ASQAV_API_KEY", "sk_env")
    mock_get.return_value = {"agents": [{"agent_id": "a1"}, {"agent_id": "a2"}]}
    result = runner.invoke(app, ["whoami"])
    assert result.exit_code == 0, result.output
    assert "Key source: env" in result.output
    assert "Key valid" in result.output
    assert "2 agent(s)" in result.output


@patch("asqav.client._get")
@patch("asqav.init")
def test_whoami_reports_file_source(
    mock_init: MagicMock, mock_get: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("ASQAV_API_KEY", raising=False)
    (tmp_path / "credentials").write_text(json.dumps({"api_key": "sk_file"}))
    mock_get.return_value = []
    result = runner.invoke(app, ["whoami"])
    assert result.exit_code == 0, result.output
    assert "Key source: file" in result.output


@patch("asqav.client._get")
@patch("asqav.init")
def test_whoami_reports_arg_source(
    mock_init: MagicMock, mock_get: MagicMock, monkeypatch
) -> None:
    monkeypatch.delenv("ASQAV_API_KEY", raising=False)
    mock_get.return_value = []
    result = runner.invoke(app, ["whoami", "--api-key", "sk_arg"])
    assert result.exit_code == 0
    assert "Key source: arg" in result.output


def test_whoami_no_key_exits_nonzero(monkeypatch) -> None:
    monkeypatch.delenv("ASQAV_API_KEY", raising=False)
    result = runner.invoke(app, ["whoami"])
    assert result.exit_code == 1
    assert "asqav login" in result.output


@patch("asqav.client._get")
@patch("asqav.init")
def test_whoami_rejected_key_exits_nonzero(
    mock_init: MagicMock, mock_get: MagicMock, monkeypatch
) -> None:
    from asqav.client import AuthenticationError

    monkeypatch.setenv("ASQAV_API_KEY", "sk_bad")
    mock_get.side_effect = AuthenticationError("Invalid API key")
    result = runner.invoke(app, ["whoami"])
    assert result.exit_code == 1
    assert "rejected" in result.output


@patch("asqav.client._get")
@patch("asqav.init")
def test_status_alias_reports_source(
    mock_init: MagicMock, mock_get: MagicMock, monkeypatch
) -> None:
    monkeypatch.setenv("ASQAV_API_KEY", "sk_env")
    mock_get.return_value = []
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Key source: env" in result.output


# === init command ===


def test_init_prints_snippet_without_writing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert "Detected framework: python" in result.output
    assert "asqav.govern" in result.output
    assert "@asqav.secure" in result.output
    assert os.listdir(tmp_path) == []


def test_init_detects_framework(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "requirements.txt").write_text("litellm>=1.0\n")
    result = runner.invoke(app, ["init"])
    assert "Detected framework: litellm" in result.output
    assert "api:litellm:completion" in result.output


def test_init_write_creates_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "--write"])
    assert result.exit_code == 0
    written = tmp_path / "asqav_governance.py"
    assert written.exists()
    assert "asqav.govern" in written.read_text()


def test_init_demo_skips_without_key(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASQAV_API_KEY", raising=False)
    result = runner.invoke(app, ["init", "--demo"])
    assert result.exit_code == 0
    assert "skipping demo" in result.output


@patch("asqav.govern")
def test_init_demo_signs_with_key(
    mock_govern: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASQAV_API_KEY", "sk_env")
    mock_govern.return_value.sign.return_value = MagicMock(signature_id="sig_demo")
    result = runner.invoke(app, ["init", "--demo"])
    assert result.exit_code == 0, result.output
    assert "sig_demo" in result.output
    assert "asqav verify sig_demo" in result.output
    mock_govern.assert_called_once_with(api_key="sk_env", agent_name="asqav-init-demo")
