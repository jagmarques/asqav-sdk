"""File-backed credential layer for the Asqav SDK.

Stores an API key (and optional API base) in ``~/.asqav/credentials`` so the SDK
and CLI can resolve a key without an environment variable. Resolution mirrors
:mod:`asqav.local`: explicit argument, then environment, then file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = [
    "CREDENTIALS_PATH",
    "credentials_path",
    "load_credentials",
    "save_credentials",
    "resolve_api_key",
    "resolve_api_base",
]

_DEFAULT_API_BASE = "https://api.asqav.com/api/v1"

CREDENTIALS_PATH = Path(os.path.expanduser("~")) / ".asqav" / "credentials"


def credentials_path() -> Path:
    """Resolve the credentials file location (env override, else ~/.asqav/credentials)."""
    override = os.environ.get("ASQAV_CREDENTIALS_PATH")
    if override:
        return Path(override)
    return Path(os.path.expanduser("~")) / ".asqav" / "credentials"


def load_credentials() -> dict[str, object]:
    """Read the credentials file. Missing or corrupt file returns {} (never raises)."""
    try:
        data = json.loads(credentials_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_credentials(api_key: str, api_base: str | None = None) -> Path:
    """Write the credentials file with mode 0600 under a mode 0700 ~/.asqav dir."""
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)

    payload: dict[str, str] = {"api_key": api_key}
    if api_base:
        payload["api_base"] = api_base

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    path.chmod(0o600)
    return path


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Resolve an API key: explicit arg, then ASQAV_API_KEY env, then credentials file."""
    if explicit:
        return explicit
    env_key = os.environ.get("ASQAV_API_KEY")
    if env_key:
        return env_key
    file_key = load_credentials().get("api_key")
    if isinstance(file_key, str) and file_key:
        return file_key
    return None


def resolve_api_base(explicit: str | None = None) -> str:
    """Resolve an API base: explicit arg, then ASQAV_API_BASE env, then file, then default."""
    if explicit:
        return explicit
    env_base = os.environ.get("ASQAV_API_BASE")
    if env_base:
        return env_base
    file_base = load_credentials().get("api_base")
    if isinstance(file_base, str) and file_base:
        return file_base
    return _DEFAULT_API_BASE
