"""Test-suite isolation for the file-backed credential layer.

Points ASQAV_CREDENTIALS_PATH at a per-test nonexistent path so no test reads a
real ~/.asqav/credentials file, which would otherwise flip the "no key" paths.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_asqav_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ASQAV_CREDENTIALS_PATH", str(tmp_path / "credentials"))
