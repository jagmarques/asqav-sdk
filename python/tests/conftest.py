"""Test-suite isolation for the file-backed credential layer.

Points ASQAV_CREDENTIALS_PATH at a per-test nonexistent path so no test reads a
real ~/.asqav/credentials file, which would otherwise flip the "no key" paths.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_asqav_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ASQAV_CREDENTIALS_PATH", str(tmp_path / "credentials"))


@pytest.fixture
def signing_transport():
    from asqav.client import _transport

    saved = (_transport._mode, _transport._org_salt, _transport._api_key)
    _transport._api_key = "sk_test"
    yield
    _transport._mode, _transport._org_salt, _transport._api_key = saved
