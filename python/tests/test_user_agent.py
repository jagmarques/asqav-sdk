"""Every outbound HTTP call carries the asqav-python User-Agent.

The api.asqav.com edge 403s (Cloudflare error 1010) the bare
``Python-urllib/x.y`` default, which made every stranger's first sign call
fail with a bare "HTTP Error 403: Forbidden". These tests pin the header on
each transport so the regression cannot come back silently.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import urllib.request
from unittest.mock import patch

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav._useragent import USER_AGENT
from asqav.verifier import verify_receipt


def test_user_agent_format() -> None:
    assert re.fullmatch(
        r"asqav-python/\S+ \(\+https://www\.asqav\.com\)", USER_AGENT
    ), USER_AGENT


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_urllib_request_sends_user_agent() -> None:
    """The stdlib fallback transport (no httpx) sends the SDK User-Agent."""
    from asqav import client

    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["ua"] = req.get_header("User-agent")
        return _FakeResponse(json.dumps({"ok": True}).encode())

    with (
        patch.object(client, "_api_key", "sk_test"),
        patch.object(client, "_api_base", "https://api.example.test/api/v1/"),
        patch.object(urllib.request, "urlopen", fake_urlopen),
    ):
        client._urllib_request("GET", "health")

    assert captured["ua"] == USER_AGENT


def test_httpx_client_sends_user_agent() -> None:
    """init() builds the persistent httpx client with the SDK User-Agent."""
    from asqav import client

    if not client._HTTPX_AVAILABLE:
        import pytest

        pytest.skip("httpx not installed")

    import asqav

    asqav.init(api_key="sk_test_user_agent")
    assert client._client is not None
    assert client._client.headers["user-agent"] == USER_AGENT


def test_verifier_get_json_sends_user_agent() -> None:
    """The standalone verifier's network fetch sends a real User-Agent."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["ua"] = req.get_header("User-agent")
        return _FakeResponse(b"{}")

    with patch.object(urllib.request, "urlopen", fake_urlopen):
        verify_receipt._get_json("https://api.example.test/api/v1/verify/sig_x")

    assert captured["ua"] is not None
    assert captured["ua"].startswith("asqav-python/")


def test_async_client_headers_include_user_agent() -> None:
    """The async transport builds its httpx clients with the SDK User-Agent."""
    import inspect

    from asqav import async_client

    src = inspect.getsource(async_client)
    # Every AsyncClient that talks to the API carries the header.
    assert '"User-Agent": USER_AGENT' in src
