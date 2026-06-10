"""URL-join regression tests for the stdlib (no-httpx) fallback.

A plain `pip install asqav` ships without httpx, so the SDK falls back to
urllib. That path used urljoin(base, "/agents/create"), and urljoin RESETS
a leading-slash path to the host root: the /api/v1 prefix was stripped and
the first quickstart call 404ed. These tests pin _join_url and pin that the
urllib fallback requests the exact URL the httpx client would.
"""

import urllib.request
from unittest.mock import MagicMock

import pytest

from asqav import client as client_mod
from asqav.client import _join_url

BASE_WITH_PATH = "https://api.asqav.com/api/v1"
BASE_NO_PATH = "https://api.asqav.com"


# === _join_url helper ===


def test_join_preserves_base_path() -> None:
    assert (
        _join_url(BASE_WITH_PATH, "/agents/create")
        == "https://api.asqav.com/api/v1/agents/create"
    )


def test_join_base_without_path() -> None:
    assert (
        _join_url(BASE_NO_PATH, "/agents/create")
        == "https://api.asqav.com/agents/create"
    )


def test_join_trailing_slash_base() -> None:
    assert (
        _join_url(BASE_WITH_PATH + "/", "/agents/create")
        == "https://api.asqav.com/api/v1/agents/create"
    )


def test_join_path_without_leading_slash() -> None:
    assert (
        _join_url(BASE_WITH_PATH, "agents/create")
        == "https://api.asqav.com/api/v1/agents/create"
    )


def test_join_keeps_query_string() -> None:
    assert (
        _join_url(BASE_WITH_PATH, "/export/csv?agent_id=agt_1")
        == "https://api.asqav.com/api/v1/export/csv?agent_id=agt_1"
    )


# === urllib fallback parity with httpx ===


class _FakeResponse:
    def __init__(self, body: bytes = b"{}") -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


@pytest.mark.parametrize("base", [BASE_WITH_PATH, BASE_NO_PATH])
def test_urllib_fallback_requests_same_url_as_httpx(
    monkeypatch: pytest.MonkeyPatch, base: str
) -> None:
    """The stdlib fallback must hit the exact URL the httpx client would."""
    httpx = pytest.importorskip("httpx")
    expected = str(
        httpx.Client(base_url=base).build_request("POST", "/agents/create").url
    )

    captured: dict[str, str] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0) -> _FakeResponse:
        captured["url"] = request.full_url
        return _FakeResponse()

    monkeypatch.setattr(client_mod, "_api_key", "sk_test")
    monkeypatch.setattr(client_mod, "_api_base", base)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client_mod._urllib_request("POST", "/agents/create", {"name": "qa"})

    assert captured["url"] == expected
    if base == BASE_WITH_PATH:
        assert captured["url"] == "https://api.asqav.com/api/v1/agents/create"


def test_urllib_fallback_never_strips_api_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct pin of the funnel bug: prefix survives for the quickstart call."""
    captured: dict[str, str] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0) -> _FakeResponse:
        captured["url"] = request.full_url
        return _FakeResponse()

    monkeypatch.setattr(client_mod, "_api_key", "sk_test")
    monkeypatch.setattr(client_mod, "_api_base", BASE_WITH_PATH)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client_mod._urllib_request("POST", "/agents/create", {"name": "qa"})

    assert captured["url"].startswith("https://api.asqav.com/api/v1/")


# === audit pack export URL (was doubled to /api/v1/api/v1/...) ===


def test_fetch_audit_pack_url_has_single_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asqav import compliance

    fake_httpx = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {"receipts": []}
    fake_httpx.post.return_value = resp

    monkeypatch.setattr(client_mod, "_api_key", "sk_test")
    monkeypatch.setattr(client_mod, "_api_base", BASE_WITH_PATH)
    monkeypatch.setattr(client_mod, "httpx", fake_httpx)

    compliance.fetch_audit_pack(
        start="2026-04-01T00:00:00Z", end="2026-05-01T00:00:00Z"
    )

    url = fake_httpx.post.call_args.args[0]
    assert url == "https://api.asqav.com/api/v1/audit-pack/export"
    assert "/api/v1/api/v1/" not in url
