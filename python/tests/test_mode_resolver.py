"""Tests for the hash-only / full-payload mode resolver."""

from __future__ import annotations

import pytest

from asqav._mode import _is_asqav_cloud_host, _resolve_mode

# ---------------------------------------------------------------------------
# Cloud host detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host,expected",
    [
        ("api.asqav.com", True),
        ("API.ASQAV.COM", True),
        ("staging.asqav.com", True),
        ("app.asqav.com", True),
        ("foo.bar.asqav.com", True),
        ("asqav.com", False),  # marketing site, not API
        ("myasqav.com", False),  # different domain entirely
        ("asqav.com.evil.com", False),  # subdomain attack
        ("evilasqav.com", False),
        ("localhost", False),
        ("127.0.0.1", False),
        ("internal.example.com", False),
        ("", False),
        (None, False),
    ],
)
def test_is_asqav_cloud_host(host: str | None, expected: bool) -> None:
    assert _is_asqav_cloud_host(host) is expected


# ---------------------------------------------------------------------------
# Resolution precedence
# ---------------------------------------------------------------------------


def test_explicit_overrides_env_and_url() -> None:
    """Explicit always wins."""
    assert _resolve_mode(
        api_base_url="https://api.asqav.com/api/v1",
        env="full-payload",
        explicit="hash-only",
    ) == "hash-only"
    assert _resolve_mode(
        api_base_url="https://localhost:8000",
        env="hash-only",
        explicit="full-payload",
    ) == "full-payload"


def test_env_overrides_auto_detection() -> None:
    """When explicit is auto, env wins over URL."""
    assert _resolve_mode(
        api_base_url="https://localhost:8000",
        env="hash-only",
        explicit="auto",
    ) == "hash-only"
    assert _resolve_mode(
        api_base_url="https://api.asqav.com/api/v1",
        env="full-payload",
        explicit="auto",
    ) == "full-payload"


def test_garbage_env_falls_through_to_auto() -> None:
    """Unknown env values are silently ignored."""
    assert _resolve_mode(
        api_base_url="https://api.asqav.com/api/v1",
        env="banana",
        explicit="auto",
    ) == "hash-only"


def test_invalid_explicit_raises() -> None:
    with pytest.raises(ValueError):
        _resolve_mode(api_base_url=None, env=None, explicit="banana")


# ---------------------------------------------------------------------------
# Auto detection scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://api.asqav.com/api/v1", "hash-only"),
        ("https://app.asqav.com", "hash-only"),
        ("https://staging.asqav.com/api/v1", "hash-only"),
        ("http://localhost:8000", "full-payload"),
        ("http://localhost:8000/api/v1", "full-payload"),
        ("https://internal.example.com", "full-payload"),
        ("https://myasqav.com/api/v1", "full-payload"),
        ("https://asqav.com.evil.com", "full-payload"),
        ("https://asqav.com", "full-payload"),  # marketing, not API
        ("api.asqav.com/api/v1", "hash-only"),  # missing scheme
        (None, "full-payload"),
        ("", "full-payload"),
    ],
)
def test_auto_resolves_correctly(url: str | None, expected: str) -> None:
    assert _resolve_mode(api_base_url=url, env=None, explicit="auto") == expected
