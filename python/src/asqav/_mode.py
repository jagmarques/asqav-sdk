"""Mode resolver for hash-only vs full-payload signing.

Resolution order: explicit kwarg, ``ASQAV_MODE`` env, then hostname auto-detection
(``*.asqav.com`` -> hash-only, else full-payload).
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

Mode = Literal["hash-only", "full-payload"]
_VALID_EXPLICIT = {"auto", "hash-only", "full-payload"}
_VALID_ENV = {"hash-only", "full-payload"}


def _is_asqav_cloud_host(hostname: str | None) -> bool:
    """True if ``hostname`` is ``api.asqav.com`` or any ``*.asqav.com`` subdomain."""
    if not hostname:
        return False
    h = hostname.lower().strip().rstrip(".")
    if h == "asqav.com":
        return False  # marketing site, not the API
    return h == "api.asqav.com" or h.endswith(".asqav.com")


def _resolve_mode(
    api_base_url: str | None,
    env: str | None,
    explicit: str | None = "auto",
) -> Mode:
    """Resolve the wire mode; ``explicit`` wins, then ``env``, then host auto-detection."""
    if explicit is None:
        explicit = "auto"
    if explicit not in _VALID_EXPLICIT:
        raise ValueError(
            f"mode must be one of {sorted(_VALID_EXPLICIT)}, got {explicit!r}"
        )
    if explicit != "auto":
        return explicit  # type: ignore[return-value]

    if env:
        env_norm = env.strip().lower()
        if env_norm in _VALID_ENV:
            return env_norm  # type: ignore[return-value]
        # silently ignore garbage env values; fall through to auto

    hostname: str | None = None
    if api_base_url:
        try:
            parsed = urlparse(
                api_base_url
                if "://" in api_base_url
                else f"https://{api_base_url}"
            )
            hostname = parsed.hostname
        except Exception:  # pragma: no cover - extremely defensive
            hostname = None

    return "hash-only" if _is_asqav_cloud_host(hostname) else "full-payload"
