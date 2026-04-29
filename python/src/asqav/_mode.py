"""Mode resolver for hash-only vs full-payload signing.

The asqav SDK supports two signing wire formats:

* ``"full-payload"``: send the raw ``context`` dict to the server. This is
  what every release before 0.3.1 did, and remains the default for
  self-hosted deployments where the server already holds the data.
* ``"hash-only"``: hash the canonical ``{action_type, context}`` locally
  and send only the hash plus a small whitelisted metadata bag. This is
  the GDPR data-minimization mode used by the asqav cloud (api.asqav.com).

Resolution precedence (highest first):

1. Explicit constructor / init kwarg (``mode="hash-only"``).
2. Environment variable ``ASQAV_MODE``.
3. Auto-detection from ``api_base_url``: hostnames matching the asqav
   cloud (``api.asqav.com`` or any ``*.asqav.com`` API host) default to
   hash-only. Anything else (localhost, custom domains, ``myasqav.com``,
   subdomain attacks like ``asqav.com.evil.com``) defaults to
   full-payload.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

Mode = Literal["hash-only", "full-payload"]
_VALID_EXPLICIT = {"auto", "hash-only", "full-payload"}
_VALID_ENV = {"hash-only", "full-payload"}


def _is_asqav_cloud_host(hostname: str | None) -> bool:
    """True if ``hostname`` is the asqav cloud API.

    Matches ``api.asqav.com`` exactly, or any subdomain of ``asqav.com``
    (e.g. ``staging.asqav.com``). Rejects look-alikes such as
    ``myasqav.com`` (different domain) and ``asqav.com.evil.com``
    (subdomain attack: the registrable domain is ``evil.com``).
    """
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
    """Resolve the wire mode for a client.

    Args:
        api_base_url: The configured API URL (may be ``None``).
        env: Value of the ``ASQAV_MODE`` environment variable, or ``None``.
        explicit: Explicit mode passed by the caller. ``"auto"`` (the
            default) defers to env then auto-detection.

    Returns:
        Either ``"hash-only"`` or ``"full-payload"``.

    Raises:
        ValueError: If ``explicit`` is not one of ``auto``, ``hash-only``,
            ``full-payload``.
    """
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
