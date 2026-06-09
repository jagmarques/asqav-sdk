"""Shared User-Agent for every outbound HTTP call the SDK makes.

The api.asqav.com edge runs a Cloudflare browser-integrity check that
rejects the bare ``Python-urllib/x.y`` default with a 403 (error 1010),
so every request must identify itself with a real product token.
"""

from __future__ import annotations


def _sdk_version() -> str:
    """Resolve the installed package version without importing the package."""
    try:
        from importlib.metadata import version

        return version("asqav")
    except Exception:
        # Source checkout without installed dist metadata.
        try:
            from asqav import __version__

            return __version__
        except Exception:
            return "unknown"


USER_AGENT = f"asqav-python/{_sdk_version()} (+https://www.asqav.com)"
