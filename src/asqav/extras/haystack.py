"""Haystack integration for asqav.

Install: pip install asqav[haystack]

Usage (Phase 8):
    from asqav.extras.haystack import AsqavComponent
    component = AsqavComponent()
"""

try:
    from haystack import component  # noqa: F401
except ImportError:
    raise ImportError(
        "Haystack integration requires haystack-ai. "
        "Install with: pip install asqav[haystack]"
    )


class AsqavComponent:
    """Pipeline component that signs Haystack pipeline runs."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "AsqavComponent will be implemented in Phase 8. "
            "See: https://asqav.com/docs/integrations/haystack"
        )
