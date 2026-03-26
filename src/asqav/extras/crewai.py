"""CrewAI integration for asqav.

Install: pip install asqav[crewai]

Usage (Phase 8):
    from asqav.extras.crewai import AsqavCrewHook
    hook = AsqavCrewHook()
"""

try:
    import crewai  # noqa: F401
except ImportError:
    raise ImportError(
        "CrewAI integration requires crewai. "
        "Install with: pip install asqav[crewai]"
    )


class AsqavCrewHook:
    """Hook that signs CrewAI task lifecycle events."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "AsqavCrewHook will be implemented in Phase 8. "
            "See: https://asqav.com/docs/integrations/crewai"
        )
