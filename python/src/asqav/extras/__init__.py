"""Framework integration extras for asqav.
Install one with ``pip install asqav[<extra>]``, e.g. ``asqav[langchain]``."""

# crewai is duck-typed (no hard import), so this default-on entrypoint is safe
# to re-export here. langchain entrypoints stay lazy (framework hard-imports).
from .crewai import enable_crew_governance

__all__ = ["enable_crew_governance"]
