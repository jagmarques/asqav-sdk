"""Framework integration extras for asqav.

Install specific extras:
    pip install asqav[langchain]
    pip install asqav[mcp]
    pip install asqav[crewai]
    pip install asqav[litellm]
    pip install asqav[haystack]
    pip install asqav[openai-agents]
    pip install asqav[llamaindex]
    pip install asqav[smolagents]
    pip install asqav[dspy]
    pip install asqav[letta]
    pip install asqav[all]
"""

# crewai is duck-typed (no hard import), so this default-on entrypoint is safe
# to re-export here. langchain/mcp entrypoints stay lazy (framework hard-imports).
from .crewai import enable_crew_governance

__all__ = ["enable_crew_governance"]
