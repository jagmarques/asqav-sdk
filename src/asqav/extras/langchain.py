"""LangChain integration for asqav.

Install: pip install asqav[langchain]

Usage (Phase 8):
    from asqav.extras.langchain import AsqavCallbackHandler
    handler = AsqavCallbackHandler()
    chain.invoke(input, config={"callbacks": [handler]})
"""

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install asqav[langchain]"
    )


class AsqavCallbackHandler(BaseCallbackHandler):
    """Callback handler that signs LangChain chain runs and tool calls."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "AsqavCallbackHandler will be implemented in Phase 8. "
            "See: https://asqav.com/docs/integrations/langchain"
        )
