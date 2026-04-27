"""Haystack integration for asqav.

Install: pip install asqav[haystack]

Usage::

    import asqav
    from asqav.extras.haystack import AsqavComponent
    from haystack import Pipeline

    asqav.init("sk_live_...")
    pipe = Pipeline()
    pipe.add_component("asqav", AsqavComponent(agent_name="my-pipeline"))
"""

from __future__ import annotations

from typing import Optional

try:
    from haystack import component
except ImportError:
    raise ImportError(
        "Haystack integration requires haystack-ai. "
        "Install with: pip install asqav[haystack]"
    )

from ._base import AsqavAdapter


@component
class AsqavComponent(AsqavAdapter):
    """Haystack pipeline component that signs data through asqav governance.

    Signs each pipeline run with data length and metadata keys.
    Signing failures are fail-open - data always flows through.

    Args:
        api_key: Optional API key override (uses asqav.init() default).
        agent_name: Name for a new agent (calls Agent.create).
        agent_id: ID of an existing agent (calls Agent.get).
    """

    @component.output_types(
        data=str,
        metadata=dict,
        signature_id=Optional[str],
    )
    def run(
        self,
        data: str,
        metadata: dict | None = None,
    ) -> dict:
        """Sign data flowing through the pipeline.

        Args:
            data: The data string to process.
            metadata: Optional metadata dict to pass through.

        Returns:
            Dict with data, metadata, and signature_id (None on sign failure).
        """
        resolved_metadata = metadata or {}
        sig = self._sign_action(
            "pipeline:process",
            {
                "data_length": len(data),
                "metadata_keys": list(resolved_metadata.keys()),
            },
        )
        return {
            "data": data,
            "metadata": resolved_metadata,
            "signature_id": sig.signature_id if sig else None,
        }
