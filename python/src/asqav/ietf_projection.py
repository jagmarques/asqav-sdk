"""Compliance Receipts wire-shape projection helpers for offline receipt analysis."""

from __future__ import annotations

from typing import Any


def signature_envelope_from_response(response: Any) -> dict[str, str] | None:
    """Return ``{alg, kid, sig}`` for compliance-mode receipts, else ``None`` (flat shape)."""
    sig = _get(response, "signature")
    if isinstance(sig, dict):
        keys = set(sig.keys())
        if keys >= {"alg", "kid", "sig"}:
            return dict(sig)
    return None


def anchors_from_response(response: Any) -> list[dict[str, Any]] | None:
    """Return the ``anchors[]`` array when present, else ``None``."""
    anchors = _get(response, "anchors")
    if isinstance(anchors, list):
        return [dict(e) for e in anchors]
    return None


def encode_anchor_value(raw: bytes) -> str:
    """Helper for callers that hold raw anchor bytes locally."""
    import base64

    return base64.b64encode(raw).decode()


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
