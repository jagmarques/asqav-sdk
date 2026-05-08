"""Compliance Receipts wire-shape projection helpers.

Pure helpers for offline analysis of a receipt: extract the spec-shape
signature envelope `{alg, kid, sig}` and the anchors[] array. The
cloud emits the three-key envelope directly under compliance_mode; the
helpers below return the cloud-supplied values and None on legacy
receipts.
"""

from __future__ import annotations

from typing import Any


def signature_envelope_from_response(response: Any) -> dict[str, str] | None:
    """Return `{alg, kid, sig}` when the response carries it, else None.

    The cloud emits `signature` as the object form under compliance_mode
    and as a base64 string in legacy mode. None on legacy receipts so
    callers can branch cleanly.
    """
    sig = _get(response, "signature")
    if isinstance(sig, dict):
        keys = set(sig.keys())
        if keys >= {"alg", "kid", "sig"}:
            return dict(sig)
    return None


def anchors_from_response(response: Any) -> list[dict[str, Any]] | None:
    """Return the `anchors[]` array when present, else None.

    Cloud emits the array directly under compliance_mode (possibly as
    []). None on legacy receipts.
    """
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
