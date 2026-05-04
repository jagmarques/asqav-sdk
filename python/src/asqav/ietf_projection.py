"""IETF Compliance Receipts -01 wire-shape projection helpers.

Mirror of `asqav_cloud.core.ietf_projection`. Pure functions that
re-shape the legacy flat fields (algorithm/key_id/signature_b64) into
the spec-shape JSON structures the IETF profile defines (Section 4
"JSON Receipt Structure").

Callers use these helpers to produce a draft-conformant view of a
receipt for offline verification or audit-pack export when working
against an older cloud that does not yet emit `signatureObject`
directly.
"""

from __future__ import annotations

from typing import Any


def signature_object_from_response(response: Any) -> dict[str, str] | None:
    """Project a SignatureResponse / verify dict into `{alg, kid, sig}`.

    Returns the cloud-supplied `signatureObject` when present so callers
    do not double-project; otherwise builds the envelope from the flat
    fields (`algorithm`, `key_id` if exposed, `signature`). Returns None
    when the response is non-compliance (no `compliance_mode=True`) so
    legacy receipts are not silently re-shaped.
    """
    cloud_supplied = _get(response, "signatureObject")
    if isinstance(cloud_supplied, dict) and cloud_supplied:
        return dict(cloud_supplied)

    if not _get(response, "compliance_mode"):
        return None

    alg = _get(response, "algorithm") or ""
    sig = _get(response, "signature") or ""
    # `key_id` is not surfaced on the dataclass today; cloud will pass
    # it via `signatureObject` once available. Fall back to the empty
    # string so the envelope keeps the spec-shape three keys.
    kid = _get(response, "key_id") or _get(response, "kid") or ""
    return {"alg": alg, "kid": kid, "sig": sig}


def _get(obj: Any, name: str) -> Any:
    """Read attribute or dict key, whichever exists. None when absent."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
