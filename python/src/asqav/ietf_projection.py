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
    # signature may be the object form directly or under signatureObject.
    direct_signature = _get(response, "signature")
    if isinstance(direct_signature, dict) and direct_signature:
        return dict(direct_signature)
    cloud_supplied = _get(response, "signatureObject")
    if isinstance(cloud_supplied, dict) and cloud_supplied:
        return dict(cloud_supplied)

    if not _get(response, "compliance_mode"):
        return None

    alg = _get(response, "algorithm") or ""
    sig = _get(response, "signature_b64") or _get(response, "signature") or ""
    sig = sig if isinstance(sig, str) else ""
    kid = _get(response, "kid") or _get(response, "issuer_id") or _get(response, "key_id") or ""
    return {"alg": alg, "kid": kid, "sig": sig}


def anchors_from_response(response: Any) -> list[dict[str, Any]] | None:
    """Project a SignatureResponse / verify dict into `anchors[]`.

    IETF -01 N7 / Section 4.4. Returns the cloud-supplied list when
    present (deep-copied so callers cannot mutate the response),
    otherwise builds entries from the flat columns
    (`rfc3161_serial`, `rfc3161_tsa`, `bitcoin_anchor.status`).
    Returns None when the response is non-compliance so legacy
    callers stay byte-stable; returns [] (not None) when
    compliance_mode is on but no anchor columns are populated.
    """
    cloud_supplied = _get(response, "anchors")
    if isinstance(cloud_supplied, list):
        return [dict(e) for e in cloud_supplied]

    if not _get(response, "compliance_mode"):
        return None

    anchors: list[dict[str, Any]] = []

    rfc3161_serial = _get(response, "rfc3161_serial")
    rfc3161_tsa = _get(response, "rfc3161_tsa")
    if rfc3161_serial or rfc3161_tsa:
        entry: dict[str, Any] = {"type": "rfc3161", "value": ""}
        if rfc3161_serial:
            entry["serial"] = rfc3161_serial
        if rfc3161_tsa:
            entry["tsa"] = rfc3161_tsa
        anchors.append(entry)

    bitcoin_anchor = _get(response, "bitcoin_anchor")
    if bitcoin_anchor is not None:
        # The SDK BitcoinAnchor dataclass exposes `status`; the OTS
        # proof bytes are not in the SDK type, so we surface a
        # presence-only entry. Strict verifiers ignore this projection
        # unless the cloud has populated `anchors[]` directly.
        status = _get(bitcoin_anchor, "status")
        if status:
            anchors.append(
                {
                    "type": "opentimestamps",
                    "value": "",
                    "status": status,
                }
            )

    return anchors


def encode_anchor_value(raw: bytes) -> str:
    """Helper for callers that hold raw anchor bytes locally."""
    import base64

    return base64.b64encode(raw).decode()


def _get(obj: Any, name: str) -> Any:
    """Read attribute or dict key, whichever exists. None when absent."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
