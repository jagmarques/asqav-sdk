"""Shared full-envelope archive pages (criterion 761)."""

from __future__ import annotations

import base64
import hashlib

from asqav._jcs import canonical_json


def _ed25519_keypair() -> tuple[bytes, bytes]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    from cryptography.hazmat.primitives import serialization

    public_bytes = public.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return public_bytes, private


def _envelope(organization_id, request, window, next_cursor, has_more, receipts):
    return {
        "organization_id": organization_id,
        "request": request,
        "window": window,
        "next_cursor": next_cursor,
        "has_more": has_more,
        "receipts": receipts,
    }


def _page_full(
    receipts,
    next_cursor,
    has_more,
    *,
    cursor=0,
    limit=100,
    keypair=None,
    organization_id="org_arc",
):
    """A page shaped exactly like GET /audit-pack/archive emits."""
    public_bytes, private = keypair or _ed25519_keypair()
    request = {
        "cursor": cursor,
        "limit": limit,
        "agent_id": "",
        "only_compliance": True,
    }
    window = {"first_id": 1 if receipts else None, "last_id": 1 if receipts else None}
    body = _envelope(organization_id, request, window, next_cursor, has_more, receipts)
    page_bytes = canonical_json(body)
    signature = private.sign(page_bytes)
    return {
        "organization_id": organization_id,
        "receipts": receipts,
        "next_cursor": next_cursor,
        "has_more": has_more,
        "page_request": request,
        "id_window": window,
        "page_digest": "sha256:" + hashlib.sha256(page_bytes).hexdigest(),
        "page_signature": base64.b64encode(signature).decode("ascii"),
        "page_signature_algorithm": "Ed25519",
        "page_public_key": base64.b64encode(public_bytes).decode("ascii"),
    }
