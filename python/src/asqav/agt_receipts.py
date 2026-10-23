"""Asqav as an independent witness for Microsoft AGT receipts (asqav[agt]).
Asqav signs its own receipt linking the AGT digest; it never signs AGT bytes."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

try:
    from mcp_receipt_governed import receipt as _agt
except ImportError as err:
    raise ImportError(
        "The AGT integration requires Microsoft Agent Governance Toolkit receipts. "
        "Install with: pip install 'asqav[agt]'"
    ) from err

from .client import _compute_action_ref
from .extras._base import AsqavAdapter

__all__ = [
    "AUTHORIZER_PREFIX",
    "AUTHORIZE_ACTION",
    "COUNTERSIGN_ACTION",
    "AgtSigningError",
    "AsqavAgtSigner",
    "AsqavLink",
    "supports_external_authorization",
    "verify_asqav_link",
]

#: Prefix of the AGT ``authorizer_id`` that names an Asqav receipt.
AUTHORIZER_PREFIX = "asqav:"
#: Asqav action type for a receipt recorded with ``countersign``.
COUNTERSIGN_ACTION = "agt:countersign"
#: Action type for a receipt authorized through AGT's external profile.
AUTHORIZE_ACTION = "agt:authorize"

#: Signed decisions each link kind accepts; authorization needs ``allow``.
_ACCEPTED_DECISIONS = {
    COUNTERSIGN_ACTION: frozenset({"allow", "observation"}),
    AUTHORIZE_ACTION: frozenset({"allow"}),
}


class AgtSigningError(RuntimeError):
    """Raised when Asqav cannot sign or authorize an AGT receipt; nothing is authorized."""


@dataclass(frozen=True)
class AsqavLink:
    """Reference from one AGT receipt to the Asqav receipt that signed it.

    ``action_ref`` is the sha256 of the canonical Asqav action binding the AGT
    receipt id, payload hash and signer key (plus nonce, expiry and authorizer
    key when authorized). ``asqav_receipt`` is the signed Asqav envelope.
    """

    agt_receipt_id: str
    action_type: str
    action_ref: str
    signature_id: str
    verification_url: str
    asqav_receipt: dict[str, Any] = field(repr=False)


def supports_external_authorization() -> bool:
    """True when the installed AGT ships its external-authorization profile."""
    return callable(getattr(_agt, "authorize_receipt", None))


def _ed25519_public_hex(seed_hex: str) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    try:
        key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
    except (TypeError, ValueError) as exc:
        raise ValueError("authorizer_key_hex must be a 32-byte Ed25519 seed in hex") from exc
    return key.public_key().public_bytes_raw().hex()


def _link_context(receipt: Any, authorization: dict[str, str] | None = None) -> dict[str, str]:
    context = {
        "agt_receipt_id": str(receipt.receipt_id),
        "agt_payload_hash": receipt.payload_hash(),
        "agt_signer_public_key": str(receipt.signer_public_key or "").lower(),
        "agt_cedar_decision": str(receipt.cedar_decision),
    }
    context.update(authorization or {})
    return context


def _authorization_fields(nonce: str, expires_at: float, authorizer_key: str) -> dict[str, str]:
    # Floats never enter the canonical context; repr round-trips the value.
    return {
        "authorization_nonce": nonce,
        "authorization_expires_at": repr(float(expires_at)),
        "authorizer_public_key": authorizer_key.lower(),
    }


def _expected_action(receipt: Any) -> tuple[str, dict[str, str]]:
    if getattr(receipt, "assurance_level", "self_attested") != "externally_authorized":
        return COUNTERSIGN_ACTION, _link_context(receipt)
    nonce = receipt.authorization_nonce
    expires_at = receipt.authorization_expires_at
    key = receipt.authorizer_public_key
    if not isinstance(nonce, str) or not isinstance(key, str):
        raise ValueError("AGT receipt is missing its authorization nonce or key")
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        raise ValueError("AGT receipt has a malformed authorization expiry")
    fields = _authorization_fields(nonce, expires_at, key)
    return AUTHORIZE_ACTION, _link_context(receipt, fields)


def _envelope(sig: Any) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "signature": sig.signature,
        "payload": sig.payload,
        "signature_id": sig.signature_id,
        "verification_url": sig.verification_url,
    }
    if sig.anchors is not None:
        envelope["anchors"] = sig.anchors
    return envelope


def _require_agt_verified(receipt: Any) -> None:
    if not _agt.verify_receipt(receipt):
        raise AgtSigningError(
            "agt_receipt_unverified: AGT's verify_receipt rejected the receipt, "
            "so Asqav does not sign it."
        )


class AsqavAgtSigner(AsqavAdapter):
    """Signs AGT governance receipts through Asqav; usable as AGT's ``external_authorizer``.

    Fails closed: a receipt AGT's ``verify_receipt`` rejects, an Asqav sign error,
    a refused Asqav decision or an Asqav receipt that does not bind the AGT receipt
    raises ``AgtSigningError``. ``links`` maps each AGT receipt id to its ``AsqavLink``.
    """

    def __init__(
        self,
        *,
        authorizer_key_hex: str | None = None,
        authorization_ttl_seconds: float = 300.0,
        api_key: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        ttl = authorization_ttl_seconds
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl):
            raise ValueError("authorization_ttl_seconds must be a finite number")
        if ttl <= 0:
            raise ValueError("authorization_ttl_seconds must be positive")
        self._authorizer_key_hex = authorizer_key_hex
        self._authorizer_public_key = (
            _ed25519_public_hex(authorizer_key_hex) if authorizer_key_hex is not None else None
        )
        self._ttl = float(ttl)
        self.links: dict[str, AsqavLink] = {}
        super().__init__(api_key=api_key, agent_name=agent_name, agent_id=agent_id)

    @property
    def authorizer_public_key(self) -> str | None:
        """Hex Ed25519 public key to list in AGT's ``trusted_authorizer_keys``."""
        return self._authorizer_public_key

    def countersign(self, receipt: Any) -> AsqavLink:
        """Record an AGT-verified receipt in an Asqav receipt and return the link."""
        _require_agt_verified(receipt)
        link = self._sign(receipt, COUNTERSIGN_ACTION, _link_context(receipt))
        self.links[link.agt_receipt_id] = link
        return link

    def __call__(self, receipt: Any) -> Any:
        """AGT ``external_authorizer`` hook: sign through Asqav, then authorize the receipt."""
        authorize = getattr(_agt, "authorize_receipt", None)
        if not callable(authorize):
            raise AgtSigningError(
                "agt_external_authorization_unavailable: the installed AGT has no "
                "authorize_receipt; use countersign() instead."
            )
        key_hex, public_key = self._authorizer_key_hex, self._authorizer_public_key
        if key_hex is None or public_key is None:
            raise AgtSigningError("authorizer_key_required: pass authorizer_key_hex")
        _require_agt_verified(receipt)
        if public_key == str(receipt.signer_public_key or "").lower():
            raise AgtSigningError("authorizer_key_reused: it must differ from the AGT signer key")
        nonce = uuid.uuid4().hex
        expires_at = time.time() + self._ttl
        fields = _authorization_fields(nonce, expires_at, public_key)
        link = self._sign(receipt, AUTHORIZE_ACTION, _link_context(receipt, fields))
        authorize(
            receipt,
            key_hex,
            authorizer_id=AUTHORIZER_PREFIX + link.signature_id,
            expires_at=expires_at,
            nonce=nonce,
        )
        self.links[link.agt_receipt_id] = link
        return receipt

    def _sign(self, receipt: Any, action_type: str, context: dict[str, str]) -> AsqavLink:
        expected_ref = _compute_action_ref(action_type, context)
        try:
            sig = self._agent.sign(action_type, context, compliance_mode=True)
        except Exception as exc:
            raise AgtSigningError(f"asqav_sign_failed: {exc}") from exc
        payload = sig.payload or {}
        decision = payload.get("decision", sig.decision)
        if decision not in _ACCEPTED_DECISIONS[action_type]:
            raise AgtSigningError(f"asqav_decision_refused: {decision} for {action_type}")
        signed_ref = payload.get("action_ref")
        if signed_ref != expected_ref:
            raise AgtSigningError(
                "asqav_receipt_unbound: the signed action_ref does not match this AGT receipt"
            )
        self._signatures.append(sig)
        return AsqavLink(
            agt_receipt_id=str(receipt.receipt_id),
            action_type=action_type,
            action_ref=expected_ref,
            signature_id=sig.signature_id,
            verification_url=sig.verification_url,
            asqav_receipt=_envelope(sig),
        )


#: Verifier axes that must pass for the signature to count (no anchors).
_LINK_AXES = ("issuer_key", "issuer_bind", "key_status", "signature", "key_binding")


def _signature_errors(asqav_receipt: dict[str, Any], jwks: dict[str, Any]) -> list[str]:
    from .verifier.verify_receipt import run_structured

    wire = ("signature", "payload", "anchors")
    result = run_structured({k: asqav_receipt[k] for k in wire if k in asqav_receipt}, jwks)
    axes = {axis["name"]: axis for axis in result.get("axes", [])}
    errors = []
    for name in sorted(set(_LINK_AXES) | set(axes)):
        axis = axes.get(name)
        outcome = axis["result"] if axis else "not reached"
        required = name in _LINK_AXES and outcome != "PASS"
        if required or (outcome == "FAIL" and name != "expiry"):
            note = axis["note"] if axis else "verifier stopped early"
            errors.append(f"Asqav receipt {name} axis {outcome}: {note}")
    return errors


def _link_errors(receipt: Any, asqav_receipt: dict[str, Any]) -> list[str]:
    payload = asqav_receipt["payload"]
    try:
        action_type, context = _expected_action(receipt)
        expected_ref = _compute_action_ref(action_type, context)
    except (AttributeError, TypeError, ValueError) as exc:
        return [f"AGT receipt cannot be bound: {exc}"]
    errors: list[str] = []
    if payload.get("action_ref") != expected_ref:
        errors.append("Asqav receipt does not bind this AGT receipt (action_ref mismatch).")
    decision = payload.get("decision")
    if decision not in _ACCEPTED_DECISIONS[action_type]:
        errors.append(f"Asqav decision {decision!r} does not support {action_type}.")
    # signature_id is outside the signed payload; action_ref binds.
    named = AUTHORIZER_PREFIX + str(asqav_receipt.get("signature_id"))
    if action_type == AUTHORIZE_ACTION and receipt.authorizer_id != named:
        errors.append("AGT authorizer_id does not name this Asqav receipt.")
    return errors


def _verifying_rows(asqav_receipt: dict[str, Any], jwks: dict[str, Any]) -> list[dict[str, Any]]:
    """JWKS rows whose key verifies the signature; the unsigned kid never picks the row."""
    from .verifier import verify_receipt as vr

    signature = asqav_receipt.get("signature")
    raw = signature.get("sig") if isinstance(signature, dict) else None
    keys = jwks.get("keys")
    if not isinstance(raw, str) or not isinstance(keys, list):
        return []
    msg, sig = vr.canonical_json(asqav_receipt["payload"]), vr._b64decode(raw)
    rows = []
    for row in keys:
        pk = _row_key(row)
        if pk is None:
            continue
        alg = row.get("alg") or signature.get("alg")
        if vr.verify_signature(pk, msg, sig, alg)[0] == "PASS":
            rows.append(row)
    return rows


def _row_key(row: Any) -> bytes | None:
    from .verifier import verify_receipt as vr

    if not isinstance(row, dict) or not isinstance(row.get("public_key"), str):
        return None
    try:
        return vr._b64decode(row["public_key"])
    except ValueError:
        return None


def _row_org_errors(row: dict[str, Any], claimed: Any, expected_org: str) -> list[str]:
    from .verifier import verify_receipt as vr

    errors = []
    key_org = vr.key_org_of(row)
    if key_org is not None and vr._org_key(key_org) != vr._org_key(expected_org):
        errors.append(f"Asqav signing key is published under org {key_org}, not {expected_org}.")
    result, note = vr.check_org_binding(vr.key_issuer_of(row), key_org, claimed)
    if result != "PASS":
        errors.append(f"Asqav signing key org binding {result}: {note}")
    return errors


def _thumbprint_errors(payload: dict[str, Any], row: dict[str, Any]) -> list[str]:
    from .verifier import verify_receipt as vr

    signed = payload.get("key_thumbprint")
    if signed is None:
        return []
    pk = _row_key(row)
    alg = row.get("alg")
    if pk is None or not isinstance(alg, str):
        return ["Asqav signing key cannot be thumbprinted, so key_thumbprint is unchecked."]
    if signed != vr.thumbprint_for_key(alg=alg, public_key=pk):
        return ["Asqav key_thumbprint does not name the key that verified the signature."]
    return []


def _org_errors(
    asqav_receipt: dict[str, Any], jwks: dict[str, Any], expected_org: str
) -> list[str]:
    """The pin holds only when the signed org, the verifying key's published org and the pin agree.

    The signed ``org_id`` names the org (``issuer_id`` carries a legal entity when one is
    set, so it is the claim only when ``org_id`` is absent). The org of the signing key
    comes from every JWKS row whose public key verifies the signature, never the kid.
    """
    payload = asqav_receipt["payload"]
    claimed = payload["org_id"] if "org_id" in payload else payload.get("issuer_id")
    errors: list[str] = []
    if claimed != expected_org:
        errors.append(f"Asqav receipt is not from org {expected_org} (got {claimed!r}).")
    rows = _verifying_rows(asqav_receipt, jwks)
    if not rows:
        return errors + ["No JWKS key verifies the Asqav signature, so its org is unknown."]
    for row in rows:
        errors += _row_org_errors(row, claimed, expected_org)
        errors += _thumbprint_errors(payload, row)
    return list(dict.fromkeys(errors))


def verify_asqav_link(
    receipt: Any,
    asqav_receipt: Any,
    *,
    jwks: dict[str, Any] | None = None,
    expected_org: str | None = None,
) -> list[str]:
    """Check that ``asqav_receipt`` is a signed Asqav witness of this AGT receipt.
    Returns errors; empty means verified against ``jwks`` and bound."""
    if not isinstance(asqav_receipt, dict) or not isinstance(asqav_receipt.get("payload"), dict):
        return ["Asqav receipt has no signed payload."]
    if not isinstance(jwks, dict):
        return ["Asqav signature not checked: pass the Asqav JWKS (asqav.fetch_jwks())."]
    errors = _signature_errors(asqav_receipt, jwks)
    errors += _link_errors(receipt, asqav_receipt)
    if expected_org is not None:
        errors += _org_errors(asqav_receipt, jwks, expected_org)
    return errors
