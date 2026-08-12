"""One key resolution serves the signature axis and the key-status axes alike.

``signature.kid`` sits OUTSIDE the signed payload bytes, so a holder of a receipt can
rewrite it without disturbing the signature. When the signature axis resolves its key
through a fallback the gating axes do not share, a rewritten kid makes ``key_status``
and ``issuer_bind`` resolve nothing and emit no axis at all. The aggregate in
``oracle/core.py`` weighs the axes that exist, and it can block on an axis reporting
SKIPPED but not on one that was never emitted, so a receipt signed by a REVOKED key
reads PASS.

Every gate here pins the invariant that closes that: the entry the signature verifies
against is the entry every axis reads its published fields from.
"""

from __future__ import annotations

import base64
import copy

import pytest

from asqav.verifier import verify_receipt as v
from asqav.verifier.oracle import crypto
from asqav.verifier.oracle.adapters.asqav_native import AsqavNativeAdapter
from asqav.verifier.oracle.canonical import asqav_jcs
from asqav.verifier.oracle.core import verify

_ED25519_AVAILABLE = crypto.verify_ed25519(b"\x00" * 32, b"m", b"\x00" * 64)[0] != crypto.SKIPPED

requires_ed25519 = pytest.mark.skipif(
    not _ED25519_AVAILABLE, reason="cryptography not installed; Ed25519 verify SKIPs"
)

ISSUER = "acme-legal-entity"
AGENT = "agt_shared_resolution"
REAL_KID = "key-shared-resolution"
#: A kid the directory answers for nothing, which is what a rewritten kid looks like.
ABSENT_KID = "kid-that-resolves-nothing"


    # A compliance payload naming its issuer and agent inside the signed bytes.
def _payload() -> dict:
    return {
        "type": "protectmcp:decision",
        "issued_at": "2026-06-01T19:26:44.289388Z",
        "issuer_id": ISSUER,
        "agent_id": AGENT,
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
    }


    # A directory publishing one key, resolvable by kid AND by agent_id.
def _jwks(status: str = "revoked", public_key: str = "QUFBQQ==") -> dict:
    return {
        "keys": [
            {
                "kid": REAL_KID,
                "issuer_id": ISSUER,
                "agent_id": AGENT,
                "alg": "Ed25519",
                "public_key": public_key,
                "status": status,
            }
        ]
    }


def _receipt(kid: str, sig: str = "AAAA") -> dict:
    return {
        "payload": _payload(),
        "signature": {"alg": "Ed25519", "kid": kid, "sig": sig},
        "anchors": [],
    }


def _axes(doc: dict, jwks: dict) -> dict[str, str]:
    return {name: res for name, res, _note in AsqavNativeAdapter().extra_axes(doc, jwks)}


# --- the axes must exist whichever route resolved the key ---


    # A kid the directory does not answer for still yields both gating axes.
def test_absent_kid_still_emits_the_gating_axes() -> None:
    axes = _axes(_receipt(ABSENT_KID), _jwks())
    assert "key_status" in axes, "a rewritten kid must not make the axis vanish"
    assert "issuer_bind" in axes


    # The status read is the one published for the key the agent route resolves.
def test_absent_kid_reports_the_revoked_status_of_the_signing_key() -> None:
    assert _axes(_receipt(ABSENT_KID), _jwks(status="revoked"))["key_status"] == "FAIL"


    # Rewriting the kid changes no axis outcome, since one entry answers both.
def test_absent_kid_agrees_with_the_real_kid_on_every_axis() -> None:
    assert _axes(_receipt(ABSENT_KID), _jwks()) == _axes(_receipt(REAL_KID), _jwks())


    # The shared resolution does not turn an active key into a failure.
def test_active_key_passes_the_status_axis_through_the_agent_route() -> None:
    axes = _axes(_receipt(ABSENT_KID), _jwks(status="active"))
    assert axes["key_status"] == "PASS"
    assert axes["issuer_bind"] == "PASS"


#: Axes that only exist once a key resolves; expiry reads the signed bytes alone.
KEY_AXES = {"key_status", "issuer_bind"}


    # With nothing resolvable the signature axis carries the verdict on its own.
def test_no_key_at_all_emits_no_key_axis() -> None:
    axes = _axes(_receipt(ABSENT_KID), {"keys": []})
    assert KEY_AXES.isdisjoint(axes), axes


    # agent_id is attacker-controlled, so the issuer bind still gates the match.
def test_agent_route_needs_the_claimed_issuer_to_match() -> None:
    jwks = _jwks()
    jwks["keys"][0]["issuer_id"] = "some-other-entity"
    axes = _axes(_receipt(ABSENT_KID), jwks)
    assert KEY_AXES.isdisjoint(axes), axes


# --- the shared entry is the one the signature verifies against ---


    # Both surfaces answer for the absent kid, rather than one answering alone.
def test_resolve_key_and_extra_axes_resolve_the_same_entry() -> None:
    ad = AsqavNativeAdapter()
    doc, jwks = _receipt(ABSENT_KID), _jwks()
    pk, note = ad.resolve_key(doc, jwks)
    assert pk is not None, note
    emitted = {name for name, _, _ in ad.extra_axes(doc, jwks)}
    assert KEY_AXES <= emitted, emitted


    # kid wins when it resolves, so the agent route only fills a genuine gap.
def test_match_signing_key_prefers_the_kid_entry() -> None:
    jwks = {
        "keys": [
            {"kid": REAL_KID, "issuer_id": ISSUER, "public_key": "QUFBQQ==", "status": "active"},
            {
                "kid": "other",
                "issuer_id": ISSUER,
                "agent_id": AGENT,
                "public_key": "QkJCQg==",
                "status": "revoked",
            },
        ]
    }
    entry = v.match_signing_key(jwks, REAL_KID, AGENT, ISSUER)
    assert entry is not None
    assert entry["kid"] == REAL_KID


# --- end to end over a real signature ---


@requires_ed25519
def test_rewritten_kid_cannot_flip_a_revoked_receipt_to_pass() -> None:
    """The whole point: one unsigned byte field cannot buy a PASS.

    The payload bytes and the signature are identical across both documents, so the
    only difference an attacker introduces is the kid the directory resolves by.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sk = Ed25519PrivateKey.generate()
    pub = base64.b64encode(sk.public_key().public_bytes_raw()).decode()
    payload = _payload()
    sig = base64.b64encode(sk.sign(asqav_jcs(payload))).decode()

    honest = {
        "payload": payload,
        "signature": {"alg": "Ed25519", "kid": REAL_KID, "sig": sig},
        "anchors": [],
    }
    rewritten = copy.deepcopy(honest)
    rewritten["signature"]["kid"] = ABSENT_KID

    jwks = _jwks(status="revoked", public_key=pub)
    ad = [AsqavNativeAdapter()]

    honest_res = verify(honest, ad, key_provider=jwks)
    rewritten_res = verify(rewritten, ad, key_provider=jwks)

    # The signature really does check out against the published key in both cases.
    assert honest_res.axis("signature").result == "PASS"
    assert rewritten_res.axis("signature").result == "PASS"
    # And the revoked status carries the verdict in both cases.
    assert honest_res.verdict == "unverified"
    assert rewritten_res.verdict == "unverified"
    assert honest_res.failure_class == "invalid"
    assert rewritten_res.failure_class == "invalid"
    assert rewritten_res.axis("key_status") is not None
    assert rewritten_res.axis("key_status").result == "FAIL"
