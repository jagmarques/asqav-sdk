"""The standalone verifier picks the signing key by what was signed, not order.

A cloud receipt's kid is the org id, which every sibling key publishes as
issuer_id; the signed key_thumbprint and agent_id name the real key and are
read first. Every directory here publishes several keys under one issuer.
"""

from __future__ import annotations

import base64

import pytest

from asqav.verifier import verify_receipt as v
from asqav.verifier.oracle.adapters.asqav_native import AsqavNativeAdapter
from asqav.verifier.oracle.core import verify as oracle_verify
from asqav.verifier.verify_receipt import AMBIGUOUS_KID_NOTE

ORG = "f94f66c0-c580-432d-a041-29374f7aee07"
OTHER_ORG = "0b6c2b1e-9f7a-4d3c-8a11-5c2e7d904f38"


def _ml_dsa_65():
    return pytest.importorskip("dilithium_py.ml_dsa").ML_DSA_65


def _key(kid: str, agent_id: str, pk: bytes, *, issuer_id: str = ORG, status: str = "active",
         thumbprint: bool = True) -> dict:
    entry = {
        "kid": kid,
        "agent_id": agent_id,
        "issuer_id": issuer_id,
        "org_id": issuer_id,
        "alg": "ML-DSA-65",
        "kty": "AKP",
        "public_key": base64.b64encode(pk).decode(),
        "status": status,
        "revoked_at": "2026-07-11T14:57:49Z" if status == "revoked" else None,
    }
    if thumbprint:
        entry["key_thumbprint"] = v.thumbprint_for_key(alg="ML-DSA-65", public_key=pk)
    return entry


def _payload(agent_id: str = "agt_two", issuer_id: str = ORG, thumbprint: str | None = None) -> dict:
    p = {
        "type": "protectmcp:decision",
        "issued_at": "2026-06-19T00:00:00+00:00",
        "issuer_id": issuer_id,
        "agent_id": agent_id,
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
    }
    if thumbprint is not None:
        p["key_thumbprint"] = thumbprint
    return p


def _envelope(payload: dict, sig: bytes, alg: str = "ML-DSA-65", kid: str = ORG) -> dict:
    return {
        "payload": payload,
        "signature": {"alg": alg, "kid": kid, "sig": base64.b64encode(sig).decode()},
        "anchors": [],
    }


def _axes(report: dict) -> dict[str, tuple[str, str]]:
    return {a["name"]: (a["result"], a["note"]) for a in report["axes"]}


def test_unsupported_alg_resolves_the_named_agent_key_not_the_revoked_sibling() -> None:
    """Signature SKIPPED (alg outside the profile) never reaches a fallback:
    key_status reads the named agent's key, not the revoked sibling."""
    ml44 = pytest.importorskip("dilithium_py.ml_dsa").ML_DSA_44
    old_pk, _ = ml44.keygen()
    two_pk, two_sk = ml44.keygen()
    payload = _payload("agt_two")
    sig = ml44.sign(two_sk, v.canonical_json(payload))
    old = _key("k-old", "agt_one", old_pk, status="revoked", thumbprint=False)
    two = _key("k-two", "agt_two", two_pk, thumbprint=False)
    old["alg"] = two["alg"] = "ML-DSA-44"
    jwks = {"keys": [old, two]}
    report = v.run_structured(_envelope(payload, sig, alg="ML-DSA-44"), jwks, None)
    axes = _axes(report)
    assert axes["signature"][0] == "SKIPPED"
    assert axes["key_status"][0] == "PASS", axes["key_status"]
    assert axes["issuer_bind"][0] == "PASS"


def test_rotated_agent_resolves_by_signed_thumbprint_across_two_published_keys() -> None:
    """Two active keys, the stale one first: the thumbprint picks the signer."""
    ml = _ml_dsa_65()
    stale_pk, _ = ml.keygen()
    new_pk, new_sk = ml.keygen()
    thumb = v.thumbprint_for_key(alg="ML-DSA-65", public_key=new_pk)
    payload = _payload("agt_two", thumbprint=thumb)
    sig = ml.sign(new_sk, v.canonical_json(payload))
    jwks = {"keys": [_key("k-stale", "agt_two", stale_pk), _key("k-new", "agt_two", new_pk)]}
    report = v.run_structured(_envelope(payload, sig), jwks, None)
    axes = _axes(report)
    assert axes["signature"][0] == "PASS", axes["signature"]
    assert axes["key_binding"][0] == "PASS", axes["key_binding"]
    assert axes["key_status"][0] == "PASS"


def test_rotated_agent_without_a_thumbprint_verifies_against_the_key_that_signed() -> None:
    """With no published thumbprint, every candidate is tried for the signer."""
    ml = _ml_dsa_65()
    stale_pk, _ = ml.keygen()
    new_pk, new_sk = ml.keygen()
    payload = _payload("agt_two")
    sig = ml.sign(new_sk, v.canonical_json(payload))
    jwks = {"keys": [_key("k-stale", "agt_two", stale_pk), _key("k-new", "agt_two", new_pk)]}
    axes = _axes(v.run_structured(_envelope(payload, sig), jwks, None))
    assert axes["signature"][0] == "PASS", axes["signature"]
    assert "k-new" in axes["issuer_key"][1]


def test_thumbprint_naming_an_unused_key_is_reported_as_substitution() -> None:
    """Thumbprint of D, C signed: C verifies, D is bound, the binding breaks."""
    ml = _ml_dsa_65()
    c_pk, c_sk = ml.keygen()
    d_pk, _ = ml.keygen()
    payload = _payload("agt_two", thumbprint=v.thumbprint_for_key(alg="ML-DSA-65", public_key=d_pk))
    sig = ml.sign(c_sk, v.canonical_json(payload))
    jwks = {"keys": [_key("k-c", "agt_two", c_pk), _key("k-d", "agt_two", d_pk)]}
    report = v.run_structured(_envelope(payload, sig), jwks, None)
    axes = _axes(report)
    assert axes["signature"][0] == "PASS"
    assert axes["key_binding"][0] == "FAIL" and "key_substituted" in axes["key_binding"][1]
    assert report["verdict"] == v.VERDICT_UNVERIFIED
    assert report["failure_class"] == v.FAILURE_INVALID


def test_thumbprint_of_a_foreign_org_key_cannot_pass_the_issuer_bind() -> None:
    """An attacker's own thumbprint verifies the signature, fails the bind."""
    ml = _ml_dsa_65()
    attacker_pk, attacker_sk = ml.keygen()
    victim_pk, _ = ml.keygen()
    payload = _payload("agt_attacker", issuer_id=ORG,
                       thumbprint=v.thumbprint_for_key(alg="ML-DSA-65", public_key=attacker_pk))
    sig = ml.sign(attacker_sk, v.canonical_json(payload))
    jwks = {"keys": [_key("k-victim", "agt_victim", victim_pk),
                     _key("k-attacker", "agt_attacker", attacker_pk, issuer_id=OTHER_ORG)]}
    report = v.run_structured(_envelope(payload, sig), jwks, None)
    axes = _axes(report)
    assert axes["issuer_bind"][0] == "FAIL", axes["issuer_bind"]
    assert report["verdict"] == v.VERDICT_UNVERIFIED


def test_match_signing_key_order_is_thumbprint_kid_agent_then_issuer() -> None:
    ml = _ml_dsa_65()
    a_pk, _ = ml.keygen()
    b_pk, _ = ml.keygen()
    a, b = _key("k-a", "agt_a", a_pk), _key("k-b", "agt_b", b_pk)
    jwks = {"keys": [a, b]}
    thumb_b = b["key_thumbprint"]
    # The signed thumbprint outranks an exact kid that names another key.
    assert v.match_signing_key(jwks, "k-a", "agt_a", ORG, ORG, thumb_b) is b
    assert v.match_signing_key(jwks, "k-a", "agt_b", ORG, ORG) is a
    assert v.match_signing_key(jwks, ORG, "agt_b", ORG, ORG) is b
    # NEGCTL: restoring the first-match fallback returns sibling a, not None.
    assert v.match_signing_key(jwks, ORG, None, ORG, ORG) is None
    # A malformed thumbprint never matches and never crashes.
    assert v.match_signing_key(jwks, ORG, "agt_b", ORG, ORG, "sha256:nope") is b


def _axis(report: dict, name: str) -> dict:
    return {a["name"]: a for a in report["axes"]}[name]


def test_ambiguous_kid_fallback_is_unverifiable() -> None:
    """Two usable siblings share ORG, so the bare ORG kid picks neither."""
    ml = _ml_dsa_65()
    a_pk, a_sk = ml.keygen()
    b_pk, _ = ml.keygen()
    jwks = {"keys": [_key("k-a", "agt_a", a_pk), _key("k-b", "agt_b", b_pk)]}
    assert v.match_signing_key(jwks, ORG, None, ORG, ORG) is None
    payload = _payload("agt_two")
    sig = ml.sign(a_sk, v.canonical_json(payload))
    report = v.run_structured(_envelope(payload, sig), jwks, None)
    issuer_key = _axis(report, "issuer_key")
    assert issuer_key["failure_class"] == v.FAILURE_UNVERIFIABLE, issuer_key
    assert AMBIGUOUS_KID_NOTE in issuer_key["note"], issuer_key
    assert report["verdict"] == v.VERDICT_UNVERIFIED
    assert report["failure_class"] == v.FAILURE_UNVERIFIABLE


def test_exact_kid_among_siblings_resolves() -> None:
    """An exact kid picks its sibling where the bare issuer kid picks none."""
    ml = _ml_dsa_65()
    a_pk, _ = ml.keygen()
    b_pk, b_sk = ml.keygen()
    a, b = _key("k-a", "agt_a", a_pk), _key("k-b", "agt_b", b_pk)
    jwks = {"keys": [a, b]}
    assert v.match_signing_key(jwks, ORG, None, ORG, ORG) is None
    assert v.match_signing_key(jwks, "k-b", None, ORG, ORG) is b
    # The kid doubles as the first sibling's issuer_id and the second's kid.
    mixed = {"keys": [_key("k-a", "agt_a", a_pk),
                      _key(ORG, "agt_b", b_pk, issuer_id=OTHER_ORG)]}
    assert v.match_signing_key(mixed, ORG, None, ORG, ORG) is mixed["keys"][1]
    payload = _payload("agt_two")
    sig = ml.sign(b_sk, v.canonical_json(payload))
    report = v.run_structured(_envelope(payload, sig, kid="k-b"), jwks, None)
    assert _axis(report, "issuer_key")["result"] == "PASS"
    assert "k-b" in _axis(report, "issuer_key")["note"]


def test_thumbprint_among_siblings_resolves() -> None:
    """The signed thumbprint picks its sibling when the bare kid is shared."""
    ml = _ml_dsa_65()
    a_pk, _ = ml.keygen()
    b_pk, _ = ml.keygen()
    a, b = _key("k-a", "agt_a", a_pk), _key("k-b", "agt_b", b_pk)
    jwks = {"keys": [a, b]}
    assert v.match_signing_key(jwks, ORG, None, ORG, ORG) is None
    thumb = v.thumbprint_for_key(alg="ML-DSA-65", public_key=b_pk)
    assert v.match_signing_key(jwks, ORG, None, ORG, ORG, thumb) is b


def test_single_fallback_match_still_resolves() -> None:
    """One match resolves; adding a second usable sibling makes it ambiguous."""
    ml = _ml_dsa_65()
    a_pk, b_pk = ml.keygen()[0], ml.keygen()[0]
    a = _key("k-a", "agt_a", a_pk)
    assert v.match_signing_key({"keys": [a]}, ORG, None, ORG, ORG) is a
    two = {"keys": [a, _key("k-b", "agt_b", b_pk)]}
    assert v.match_signing_key(two, ORG, None, ORG, ORG) is None


def test_unusable_sibling_does_not_make_the_kid_ambiguous() -> None:
    """Only accepted entries count; a second usable sibling is ambiguous."""
    ml = _ml_dsa_65()
    a_pk, b_pk = ml.keygen()[0], ml.keygen()[0]
    a = _key("k-a", "agt_a", a_pk)
    unusable = {"kid": "k-junk", "agent_id": "agt_junk", "issuer_id": ORG}
    assert v.match_signing_key({"keys": [unusable, a]}, ORG, None, ORG, ORG) is a
    usable = {"keys": [unusable, a, _key("k-b", "agt_b", b_pk)]}
    assert v.match_signing_key(usable, ORG, None, ORG, ORG) is None


def test_public_resolvers_pick_no_sibling_on_a_shared_kid() -> None:
    """resolve_* read one entry: a kid two usable siblings share yields none."""
    ml = _ml_dsa_65()
    a_pk, b_pk = ml.keygen()[0], ml.keygen()[0]
    a, b = _key("k-a", "agt_a", a_pk), _key("k-b", "agt_b", b_pk)
    assert v.resolve_key({"keys": [a, b]}, ORG) == (None, None, None)
    assert v.resolve_key_issuer({"keys": [a, b]}, ORG) is None
    assert v.resolve_revoked_at({"keys": [a, b]}, ORG) is None
    assert v.resolve_key_org({"keys": [a, b]}, ORG) is None
    assert v.resolve_key({"keys": [a, b]}, "k-b")[0] == base64.b64decode(b["public_key"])
    assert v.resolve_key_issuer({"keys": [a]}, ORG) == ORG
    # The kid is one sibling's exact kid and the other's issuer_id: exact wins.
    mixed = {"keys": [a, _key(ORG, "agt_b", b_pk, issuer_id=OTHER_ORG)]}
    assert v.resolve_key(mixed, ORG)[0] == base64.b64decode(mixed["keys"][1]["public_key"])
    assert v.resolve_key_issuer(mixed, ORG) == OTHER_ORG


def test_oracle_adapter_agrees_with_the_standalone_on_the_rotated_agent() -> None:
    ml = _ml_dsa_65()
    stale_pk, _ = ml.keygen()
    new_pk, new_sk = ml.keygen()
    thumb = v.thumbprint_for_key(alg="ML-DSA-65", public_key=new_pk)
    payload = _payload("agt_two", thumbprint=thumb)
    sig = ml.sign(new_sk, v.canonical_json(payload))
    jwks = {"keys": [_key("k-stale", "agt_two", stale_pk), _key("k-new", "agt_two", new_pk)]}
    res = oracle_verify(_envelope(payload, sig), [AsqavNativeAdapter()], jwks)
    by_name = {a.axis: a.result for a in res.axes}
    assert by_name["signature"] == "PASS", res.axes

