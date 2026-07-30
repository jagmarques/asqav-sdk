"""One org, two agent keys: the entry that answers is the entry that signed.

A cloud receipt names the org in ``signature.kid`` and signs with the agent's own
key. The published directory carries ``issuer_id`` on every key, so a kid holding
an org id matches every sibling key that org owns, and position in the list decides
which one answers. Position is not a bind: the sibling it lands on holds different
key bytes, a different revocation status and a different agent, so a receipt that
is cryptographically sound verifies against a key that never signed it.

Every directory here publishes TWO agent keys under one issuer, which is the shape
of any org running more than a single agent. A one-key directory cannot tell a
matcher that binds apart from one that returns whatever it reaches first, so it
proves nothing about which entry answered and why.
"""

from __future__ import annotations

import base64

import pytest

from asqav.verifier import verify_receipt as v
from asqav.verifier.oracle.adapters.asqav_native import AsqavNativeAdapter
from asqav.verifier.oracle.canonical import asqav_jcs
from asqav.verifier.oracle.core import verify

#: A canonical org id, the form the cloud puts in a receipt's kid.
ORG = "f94f66c0-c580-432d-a041-29374f7aee07"
OTHER_ORG = "0b6c2b1e-9f7a-4d3c-8a11-5c2e7d904f38"


def _ml_dsa_65():
    return pytest.importorskip("dilithium_py.ml_dsa").ML_DSA_65


def _key(
    kid: str,
    agent_id: str,
    issuer_id: str = ORG,
    public_key: bytes | str = "QUFBQQ==",
    status: str = "active",
) -> dict:
    pub = base64.b64encode(public_key).decode() if isinstance(public_key, bytes) else public_key
    return {
        "kid": kid,
        "agent_id": agent_id,
        "issuer_id": issuer_id,
        "alg": "ML-DSA-65",
        "public_key": pub,
        "status": status,
    }


def _payload(agent_id: str = "agt_two", issuer_id: str = ORG) -> dict:
    """A compliance payload naming its issuer and its agent inside the signed bytes."""
    return {
        "type": "protectmcp:decision",
        "issued_at": "2026-06-19T00:00:00.000000Z",
        "issuer_id": issuer_id,
        "agent_id": agent_id,
        "action_ref": "sha256:" + "8" * 64,
        "payload_digest": {"hash": "8" * 64, "size": 512},
        "policy_digest": "sha256:" + "3" * 64,
        "previousReceiptHash": "0" * 64,
        "decision": "allow",
    }


def _receipt(payload: dict, sig: str, kid: str = ORG) -> dict:
    return {
        "payload": payload,
        "signature": {"alg": "ML-DSA-65", "kid": kid, "sig": sig},
        "anchors": [],
    }


def _axes(doc: dict, jwks: dict) -> dict[str, str]:
    return {name: res for name, res, _note in AsqavNativeAdapter().extra_axes(doc, jwks)}


# --- the matcher picks the key the signed bytes name ---


def test_org_kid_resolves_the_agent_the_receipt_names() -> None:
    """Two siblings share the issuer, so the agent bind decides which one answers."""
    jwks = {"keys": [_key("key-agent-one", "agt_one"), _key("key-agent-two", "agt_two")]}
    entry = v.match_signing_key(jwks, ORG, "agt_two", ORG)
    assert entry is not None
    assert entry["kid"] == "key-agent-two"
    assert entry["agent_id"] == "agt_two"


def test_org_kid_resolves_the_first_agent_when_it_is_the_named_one() -> None:
    """Order carries no meaning: the same directory answers for either sibling."""
    jwks = {"keys": [_key("key-agent-one", "agt_one"), _key("key-agent-two", "agt_two")]}
    entry = v.match_signing_key(jwks, ORG, "agt_one", ORG)
    assert entry is not None
    assert entry["kid"] == "key-agent-one"


def test_exact_kid_outranks_the_agent_bind() -> None:
    """A kid naming one key names it exactly, so that key answers."""
    jwks = {"keys": [_key("key-agent-one", "agt_one"), _key("key-agent-two", "agt_two")]}
    entry = v.match_signing_key(jwks, "key-agent-one", "agt_two", ORG)
    assert entry is not None
    assert entry["kid"] == "key-agent-one"


def test_org_kid_answers_when_the_receipt_names_no_agent() -> None:
    """The bare-kid wire form keeps resolving through the published issuer id."""
    jwks = {"keys": [_key("key-agent-one", "agt_one")]}
    entry = v.match_signing_key(jwks, ORG, None, ORG)
    assert entry is not None
    assert entry["kid"] == "key-agent-one"


def test_org_kid_answers_for_an_agent_the_directory_omits() -> None:
    """An agent with no published key falls back to the issuer match, not to nothing."""
    jwks = {"keys": [_key("key-agent-one", "agt_one")]}
    entry = v.match_signing_key(jwks, ORG, "agt_absent", ORG)
    assert entry is not None
    assert entry["kid"] == "key-agent-one"


# --- adversarial: agent_id is attacker-controlled ---


def test_agent_bind_rejects_a_key_published_under_another_issuer() -> None:
    """agent_id alone never answers: the key's issuer must equal the claimed one."""
    jwks = {"keys": [_key("key-foreign", "agt_two", issuer_id=OTHER_ORG)]}
    assert v.match_signing_key(jwks, "kid-absent", "agt_two", ORG) is None


def test_matcher_survives_a_directory_holding_junk_entries() -> None:
    """A malformed sibling is a miss, never a crash, and never blocks the real key."""
    jwks = {"keys": [None, 7, "text", {"kid": "no-bytes", "issuer_id": ORG, "agent_id": "agt_two"},
                     _key("key-agent-two", "agt_two")]}
    entry = v.match_signing_key(jwks, ORG, "agt_two", ORG)
    assert entry is not None
    assert entry["kid"] == "key-agent-two"


# --- the axes read the entry that signed ---


def test_key_status_reads_the_signing_agent_not_its_sibling() -> None:
    """A revoked signer stays revoked even with an active sibling ahead of it."""
    jwks = {
        "keys": [
            _key("key-agent-one", "agt_one", status="active"),
            _key("key-agent-two", "agt_two", status="revoked"),
        ]
    }
    axes = _axes(_receipt(_payload("agt_two"), "AAAA"), jwks)
    assert axes["key_status"] == "FAIL"


def test_active_signer_is_not_reported_revoked_by_a_sibling() -> None:
    """The mirror case: a revoked sibling ahead of the signer changes no axis."""
    jwks = {
        "keys": [
            _key("key-agent-one", "agt_one", status="revoked"),
            _key("key-agent-two", "agt_two", status="active"),
        ]
    }
    axes = _axes(_receipt(_payload("agt_two"), "AAAA"), jwks)
    assert axes["key_status"] == "PASS"
    assert axes["issuer_bind"] == "PASS"


# --- end to end over a real ML-DSA-65 signature ---


def test_oracle_passes_a_receipt_signed_by_the_second_agent() -> None:
    """The whole point: an org with two agents can verify its own receipt."""
    ml = _ml_dsa_65()
    one_pk, _one_sk = ml.keygen()
    two_pk, two_sk = ml.keygen()
    payload = _payload("agt_two")
    sig = base64.b64encode(ml.sign(two_sk, asqav_jcs(payload))).decode()
    jwks = {
        "keys": [
            _key("key-agent-one", "agt_one", public_key=one_pk),
            _key("key-agent-two", "agt_two", public_key=two_pk),
        ]
    }
    res = verify(_receipt(payload, sig), [AsqavNativeAdapter()], key_provider=jwks)
    axes = {a.axis: a.result for a in res.axes}
    assert axes["signature"] == "PASS", axes
    assert axes["key_status"] == "PASS"
    assert axes["issuer_bind"] == "PASS"
    assert res.verdict == "PASS", axes


def test_oracle_rejects_a_signature_from_the_wrong_agent() -> None:
    """Naming a sibling in the payload never lends that sibling's key to a forgery."""
    ml = _ml_dsa_65()
    one_pk, _one_sk = ml.keygen()
    two_pk, _two_sk = ml.keygen()
    forger_pk, forger_sk = ml.keygen()
    payload = _payload("agt_two")
    sig = base64.b64encode(ml.sign(forger_sk, asqav_jcs(payload))).decode()
    jwks = {
        "keys": [
            _key("key-agent-one", "agt_one", public_key=one_pk),
            _key("key-agent-two", "agt_two", public_key=two_pk),
        ]
    }
    assert forger_pk not in (one_pk, two_pk)
    res = verify(_receipt(payload, sig), [AsqavNativeAdapter()], key_provider=jwks)
    assert res.verdict == "FAIL"
    assert {a.axis: a.result for a in res.axes}["signature"] == "FAIL"


def test_oracle_rejects_an_agent_key_from_another_org() -> None:
    """A valid signature under a foreign org's key never proves the claimed issuer."""
    ml = _ml_dsa_65()
    one_pk, _one_sk = ml.keygen()
    foreign_pk, foreign_sk = ml.keygen()
    payload = _payload("agt_two")
    sig = base64.b64encode(ml.sign(foreign_sk, asqav_jcs(payload))).decode()
    jwks = {
        "keys": [
            _key("key-agent-one", "agt_one", public_key=one_pk),
            _key("key-foreign", "agt_two", issuer_id=OTHER_ORG, public_key=foreign_pk),
        ]
    }
    res = verify(_receipt(payload, sig), [AsqavNativeAdapter()], key_provider=jwks)
    assert res.verdict == "FAIL"
    assert {a.axis: a.result for a in res.axes}["signature"] == "FAIL"
