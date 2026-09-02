"""Shape predicates the two languages must answer identically.

Python half of typescript/tests/verifier-cross-language-parity.test.ts: the same inputs, the
same answers, so a receipt earns the same note whichever engine a customer runs.
"""

from __future__ import annotations

from asqav.verifier import verify_receipt as vr


# === thumbprint shape (448c) ===


def test_a_trailing_newline_is_not_a_well_formed_thumbprint() -> None:
    good = "sha256:" + "a" * 64
    assert vr.is_well_formed(good) is True
    assert vr.is_well_formed(good + "\n") is False
    assert vr.is_well_formed(good + "\n ") is False
    assert vr.is_well_formed("\n" + good) is False


# === an empty agent id is a missing agent id (448c) ===


def test_an_empty_agent_id_falls_through_to_the_envelope() -> None:
    payload = {"agent_id": "", "org_id": "", "issuer_id": "org-A"}
    envelope = {"agent_id": "agt-A", "org_id": "org-A"}
    row = {
        "kid": "k-A",
        "agent_id": "agt-A",
        "issuer_id": "org-A",
        "org_id": "org-A",
        "alg": "ML-DSA-65",
        "kty": "AKP",
        "public_key": "AAEC",
        "status": "active",
        "revoked_at": None,
    }
    entry = vr._signing_key_entry({"keys": [row]}, "", payload, envelope)
    assert entry is not None and entry["kid"] == "k-A"


# === a lone surrogate is not a supplementary character (448c) ===


def test_a_lone_surrogate_member_name_is_not_supplementary() -> None:
    assert vr.has_supplementary_member_name({"\ud800": 1}) is False
    assert vr.has_supplementary_member_name({"\udc00": 1}) is False
    assert vr.has_supplementary_member_name({"a\ud800b": 1}) is False
    assert vr.has_supplementary_member_name({"\U0001f600": 1}) is True
    assert vr.has_supplementary_member_name({"￿": 1}) is False
