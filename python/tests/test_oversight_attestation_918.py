import base64

import pytest
from dilithium_py.ml_dsa import ML_DSA_65

from asqav.canonicalize import canonicalize
from asqav.keys import ALGORITHM_ML_DSA_65
from asqav.verifier.oversight_attestation import (
    ASSERTED,
    METHODS,
    MISSING,
    NOT_APPLICABLE,
    SIGNATURE_VERIFIED,
    oversight_attestation,
)


@pytest.fixture
def env():
    return {"type": "protectmcp:lifecycle:oversight_ruling",
            "reviewer": {"principal": "reviewer_opaque_001",
                         "authority": "oversight-charter-2026-s4", "role": "oversight-judge"}}


@pytest.fixture
def signed(env, request):
    public, private = ML_DSA_65.keygen()
    attestation = {"method": getattr(request, "param", "sso"), "evidence":
                   "sha256:9b71d224bd62f3785d96d46ad4a5eae02a2dff72519673ca7a7bfbc2890caadae02a2dff72519673ca"}
    attestation["public_key"] = base64.urlsafe_b64encode(public).decode()
    signature = ML_DSA_65.sign(private, canonicalize(attestation))
    attestation["signature"] = {"alg": ALGORITHM_ML_DSA_65, "kid": "reviewer",
                                "sig": base64.urlsafe_b64encode(signature).decode()}
    env["reviewer"]["attestation"] = attestation
    return env, (ALGORITHM_ML_DSA_65, public)


def test_vector_048_no_attestation_is_missing(env):
    assert oversight_attestation(env) == MISSING


def test_vector_049_manual_assertion_is_asserted(env):
    env["reviewer"]["attestation"] = {"method": "manual_assertion"}
    assert oversight_attestation(env) == ASSERTED


def test_vector_050_unsigned_is_asserted(signed):
    env, key = signed
    del env["reviewer"]["attestation"]["signature"]
    assert oversight_attestation(env, reviewer_key=key) == ASSERTED


def test_vector_050_signed_under_reviewer_key_is_signature_verified(signed):
    env, key = signed
    assert oversight_attestation(env, reviewer_key=key) == SIGNATURE_VERIFIED
    env["reviewer"]["attestation"]["evidence"] += "changed"
    assert oversight_attestation(env, reviewer_key=key) == ASSERTED


# NEGCTL: byte 0 of the true-key signature flipped stays asserted
def test_negctl_tampered_signature_is_asserted(signed):
    env, key = signed
    signature = env["reviewer"]["attestation"]["signature"]
    raw = bytearray(base64.urlsafe_b64decode(signature["sig"]))
    raw[0] ^= 0xFF
    signature["sig"] = base64.urlsafe_b64encode(raw).decode()
    result = oversight_attestation(env, reviewer_key=key)
    assert result == ASSERTED and result != SIGNATURE_VERIFIED


def test_self_carried_key_is_asserted(signed):
    assert oversight_attestation(signed[0]) == ASSERTED


def test_other_signer_key_is_asserted(signed):
    env, key = signed
    assert oversight_attestation(env, reviewer_key=(key[0], ML_DSA_65.keygen()[0])) == ASSERTED


@pytest.mark.parametrize("signed", ["manual_assertion"], indirect=True)
def test_signed_manual_assertion_is_asserted(signed):
    env, key = signed
    assert oversight_attestation(env, reviewer_key=key) == ASSERTED


@pytest.mark.parametrize("attestation", [None, [], {}, {"method": []}, {"method": "unknown"}])
def test_invalid_attestation_is_missing(env, attestation):
    env["reviewer"]["attestation"] = attestation
    assert oversight_attestation(env) == MISSING


@pytest.mark.parametrize("field", ["alg", "kid", "sig"])
@pytest.mark.parametrize("value", [None, "", " ", [], {}])
def test_invalid_signature_is_asserted(signed, field, value):
    env, key = signed
    env["reviewer"]["attestation"]["signature"][field] = value
    assert oversight_attestation(env, reviewer_key=key) == ASSERTED


def test_other_type_is_not_applicable(env):
    assert oversight_attestation({**env, "type": "other"}) == NOT_APPLICABLE


@pytest.mark.parametrize("method", sorted(METHODS))
def test_method_without_key_is_asserted(env, method):
    env["reviewer"]["attestation"] = {"method": method}
    assert oversight_attestation(env) == ASSERTED
