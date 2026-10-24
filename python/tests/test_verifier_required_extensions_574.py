"""Policy-required optional extensions, Python half (criterion 574-D3).

A selected policy can require an optional extension the default evaluation
never checks. The standalone verifier then evaluates it per-axis: a carried
and well-formed member passes, a malformed one fails invalid, and an absent
or unsupported one skips unverifiable, so the receipt never reads as
verified. Without the --require-extension input the evaluation is unchanged.
"""

from __future__ import annotations

import json
import sys

import pytest

from asqav.verifier import verify_receipt as vr
from tests.test_verify_receipt import (
    _envelope,
    _jwks_key,
    _ml_dsa_65,
    _trusted_envelope,
    _valid_payload,
)


def _beacon(**overrides):
    beacon = {
        "source": "drand",
        "chain": "a" * 64,
        "round": 7,
        "signature": "b" * 96,
        "observed_at": "2026-10-18T00:00:00Z",
    }
    beacon.update(overrides)
    return beacon


def _signed(issuer_id="org-legit", agent_id="agt_one", **payload_overrides):
    ml = _ml_dsa_65()
    pk, sk = ml.keygen()
    payload = _valid_payload(issuer_id, agent_id)
    payload.update(payload_overrides)
    sig = ml.sign(sk, vr.canonical_json(payload))
    jwks = {"keys": [_jwks_key("k1", agent_id, issuer_id, pk)]}
    return payload, sig, jwks


def test_supported_set_names_beacon_ref_only() -> None:
    assert vr.SUPPORTED_REQUIRED_EXTENSIONS == frozenset({"beacon_ref"})


def test_binary_key_material_never_strips() -> None:
    """Raw key bytes pass through untouched, even with whitespace edges.

    Regression pin for an intermittent CLI failure: tsa_key_material stripped
    the file bytes, so a raw binary TSA key whose edge bytes are ASCII
    whitespace lost a byte and every signature it should verify failed.
    """
    raw = b" " + bytes(i % 256 for i in range(1950)) + b"\n"
    assert len(raw) == 1952
    assert vr.tsa_key_material(raw) == raw
    assert vr.tsa_key_material(raw + b"\n") == raw + b"\n"


@pytest.mark.parametrize(
    "mutation",
    [
        {"source": "nist"},
        {"chain": "xyz"},
        {"chain": "a" * 63},
        {"round": 0},
        {"round": "7"},
        {"round": True},
        {"signature": "b" * 95},
        {"signature": "B" * 96},
        {"observed_at": None},
    ],
)
def test_malformed_beacon_shapes_fail(mutation) -> None:
    res, _note = vr.check_required_extensions({"beacon_ref": _beacon(**mutation)}, ["beacon_ref"])
    assert res == "FAIL"


def test_well_formed_beacon_passes() -> None:
    res, note = vr.check_required_extensions({"beacon_ref": _beacon()}, ["beacon_ref"])
    assert res == "PASS"
    assert "well-formed" in note


def test_absent_beacon_skips_unverifiable() -> None:
    res, note = vr.check_required_extensions({"mode": "payload"}, ["beacon_ref"])
    assert res == "SKIPPED"
    assert "absent" in note


def test_unsupported_required_extension_skips() -> None:
    res, note = vr.check_required_extensions({"beacon_ref": _beacon()}, ["time_capsule"])
    assert res == "SKIPPED"
    assert "not supported" in note


def test_empty_required_list_passes_without_axis() -> None:
    payload, sig, jwks = _signed(beacon_ref=_beacon())
    out = vr.run_structured(_envelope(payload, sig), jwks, None, required_extensions=[])
    names = [a["name"] for a in out["axes"]]
    assert "required_extensions" not in names


def test_default_evaluation_appends_no_axis() -> None:
    payload, sig, jwks = _signed(beacon_ref=_beacon())
    out = vr.run_structured(_envelope(payload, sig), jwks, None)
    assert "required_extensions" not in [a["name"] for a in out["axes"]]


def test_required_valid_beacon_verifies() -> None:
    payload, sig, jwks = _signed(beacon_ref=_beacon())
    env, tsa_pk = _trusted_envelope(payload, sig, kid="k1")
    out = vr.run_structured(
        env, jwks, None, trusted_tsa_keys=[tsa_pk], required_extensions=["beacon_ref"]
    )
    axis = next(a for a in out["axes"] if a["name"] == "required_extensions")
    assert (axis["result"], axis["failure_class"]) == ("PASS", None), axis
    assert out["verdict"] == "verified", [a for a in out["axes"] if a["result"] != "PASS"]
    assert out["failure_class"] is None


def test_required_malformed_beacon_fails_invalid() -> None:
    payload, sig, jwks = _signed(beacon_ref=_beacon(round=0))
    env, tsa_pk = _trusted_envelope(payload, sig, kid="k1")
    out = vr.run_structured(
        env, jwks, None, trusted_tsa_keys=[tsa_pk], required_extensions=["beacon_ref"]
    )
    axis = next(a for a in out["axes"] if a["name"] == "required_extensions")
    assert (axis["result"], axis["failure_class"]) == ("FAIL", "invalid"), axis
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "invalid"


def test_required_absent_beacon_blocks_unverifiable() -> None:
    payload, sig, jwks = _signed()
    assert "beacon_ref" not in payload
    env, tsa_pk = _trusted_envelope(payload, sig, kid="k1")
    out = vr.run_structured(
        env, jwks, None, trusted_tsa_keys=[tsa_pk], required_extensions=["beacon_ref"]
    )
    axis = next(a for a in out["axes"] if a["name"] == "required_extensions")
    assert (axis["result"], axis["failure_class"]) == ("SKIPPED", "unverifiable"), axis
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "unverifiable"


def test_required_unsupported_extension_blocks_unverifiable() -> None:
    payload, sig, jwks = _signed(beacon_ref=_beacon())
    env, tsa_pk = _trusted_envelope(payload, sig, kid="k1")
    out = vr.run_structured(
        env, jwks, None, trusted_tsa_keys=[tsa_pk], required_extensions=["time_capsule"]
    )
    axis = next(a for a in out["axes"] if a["name"] == "required_extensions")
    assert (axis["result"], axis["failure_class"]) == ("SKIPPED", "unverifiable"), axis
    assert out["verdict"] == "unverified"
    assert out["failure_class"] == "unverifiable"


def test_flag_is_what_evaluates_the_beacon() -> None:
    """The same malformed receipt verifies without the flag and fails with it."""
    payload, sig, jwks = _signed(beacon_ref=_beacon(round=0))
    env, tsa_pk = _trusted_envelope(payload, sig, kid="k1")
    plain = vr.run_structured(env, jwks, None, trusted_tsa_keys=[tsa_pk])
    assert plain["verdict"] == "verified", [a for a in plain["axes"] if a["result"] != "PASS"]
    required = vr.run_structured(
        env, jwks, None, trusted_tsa_keys=[tsa_pk], required_extensions=["beacon_ref"]
    )
    assert required["verdict"] == "unverified"
    assert required["failure_class"] == "invalid"


def test_print_path_exit_codes(tmp_path, monkeypatch, capsys) -> None:
    def _files(payload):
        ml = _ml_dsa_65()
        pk, sk = ml.keygen()
        payload = dict(payload)
        sig = ml.sign(sk, vr.canonical_json(payload))
        env, tsa_pk = _trusted_envelope(payload, sig, kid="k1")
        rp = tmp_path / "receipt.json"
        jp = tmp_path / "jwks.json"
        tp = tmp_path / "tsa.key"
        rp.write_text(json.dumps(env))
        key = _jwks_key("k1", payload["agent_id"], payload["issuer_id"], pk)
        jp.write_text(json.dumps({"keys": [key]}))
        tp.write_bytes(tsa_pk)
        return str(rp), str(jp), str(tp)

    base = _valid_payload("org-legit", "agt_one")
    rp, jp, tp = _files(dict(base, beacon_ref=_beacon()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_receipt.py", "--receipt", rp, "--jwks", jp, "--tsa-key", tp,
            "--offline", "--require-extension", "beacon_ref",
        ],
    )
    assert vr.main() == 0
    out = capsys.readouterr().out
    assert "required_extensions" in out

    rp, jp, tp = _files(dict(base, beacon_ref=_beacon(round=0)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_receipt.py", "--receipt", rp, "--jwks", jp, "--tsa-key", tp,
            "--offline", "--require-extension", "beacon_ref",
        ],
    )
    assert vr.main() == 1

    rp, jp, tp = _files(dict(base))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_receipt.py", "--receipt", rp, "--jwks", jp, "--tsa-key", tp,
            "--offline", "--require-extension", "beacon_ref",
        ],
    )
    assert vr.main() == 2
