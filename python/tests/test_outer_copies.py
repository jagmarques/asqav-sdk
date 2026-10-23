"""Unsigned outer copies of signed fields (criterion 742)."""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import asqav
from asqav.verifier import verify_receipt as vr

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
VECTORS = ROOT / "verifier" / "conformance-vectors"
CASES = json.loads((ROOT / "verifier" / "outer-copy-cases.json").read_text())["cases"]

requires_mldsa = pytest.mark.skipif(
    importlib.util.find_spec("dilithium_py") is None,
    reason="ML-DSA-65 verification needs dilithium-py",
)


def _prod() -> tuple[dict, dict]:
    return (
        json.loads((FIXTURES / "prod_audit_pack_receipt_2026_09_02.json").read_text()),
        json.loads((FIXTURES / "prod_audit_pack_jwks_2026_09_02.json").read_text()),
    )


def _vector(name: str) -> tuple[dict, dict]:
    base = VECTORS / name
    return (
        json.loads((base / "receipt.json").read_text()),
        json.loads((base / "jwks.json").read_text()),
    )


def _axes(result: dict) -> dict:
    return {a["name"]: a for a in result["axes"]}


def test_case_table_is_populated() -> None:
    assert len(CASES) >= 100


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_shared_case_table(case: dict) -> None:
    key = None if case["key_b64"] is None else base64.b64decode(case["key_b64"])
    message = base64.b64decode(case["message_b64"])
    result, note = vr.check_outer_copies(case["raw"], message, key, case.get("key_kid"))
    assert (result, note) == (case["expect"]["result"], case["expect"]["note"])


def test_the_table_covers_both_results() -> None:
    assert {c["expect"]["result"] for c in CASES} == {"PASS", "FAIL"}


# ---- edit the unsigned copies of a real prod export ----

@pytest.mark.parametrize(
    ("member", "value"),
    [
        ("signed_at", "2026-03-03T00:00:00"),
        ("agent_id", "agt_someoneElse000000"),
        ("action_id", "act_forgedForgedForgedFo"),
        ("algorithm", "ML-DSA-87"),
        ("compliance_mode", False),
    ],
)
@requires_mldsa
def test_edited_copy_of_a_prod_export_fails_every_verifier(member: str, value: object) -> None:
    receipt, jwks = _prod()
    receipt[member] = value
    standalone = vr.run_structured(receipt, jwks)
    offline = asqav.verify_receipt_offline(receipt, jwks)
    for result in (standalone, offline):
        outer = _axes(result)["outer_copies"]
        assert outer["result"] == "FAIL", outer
        assert outer["failure_class"] == "invalid"
        assert result["verdict"] == "unverified"
        assert result["failure_class"] == "invalid"
    # The signed bytes are untouched, so the signature is not what refuses it
    assert _axes(standalone)["signature"]["result"] == "PASS"
    assert _axes(offline)["signature"]["result"] == "PASS"


@requires_mldsa
def test_the_untouched_prod_export_passes_the_axis_and_lists_what_is_unattested() -> None:
    receipt, jwks = _prod()
    outer = _axes(vr.run_structured(receipt, jwks))["outer_copies"]
    assert outer["result"] == "PASS"
    assert "agent_id, action_id, signed_at" in outer["note"]
    # Hash mode signs no action_type, so that copy has nothing to check against
    assert "not signed, do not rely on: action_type, signature_id" in outer["note"]
    assert _axes(asqav.verify_receipt_offline(receipt, jwks))["outer_copies"]["result"] == "PASS"


def test_an_unattested_copy_is_not_a_mismatch() -> None:
    receipt, jwks = _prod()
    receipt["action_type"] = "anything:at:all"
    receipt["signature_id"] = "sig_not_the_real_one"
    assert _axes(vr.run_structured(receipt, jwks))["outer_copies"]["result"] == "PASS"


def test_removing_a_copy_is_not_a_mismatch() -> None:
    receipt, jwks = _prod()
    for member in ("signed_at", "agent_id", "message_b64", "public_key_b64"):
        del receipt[member]
    assert _axes(vr.run_structured(receipt, jwks))["outer_copies"]["result"] == "PASS"


def test_a_three_key_envelope_is_unaffected() -> None:
    receipt, jwks = _vector("asqav-01-genesis-permit")
    result = asqav.verify_receipt_offline(receipt, jwks)
    assert result["verdict"] == "verified"
    assert _axes(result)["outer_copies"]["result"] == "PASS"


# ---- the standalone command line --------------------------------------------

def _cli(tmp_path: Path, receipt: dict, jwks: dict) -> subprocess.CompletedProcess:
    (tmp_path / "r.json").write_text(json.dumps(receipt))
    (tmp_path / "j.json").write_text(json.dumps(jwks))
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "asqav.verifier.verify_receipt",
         "--receipt", str(tmp_path / "r.json"), "--jwks", str(tmp_path / "j.json"), "--offline"],
        capture_output=True, text=True,
    )


def test_cli_exits_one_and_prints_the_failing_axis(tmp_path: Path) -> None:
    receipt, jwks = _prod()
    receipt["signed_at"] = "2026-03-03T00:00:00"
    run = _cli(tmp_path, receipt, jwks)
    assert run.returncode == 1, run.stdout
    assert "[FAIL] outer_copies" in run.stdout
    assert "signed_at (signed: issued_at)" in run.stdout
    assert "failure_class: invalid" in run.stdout


def test_cli_run_reports_the_axis_on_a_clean_export(tmp_path: Path) -> None:
    receipt, jwks = _prod()
    run = _cli(tmp_path, receipt, jwks)
    assert "[  ok] outer_copies" in run.stdout


# ---- conformance vectors (verifier/conformance-vectors/asqav-43..51) --------

EDITED = [
    "asqav-44-outer-signed-at-edited",
    "asqav-45-outer-agent-id-edited",
    "asqav-46-outer-action-type-edited",
    "asqav-47-outer-message-copy-edited",
    "asqav-48-outer-public-key-edited",
]


@requires_mldsa
def test_the_control_vector_verifies() -> None:
    receipt, jwks = _vector("asqav-43-outer-copies-agree")
    result = asqav.verify_receipt_offline(receipt, jwks)
    assert result["verdict"] == "verified", result["axes"]
    assert _axes(result)["outer_copies"]["result"] == "PASS"


@pytest.mark.parametrize("name", EDITED)
@requires_mldsa
def test_every_edited_vector_is_refused_by_both_python_verifiers(name: str) -> None:
    receipt, jwks = _vector(name)
    for result in (asqav.verify_receipt_offline(receipt, jwks), vr.run_structured(receipt, jwks)):
        assert result["verdict"] == "unverified", name
        assert result["failure_class"] == "invalid", name
        axes = _axes(result)
        assert axes["outer_copies"]["result"] == "FAIL"
        assert axes["signature"]["result"] == "PASS"


@pytest.mark.parametrize("name", EDITED)
def test_an_edited_vector_differs_from_the_control_only_in_one_copy(name: str) -> None:
    control, _ = _vector("asqav-43-outer-copies-agree")
    edited, _ = _vector(name)
    assert [k for k in control if control[k] != edited[k]] != []
    assert control["payload"] == edited["payload"]
    assert control["signature"] == edited["signature"]
    assert len([k for k in control if control[k] != edited[k]]) == 1


def test_the_observation_time_rule_does_not_open_the_signed_at_check() -> None:
    receipt, jwks = _vector("asqav-44-outer-signed-at-edited")
    receipt["payload"] = copy.deepcopy(receipt["payload"])
    assert receipt["payload"]["type"] != "protectmcp:observation"
    assert _axes(vr.run_structured(receipt, jwks))["outer_copies"]["result"] == "FAIL"


# ---- the other two spellings of the signature (asqav-49..51) -----------------

SHAPE_EDITED = [
    "asqav-50-outer-flat-signature-agent-id-edited",
    "asqav-51-outer-envelope-signature-action-type-edited",
]


@requires_mldsa
def test_the_flat_signature_control_verifies() -> None:
    receipt, jwks = _vector("asqav-49-outer-flat-signature-agree")
    assert isinstance(receipt["signature"], str)
    result = asqav.verify_receipt_offline(receipt, jwks)
    assert result["verdict"] == "verified", result["axes"]
    assert _axes(result)["outer_copies"]["result"] == "PASS"


@pytest.mark.parametrize("name", SHAPE_EDITED)
@requires_mldsa
def test_an_edited_copy_beside_a_rebuilt_signature_is_refused(name: str) -> None:
    receipt, jwks = _vector(name)
    assert not isinstance(receipt.get("signature"), dict)
    for result in (asqav.verify_receipt_offline(receipt, jwks), vr.run_structured(receipt, jwks)):
        axes = _axes(result)
        assert axes["signature"]["result"] == "PASS", name
        assert axes["outer_copies"]["result"] == "FAIL", name
        assert (result["verdict"], result["failure_class"]) == ("unverified", "invalid"), name


# ---- members no signature covers, and the signature that did not verify ----

def _both(receipt: dict, jwks: dict) -> list:
    return [asqav.verify_receipt_offline(receipt, jwks), vr.run_structured(receipt, jwks)]


@pytest.mark.parametrize(
    "member",
    ["issued_at", "type", "issuer_id", "org_id", "decision", "v"],
)
@requires_mldsa
def test_a_top_level_member_under_a_signed_name_must_equal_it(member: str) -> None:
    receipt, jwks = _vector("asqav-43-outer-copies-agree")
    receipt[member] = "2030-01-01T00:00:00Z" if member == "issued_at" else "forged"
    for result in _both(receipt, jwks):
        assert _axes(result)["outer_copies"]["result"] == "FAIL", member
        assert result["verdict"] == "unverified"


@requires_mldsa
def test_a_nested_signature_member_under_a_signed_name_must_equal_it() -> None:
    receipt, jwks = _vector("asqav-43-outer-copies-agree")
    receipt["signature"]["agent_id"] = "agt_someoneElse000000"
    for result in _both(receipt, jwks):
        outer = _axes(result)["outer_copies"]
        assert outer["result"] == "FAIL" and "signature.agent_id" in outer["note"]


@requires_mldsa
def test_unsigned_members_are_listed_on_a_pass() -> None:
    receipt, jwks = _vector("asqav-43-outer-copies-agree")
    receipt.update(verified=True, signature_valid=True, kid="k", key_id="k", agent_name="Trusted")
    receipt["signature"]["signed_at"] = "2030-01-01T00:00:00Z"
    for result in _both(receipt, jwks):
        outer = _axes(result)["outer_copies"]
        assert outer["result"] == "PASS"
        assert (
            "not signed, do not rely on: agent_name, key_id, kid, signature_valid, verified, "
            "signature.signed_at" in outer["note"]
        )


@requires_mldsa
def test_the_withheld_flag_does_not_switch_off_the_digest_check() -> None:
    receipt, jwks = _vector("asqav-43-outer-copies-agree")
    receipt.update(message_sha256="0" * 64, signed_bytes_withheld=True)
    for result in _both(receipt, jwks):
        outer = _axes(result)["outer_copies"]
        assert _axes(result)["signature"]["result"] == "PASS"
        assert outer["result"] == "FAIL" and "message_sha256" in outer["note"]


@requires_mldsa
def test_the_withheld_flag_is_listed_beside_a_true_digest() -> None:
    import hashlib

    receipt, jwks = _vector("asqav-43-outer-copies-agree")
    receipt["message_sha256"] = hashlib.sha256(base64.b64decode(receipt["message_b64"])).hexdigest()
    receipt["signed_bytes_withheld"] = True
    for result in _both(receipt, jwks):
        outer = _axes(result)["outer_copies"]
        assert outer["result"] == "PASS" and "signed_bytes_withheld" in outer["note"]


@requires_mldsa
def test_a_key_copy_is_not_blamed_when_the_signature_did_not_verify_under_it() -> None:
    receipt, jwks = _vector("asqav-43-outer-copies-agree")
    rotated = copy.deepcopy(jwks)
    key = bytearray(base64.b64decode(rotated["keys"][0]["public_key"]))
    key[-1] ^= 0x01
    rotated["keys"][0]["public_key"] = base64.b64encode(bytes(key)).decode()
    result = asqav.verify_receipt_offline(receipt, rotated)
    axes = _axes(result)
    assert axes["signature"]["result"] != "PASS"
    assert axes["outer_copies"]["result"] == "PASS"
    assert "public_key_b64" in axes["outer_copies"]["note"]
    assert result["verdict"] == "unverified"


@requires_mldsa
def test_a_digest_copy_is_not_blamed_when_the_signature_did_not_verify() -> None:
    receipt, jwks = _vector("asqav-43-outer-copies-agree")
    receipt["message_sha256"] = "0" * 64
    receipt["payload"]["agent_id"] = receipt["agent_id"] = "agt_someoneElse000000"
    axes = _axes(asqav.verify_receipt_offline(receipt, jwks))
    assert axes["signature"]["result"] == "FAIL"
    assert axes["outer_copies"]["result"] == "FAIL"
    assert "message_sha256 (signed" not in axes["outer_copies"]["note"]


# ---- a flat hash-mode receipt signs its own fields (asqav-52) ---------------

@requires_mldsa
def test_a_flat_hash_receipt_lists_the_members_it_does_not_sign() -> None:
    receipt, jwks = _vector("asqav-52-outer-flat-hash-unsigned-members")
    for result in _both(receipt, jwks)[:1]:
        outer = _axes(result)["outer_copies"]
        assert result["verdict"] == "verified"
        assert outer["note"] == "not signed, do not rely on: action_type, agent_name, decision, signed_at, verified"


def test_compliance_mode_null_fails_beside_a_payload() -> None:
    receipt, _ = _vector("asqav-43-outer-copies-agree")
    receipt["compliance_mode"] = None
    result, note = vr.check_outer_copies(receipt, b"x", None)
    assert result == "FAIL" and "compliance_mode" in note


# ---- a flat receipt signs neither payload nor signature, nor its key_id ------

_FLAT_ATTACKS = [
    ("payload", {"payload": "Approved wire transfer of $1M by the CFO"}),
    ("payload", {"payload": ["approved", "cfo"]}),
    ("signature", {"signature": {"alg": "ML-DSA-65", "approved_by": "cfo", "agent_id": "agt_cfo"}}),
    ("signature", {"signature": "b3RoZXItc2lnbmF0dXJl"}),
]


@requires_mldsa
@pytest.mark.parametrize(("member", "edit"), _FLAT_ATTACKS, ids=[f"{m}-{i}" for i, (m, _) in enumerate(_FLAT_ATTACKS)])
def test_a_flat_receipt_lists_an_unsigned_payload_or_signature(member: str, edit: dict) -> None:
    receipt, jwks = _vector("asqav-05-hash-mode-prod")
    receipt.update(edit)
    result = asqav.verify_receipt_offline(receipt, jwks)
    assert result["verdict"] == "verified"
    assert _axes(result)["outer_copies"]["note"] == f"not signed, do not rely on: {member}"


def test_a_flat_receipt_lists_a_key_id_no_key_verified_under() -> None:
    receipt, jwks = _vector("asqav-05-hash-mode-prod")
    receipt["key_id"] = "cfo-hsm-key"
    result = asqav.verify_receipt_offline(receipt, jwks)
    assert result["verdict"] == "unverified"
    assert _axes(result)["outer_copies"]["note"] == "not signed, do not rely on: key_id"


@requires_mldsa
def test_a_flat_receipt_does_not_list_the_signature_or_key_id_that_verified() -> None:
    receipt, jwks = _vector("asqav-05-hash-mode-prod")
    receipt["signature"] = receipt["signature_b64"]
    result = asqav.verify_receipt_offline(receipt, jwks)
    assert result["verdict"] == "verified"
    assert _axes(result)["outer_copies"]["note"] == "no outer copies carried; the signed payload is the whole receipt"
