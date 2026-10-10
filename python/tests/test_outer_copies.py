"""Unsigned outer copies of signed fields (criterion 742).

An Audit Pack entry restates signed values (agent, action, time, signature string,
key, signed bytes) as plain top-level members no signature covers. Before the
outer_copies axis an edit to one of them still read "signature valid" in every
verifier. These tests pin that a copy disagreeing with the signed payload fails
the verdict in the standalone verifier and in the oracle behind
``verify_receipt_offline``, and that the shared case table gives the same answer,
note included, as the TypeScript half.
"""

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
    assert len(CASES) >= 50


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_shared_case_table(case: dict) -> None:
    key = None if case["key_b64"] is None else base64.b64decode(case["key_b64"])
    result, note = vr.check_outer_copies(case["raw"], base64.b64decode(case["message_b64"]), key)
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
    assert "no signed counterpart, do not rely on: action_type, signature_id" in outer["note"]
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


# ---- conformance vectors (verifier/conformance-vectors/asqav-43..48) --------

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
