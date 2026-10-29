"""Criterion 918: oversight-attestation mirrors and the gated review claims."""

from __future__ import annotations

import base64
import copy
import importlib
import json
import sys
from pathlib import Path

from asqav.verifier.oracle.runner import run_one
from asqav.verifier.oversight_attestation import (
    ASSERTED,
    MISSING,
    NOT_APPLICABLE,
    SIGNATURE_VERIFIED,
    oversight_attestation,
)
from asqav.verifier.verify_receipt import REGISTRY_NAMESPACES

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "verifier"))

MINT = importlib.import_module("mint_oversight_attestation_vectors")


def _manifest() -> list[dict]:
    return json.loads((MINT.CORPUS / "manifest.json").read_text(encoding="utf-8"))


def _load(dirname: str, name: str) -> dict:
    return json.loads(
        (MINT.CORPUS / dirname / name).read_text(encoding="utf-8")
    )


def _run(dirname: str):
    expected = _load(dirname, "expected.json")
    return run_one(
        MINT.CORPUS / dirname,
        expected["format"],
        expected["outcome"],
        expected.get("reason_code", ""),
        expected.get("failure_class", ""),
    )


def test_oversight_attestation_vectors_are_in_the_manifest() -> None:
    entries = {entry["dir"]: entry for entry in _manifest()}
    for dirname in MINT.vector_dirs():
        assert dirname in entries, dirname
        expected = _load(dirname, "expected.json")
        assert entries[dirname]["notes"] == expected["notes"], dirname
        assert entries[dirname]["outcome"] == expected["outcome"], dirname


def test_oversight_attestation_payloads_mirror_the_platform_policy_digest() -> None:
    for dirname in MINT.vector_dirs():
        payload = _load(dirname, "receipt.json")["payload"]
        assert payload["policy_digest"] == payload["evidence_manifest"][0], dirname
        assert payload["policy_digest"] == f"sha256:{MINT.POLICY_SHA256}", dirname


def test_oversight_attestation_reports_three_states() -> None:
    reports = []
    for dirname in MINT.vector_dirs():
        expected = _load(dirname, "expected.json")
        outcome = _run(dirname)
        assert outcome.ok, dirname
        assert outcome.oversight_attestation == expected["oversight_attestation"], dirname
        reports.append(outcome.oversight_attestation)
    assert len(set(reports)) == 3
    assert set(reports) == {MISSING, ASSERTED, SIGNATURE_VERIFIED}


def test_minted_vectors_carry_the_registered_ruling_type_the_report_gates_on() -> None:
    assert MINT.TYPE in REGISTRY_NAMESPACES
    for dirname in MINT.vector_dirs():
        payload = _load(dirname, "receipt.json")["payload"]
        assert payload["type"] == MINT.TYPE, dirname
        assert oversight_attestation(payload) != NOT_APPLICABLE, dirname


def _signature_verified_dir() -> str:
    return next(
        name
        for name in MINT.vector_dirs()
        if _load(name, "expected.json")["oversight_attestation"] == SIGNATURE_VERIFIED
    )


def _run_planted(
    tmp_path: Path, receipt: dict, jwks: dict, expected: dict | None = None
):
    dirname = _signature_verified_dir()
    expected = expected or _load(dirname, "expected.json")
    work = tmp_path / dirname
    work.mkdir()
    (work / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    (work / "jwks.json").write_text(json.dumps(jwks), encoding="utf-8")
    (work / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    return run_one(
        work,
        expected["format"],
        expected["outcome"],
        expected.get("reason_code", ""),
        expected.get("failure_class", ""),
    )


# NEGCTL: flip one byte of the reviewer's attestation signature; the runner
# reports asserted, so the declared signature-verified state no longer matches
def test_negctl_an_attestation_signature_that_does_not_verify_is_asserted(
    tmp_path: Path,
) -> None:
    dirname = _signature_verified_dir()
    bad = copy.deepcopy(_load(dirname, "receipt.json"))
    signature = bad["payload"]["reviewer"]["attestation"]["signature"]
    raw = base64.urlsafe_b64decode(signature["sig"] + "==")
    signature["sig"] = base64.urlsafe_b64encode(
        bytes([raw[0] ^ 1]) + raw[1:]
    ).decode("ascii").rstrip("=")
    outcome = _run_planted(tmp_path, bad, _load(dirname, "jwks.json"))
    assert outcome.oversight_attestation == ASSERTED
    assert outcome.ok is False


# NEGCTL: carry 058's attestation, signature intact, into a ruling that judges
# another action; the signature binds the ruling, so the claim is asserted
def test_negctl_an_attestation_moved_to_another_ruling_is_asserted(
    tmp_path: Path,
) -> None:
    dirname = _signature_verified_dir()
    moved = copy.deepcopy(_load(dirname, "receipt.json"))
    payload = moved["payload"]
    payload["judged_action_refs"] = [
        *payload["judged_action_refs"],
        payload["policy_digest"],
    ]
    outcome = _run_planted(tmp_path, moved, _load(dirname, "jwks.json"))
    assert outcome.oversight_attestation == ASSERTED
    assert outcome.ok is False


# NEGCTL: change the reviewer's role, signature intact; the signature binds the
# reviewer's identity beside the principal, so the claim is asserted
def test_negctl_an_attestation_under_another_reviewer_role_is_asserted(
    tmp_path: Path,
) -> None:
    dirname = _signature_verified_dir()
    moved = copy.deepcopy(_load(dirname, "receipt.json"))
    reviewer = moved["payload"]["reviewer"]
    reviewer["role"] = reviewer["authority"]
    outcome = _run_planted(tmp_path, moved, _load(dirname, "jwks.json"))
    assert outcome.oversight_attestation == ASSERTED
    assert outcome.ok is False


# NEGCTL: change the issuer, signature intact; the signature binds the whole
# ruling, so another issuer's ruling cannot carry the attestation
def test_negctl_an_attestation_under_another_issuer_is_asserted(
    tmp_path: Path,
) -> None:
    dirname = _signature_verified_dir()
    moved = copy.deepcopy(_load(dirname, "receipt.json"))
    moved["payload"]["issuer_id"] = MINT.ISSUER[::-1]
    outcome = _run_planted(tmp_path, moved, _load(dirname, "jwks.json"))
    assert outcome.oversight_attestation == ASSERTED
    assert outcome.ok is False


# NEGCTL: bind the receipt signer's key to the reviewer's principal; the
# receipt still verifies, the attestation signature does not: asserted
def test_negctl_a_reviewer_bound_to_another_key_is_asserted(tmp_path: Path) -> None:
    dirname = _signature_verified_dir()
    jwks = _load(dirname, "jwks.json")
    signer, reviewer = jwks["keys"]
    reviewer["pub"], reviewer["public_key"] = signer["pub"], signer["public_key"]
    outcome = _run_planted(tmp_path, _load(dirname, "receipt.json"), jwks)
    assert outcome.actual_failure_class == _load(dirname, "expected.json")["failure_class"]
    assert outcome.oversight_attestation == ASSERTED
    assert outcome.ok is False


# NEGCTL: revoke the reviewer's key before the ruling was issued; the
# signature still checks, but a revoked key backs no claim: asserted
def test_negctl_a_revoked_reviewer_key_is_asserted(tmp_path: Path) -> None:
    dirname = _signature_verified_dir()
    jwks = _load(dirname, "jwks.json")
    reviewer = jwks["keys"][1]
    reviewer["status"] = "revoked"
    reviewer["revoked_at"] = "2026-01-01T00:00:00+00:00"
    outcome = _run_planted(tmp_path, _load(dirname, "receipt.json"), jwks)
    assert outcome.actual_failure_class == _load(dirname, "expected.json")["failure_class"]
    assert outcome.oversight_attestation == ASSERTED
    assert outcome.ok is False


# NEGCTL: declare the state as a list, not a string; the runner never reads
# that declaration as met, so the vector fails though the report is right
def test_negctl_a_state_declared_as_a_non_string_never_matches(tmp_path: Path) -> None:
    dirname = _signature_verified_dir()
    expected = _load(dirname, "expected.json")
    expected["oversight_attestation"] = [SIGNATURE_VERIFIED]
    outcome = _run_planted(
        tmp_path, _load(dirname, "receipt.json"), _load(dirname, "jwks.json"), expected
    )
    assert outcome.oversight_attestation == SIGNATURE_VERIFIED
    assert outcome.ok is False
