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
    REVIEWER_KEY_FILE,
    SIGNATURE_VERIFIED,
)

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


def _signature_verified_dir() -> str:
    return next(
        name
        for name in MINT.vector_dirs()
        if _load(name, "expected.json")["oversight_attestation"] == SIGNATURE_VERIFIED
    )


def _run_planted(tmp_path: Path, receipt: dict, reviewer_key: dict):
    dirname = _signature_verified_dir()
    expected = _load(dirname, "expected.json")
    work = tmp_path / dirname
    work.mkdir()
    (work / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    (work / REVIEWER_KEY_FILE).write_text(json.dumps(reviewer_key), encoding="utf-8")
    for name in ("jwks.json", "expected.json"):
        (work / name).write_text(
            (MINT.CORPUS / dirname / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
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
    outcome = _run_planted(tmp_path, bad, _load(dirname, REVIEWER_KEY_FILE))
    assert outcome.oversight_attestation == ASSERTED
    assert outcome.ok is False


# NEGCTL: pin the receipt signer's key as the reviewer key; the receipt still
# verifies, the attestation signature does not, so the claim is asserted
def test_negctl_a_reviewer_key_that_did_not_sign_is_asserted(tmp_path: Path) -> None:
    dirname = _signature_verified_dir()
    signer = _load(dirname, "jwks.json")["keys"][0]
    outcome = _run_planted(
        tmp_path,
        _load(dirname, "receipt.json"),
        {"alg": signer["alg"], "public_key": signer["public_key"]},
    )
    assert outcome.actual_failure_class == _load(dirname, "expected.json")["failure_class"]
    assert outcome.oversight_attestation == ASSERTED
    assert outcome.ok is False
