"""Criterion 918: oversight-attestation mirrors and the gated review claims."""

from __future__ import annotations

import base64
import copy
import importlib
import json
import sys
from pathlib import Path

from asqav.verifier.oracle.core import FAILURE_INVALID, VERDICT_UNVERIFIED
from asqav.verifier.oracle.runner import run_one
from asqav.verifier.oversight_attestation import (
    ASSERTED,
    MISSING,
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


# NEGCTL: flip the signature-verified receipt's signature; the runner's report
# calls it asserted and the declared signature-verified state no longer matches
def test_negctl_a_receipt_whose_signature_does_not_verify_is_asserted(
    tmp_path: Path,
) -> None:
    dirname = next(
        name
        for name in MINT.vector_dirs()
        if _load(name, "expected.json")["oversight_attestation"] == SIGNATURE_VERIFIED
    )
    receipt = _load(dirname, "receipt.json")
    expected = _load(dirname, "expected.json")
    bad = copy.deepcopy(receipt)
    raw = base64.urlsafe_b64decode(bad["signature"]["sig"] + "==")
    bad["signature"]["sig"] = base64.urlsafe_b64encode(
        bytes([raw[0] ^ 1]) + raw[1:]
    ).decode("ascii").rstrip("=")
    work = tmp_path / dirname
    work.mkdir()
    (work / "receipt.json").write_text(json.dumps(bad), encoding="utf-8")
    (work / "jwks.json").write_text(
        (MINT.CORPUS / dirname / "jwks.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (work / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    outcome = run_one(
        work,
        expected["format"],
        VERDICT_UNVERIFIED,
        expected.get("reason_code", ""),
        FAILURE_INVALID,
    )
    assert outcome.actual_failure_class == FAILURE_INVALID
    assert outcome.oversight_attestation == ASSERTED
    assert outcome.oversight_attestation != SIGNATURE_VERIFIED
    assert outcome.ok is False
