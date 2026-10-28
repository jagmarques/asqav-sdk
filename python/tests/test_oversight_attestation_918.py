"""Criterion 918: oversight-attestation mirrors and the gated review claims."""

from __future__ import annotations

import base64
import copy
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import asqav

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "verifier"))

MINT = importlib.import_module("mint_oversight_attestation_vectors")


def _state_module():
    """The file beside the test: the venv preloads its own editable copy."""
    path = (
        _REPO_ROOT / "python" / "src" / "asqav" / "verifier"
        / "oversight_attestation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "asqav.verifier.oversight_attestation", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STATE = _state_module()
ASSERTED = _STATE.ASSERTED
MISSING = _STATE.MISSING
SIGNATURE_VERIFIED = _STATE.SIGNATURE_VERIFIED
oversight_attestation = _STATE.oversight_attestation


def _manifest() -> list[dict]:
    return json.loads((MINT.CORPUS / "manifest.json").read_text(encoding="utf-8"))


def _load(dirname: str, name: str) -> dict:
    return json.loads(
        (MINT.CORPUS / dirname / name).read_text(encoding="utf-8")
    )


def _signature_ok(receipt: dict, jwks: dict) -> bool:
    axes = asqav.verify_receipt_offline(receipt, jwks)["axes"]
    return next(axis for axis in axes if axis["name"] == "signature")["result"] == "PASS"


def test_oversight_attestation_vectors_are_in_the_manifest() -> None:
    entries = {entry["dir"]: entry for entry in _manifest()}
    for dirname in MINT.vector_dirs():
        assert dirname in entries, dirname
        expected = _load(dirname, "expected.json")
        assert entries[dirname]["notes"] == expected["notes"], dirname
        assert entries[dirname]["outcome"] == expected["outcome"], dirname


def test_oversight_attestation_reports_three_states() -> None:
    reports = []
    for dirname in MINT.vector_dirs():
        receipt = _load(dirname, "receipt.json")
        expected = _load(dirname, "expected.json")
        report = oversight_attestation(
            receipt["payload"], _signature_ok(receipt, _load(dirname, "jwks.json"))
        )
        assert report == expected["oversight_attestation"], dirname
        reports.append(report)
    assert len(set(reports)) == 3
    assert set(reports) == {MISSING, ASSERTED, SIGNATURE_VERIFIED}


# NEGCTL: flip the signature-verified vector's receipt signature; the same
# shipped report calls the claim asserted, never signature-verified.
def test_negctl_a_receipt_whose_signature_does_not_verify_is_asserted() -> None:
    dirname = next(
        name
        for name in MINT.vector_dirs()
        if _load(name, "expected.json")["oversight_attestation"] == SIGNATURE_VERIFIED
    )
    receipt = _load(dirname, "receipt.json")
    jwks = _load(dirname, "jwks.json")
    assert _signature_ok(receipt, jwks) is True
    bad = copy.deepcopy(receipt)
    raw = base64.urlsafe_b64decode(bad["signature"]["sig"] + "==")
    bad["signature"]["sig"] = base64.urlsafe_b64encode(
        bytes([raw[0] ^ 1]) + raw[1:]
    ).decode("ascii").rstrip("=")
    assert _signature_ok(bad, jwks) is False
    report = oversight_attestation(bad["payload"], _signature_ok(bad, jwks))
    assert report == ASSERTED
    assert report != SIGNATURE_VERIFIED
