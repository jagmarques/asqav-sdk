"""Signature wire-form reporting (criterion 657 C3).

A historical signature encoding verifies and names its trusted policy in the
signature axis note; junk decoding to nothing is refused, not a crypto
mismatch. Fresh unpadded-base64url signatures verify silently.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

from asqav.verifier import verify_receipt as vr

VECTORS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"


def _load(vec: str, name: str) -> dict:
    return json.loads((VECTORS / vec / name).read_text())


def _sig_axis(res: dict) -> dict:
    return next(a for a in res["axes"] if a["name"] == "signature")


def test_genesis_signature_note_names_the_legacy_policy() -> None:
    env = _load("asqav-01-genesis-permit", "receipt.json")
    jwks = _load("asqav-01-genesis-permit", "jwks.json")
    axis = _sig_axis(vr.run_structured(env, jwks))
    assert axis["result"] == "PASS"
    assert "legacy" in axis["note"]


def test_junk_signature_is_refused_not_mismatched() -> None:
    env = _load("asqav-01-genesis-permit", "receipt.json")
    jwks = _load("asqav-01-genesis-permit", "jwks.json")
    env["signature"]["sig"] = "!!*"
    res = vr.run_structured(env, jwks)
    axis = _sig_axis(res)
    assert axis["result"] == "FAIL"
    assert "decodable" in axis["note"]
    assert "mismatch" not in axis["note"]
    assert res["failure_class"] == "invalid"


def test_contract_form_verifies_without_a_legacy_note() -> None:
    env = _load("asqav-01-genesis-permit", "receipt.json")
    jwks = _load("asqav-01-genesis-permit", "jwks.json")
    raw = base64.b64decode(
        env["signature"]["sig"].replace("-", "+").replace("_", "/")
    )
    env = copy.deepcopy(env)
    env["signature"]["sig"] = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    axis = _sig_axis(vr.run_structured(env, jwks))
    assert axis["result"] == "PASS"
    assert "legacy" not in axis["note"]
