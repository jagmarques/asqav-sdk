"""Type resolution per draft anchor `type` (asqav PR #1469 branch)
protectmcp-absent is non-conformance = invalid; foreign/gap unverifiable"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asqav.verifier import verify_receipt as vr
from asqav.verifier.oracle.semantic_classify import classify as oracle_class

ROOT = Path(__file__).parent.parent.parent
NS_FILE = ROOT / "verifier" / "registry" / "type-namespaces.json"

DRAFT_TEN = {
    "protectmcp:decision",
    "protectmcp:restraint",
    "protectmcp:lifecycle",
    "protectmcp:lifecycle:configuration_change",
    "protectmcp:lifecycle:risk_acceptance",
    "protectmcp:lifecycle:oversight_ruling",
    "protectmcp:lifecycle:code_authorship",
    "protectmcp:acknowledgment",
    "protectmcp:observation",
    "protectmcp:observation:result_bound",
}


def _payload(**kw):
    base = {
        "type": "protectmcp:decision",
        "v": 1,
        "mode": "payload",
        "issuer_id": "iss_1",
        "agent_id": "ag_1",
        "org_id": "org_1",
        "previousReceiptHash": "0" * 64,
        "issued_at": "2026-09-01T12:00:00+00:00",
        "action_ref": "a" * 64,
        "payload_digest": {"hash": "b" * 64, "size": 1},
        "policy_digest": "pd",
        "decision": "allow",
    }
    base.update(kw)
    return base


ROWS = [
    ("protectmcp:decision", "PASS", None, "required fields present"),
    ("protectmcp:nope", "FAIL", "invalid", "unregistered type "),
    ("foreign:thing", "SKIPPED", "unverifiable", "outside the known namespace"),
    (
        "protectmcp:lifecycle:oversight_ruling",
        "SKIPPED",
        "unverifiable",
        "does not implement it",
    ),
    (42, "FAIL", "invalid", "malformed type "),
    (None, "FAIL", "invalid", "malformed type "),
]


@pytest.mark.parametrize("typ,res,cls,needle", ROWS)
def test_split_row(typ, res, cls, needle) -> None:
    got, note = vr.check_structure(_payload(type=typ))
    assert got == res, note
    assert needle in note, note
    if isinstance(typ, str):
        assert typ in note, note
    assert vr._axis_failure_class("structure", got, note) == cls
    assert oracle_class("structure", got, note) == cls


def test_registry_edition_reported() -> None:
    vec = ROOT / "verifier" / "conformance-vectors" / "asqav-25-payload-digest-rederives"
    env = json.loads((vec / "receipt.json").read_text())
    jwks = json.loads((vec / "jwks.json").read_text())
    want = json.loads(NS_FILE.read_text())["registry_edition"]
    assert vr.run_structured(env, jwks)["registry_edition"] == want
    assert vr.run_structured([], {})["registry_edition"] == want


def test_namespace_set_matches_draft() -> None:
    got = {e["namespace"] for e in json.loads(NS_FILE.read_text())["namespaces"]}
    assert got == DRAFT_TEN


def test_literals_match_json() -> None:
    doc = json.loads(NS_FILE.read_text())
    assert vr.REGISTRY_NAMESPACES == {e["namespace"] for e in doc["namespaces"]}
    assert vr.REGISTRY_EDITION == doc["registry_edition"]
    assert vr.ALLOWED_TYPES < vr.REGISTRY_NAMESPACES
