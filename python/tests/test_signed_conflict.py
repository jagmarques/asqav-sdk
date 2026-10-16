"""Conflicting signed terms through the real oracle (488 C21, C7 core).

Two validly signed agentreceipts geneses over one subject disagree on the
outcome. Both verify (the signatures are good); the comparison must not
fuse them into agreement, and any byte change must fail the signature.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="ed25519 verify needs the cryptography lib")

from asqav.verifier.oracle import ADAPTERS, verify
from asqav.verifier.oracle.live_compare import compare_live

_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"


def _load(name: str) -> tuple[dict, str]:
    raw = (_FIX / f"{name}.json").read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def test_both_conflicting_terms_carry_valid_signatures() -> None:
    for name in ("conflict-allow", "conflict-deny"):
        doc, _ = _load(name)
        res = verify(doc, ADAPTERS)
        assert res.axis("signature").result == "PASS", name
        assert res.verdict == "verified", name


def test_conflict_is_preserved_not_fused_into_agreement() -> None:
    allow, allow_digest = _load("conflict-allow")
    deny, deny_digest = _load("conflict-deny")
    assert allow_digest != deny_digest, "conflicting terms are different bytes"
    out = compare_live(
        allow_digest, verify(allow, ADAPTERS), deny_digest, verify(deny, ADAPTERS))
    assert out.label == "different_subject"
    assert out.differing == (), "same verdicts never imply agreement on terms"


def test_tampered_copy_fails_the_signature() -> None:
    doc, _ = _load("conflict-allow")
    tampered = copy.deepcopy(doc)
    tampered["credentialSubject"]["outcome"]["status"] = "success!"
    res = verify(tampered, ADAPTERS)
    assert res.axis("signature").result == "FAIL"


def test_dropped_evidence_fails_the_signature() -> None:
    doc, _ = _load("conflict-allow")
    dropped = copy.deepcopy(doc)
    del dropped["credentialSubject"]["outcome"]
    res = verify(dropped, ADAPTERS)
    assert res.axis("signature").result == "FAIL", "signed bytes changed"
    assert res.axis("structure").result == "PASS", "top-level shape kept"
