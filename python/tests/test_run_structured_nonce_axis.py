"""run_structured reports the nonce axis that run() reports.

run() took a seen_nonces index and reported a nonce axis; run_structured took no
index and reported no such axis, so the same receipt checked through the two
documented surfaces produced different axis lists, and a caller using the
structured surface got no replay signal at all.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from asqav.verifier.verify_receipt import run_structured

_VECTOR = (
    Path(__file__).parent.parent.parent
    / "verifier"
    / "conformance-vectors"
    / "asqav-01-genesis-permit"
)


@pytest.fixture
def receipt() -> dict:
    return json.loads((_VECTOR / "receipt.json").read_text())


@pytest.fixture
def jwks() -> dict:
    return json.loads((_VECTOR / "jwks.json").read_text())


def _nonce_axis(report: dict) -> dict:
    return next(a for a in report["axes"] if a["name"] == "nonce")


def _with_nonce(receipt: dict, nonce: str, issuer: str = "iss-1") -> dict:
    doc = copy.deepcopy(receipt)
    doc["payload"]["nonce"] = nonce
    doc["payload"]["issuer_id"] = issuer
    return doc


class TestTheNonceAxisIsReported:
    def test_the_axis_is_present_at_all(self, receipt: dict, jwks: dict) -> None:
        """Its absence was the defect: the structured surface reported no nonce axis."""
        names = [a["name"] for a in run_structured(receipt, jwks, None)["axes"]]
        assert "nonce" in names, names

    def test_it_sits_where_run_puts_it(self, receipt: dict, jwks: dict) -> None:
        """run() appends nonce directly after structure; the two surfaces agree on order."""
        names = [a["name"] for a in run_structured(receipt, jwks, None)["axes"]]
        assert names[:2] == ["structure", "nonce"], names

    def test_without_an_index_it_says_so(self, receipt: dict, jwks: dict) -> None:
        """No seen-nonce index is a passthrough, never a silent omission of the axis."""
        axis = _nonce_axis(run_structured(_with_nonce(receipt, "n-1"), jwks, None))
        assert axis["result"] == "PASS"
        assert "no seen-nonce index" in axis["note"]


class TestReplayIsActuallyDetected:
    def test_a_different_receipt_reusing_the_pair_fails(self, receipt: dict, jwks: dict) -> None:
        """The point of the axis: a DIFFERENT receipt on the same (issuer, nonce)."""
        seen: set = set()
        first = _with_nonce(receipt, "n-1")
        replay = copy.deepcopy(first)
        replay["payload"]["action_id"] = "act_DIFFERENT"

        assert _nonce_axis(run_structured(first, jwks, None, seen_nonces=seen))["result"] == "PASS"
        axis = _nonce_axis(run_structured(replay, jwks, None, seen_nonces=seen))
        assert axis["result"] == "FAIL", axis
        assert axis["failure_class"] == "invalid", axis

    def test_re_verifying_the_identical_receipt_is_not_a_replay(
        self, receipt: dict, jwks: dict
    ) -> None:
        """Checking one receipt twice must not manufacture a duplicate-emission finding."""
        seen: set = set()
        doc = _with_nonce(receipt, "n-2")
        run_structured(doc, jwks, None, seen_nonces=seen)
        axis = _nonce_axis(run_structured(copy.deepcopy(doc), jwks, None, seen_nonces=seen))
        assert axis["result"] == "PASS", axis

    def test_a_receipt_with_no_nonce_passes(self, receipt: dict, jwks: dict) -> None:
        """Most receipts declare no nonce; the axis must not punish them."""
        seen: set = set()
        axis = _nonce_axis(run_structured(receipt, jwks, None, seen_nonces=seen))
        assert axis["result"] == "PASS", axis
