"""Outcome members a cloud flat hash-mode receipt signs when set."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

from tests.conftest import run_flat_receipt

from asqav.verifier.verify_receipt import FAILURE_INVALID, VERDICT_UNVERIFIED, VERDICT_VERIFIED

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "flat_hash_outcome_members.json"
_run = partial(
    run_flat_receipt, _FIXTURE,
    extra_output='{"note": vr.check_outer_copies(receipt, b"")[1]}',
)
_OPTIONAL = {
    "plain": (),
    "result_digest": ("result_digest",),
    "action_ref": ("action_ref",),
    "invocation_ref": ("invocation_ref",),
    "all_three": ("result_digest", "action_ref", "invocation_ref"),
}


@pytest.mark.parametrize("name", list(_OPTIONAL))
def test_cloud_flat_receipt_verifies(name):
    result = _run(name)["result"]
    assert result["verdict"] == VERDICT_VERIFIED, result["axes"]


@pytest.mark.parametrize("name", list(_OPTIONAL))
def test_rebuilt_members_are_the_twelve_plus_each_set_outcome(name):
    out = _run(name)
    assert tuple(out["fields"]) == tuple(out["signed"]) + _OPTIONAL[name]


@pytest.mark.parametrize("member", ["result_digest", "action_ref", "invocation_ref"])
def test_tampered_optional_member_fails(member):
    """NEGCTL: a signed outcome member changed after signing is invalid."""
    def tamper(receipt: dict) -> None:
        receipt[member] = "f" * 64

    assert _run("all_three")["result"]["verdict"] == VERDICT_VERIFIED
    result = _run("all_three", tamper)["result"]
    assert result["verdict"] == VERDICT_UNVERIFIED, result["axes"]
    assert result["failure_class"] == FAILURE_INVALID


def test_optional_member_added_after_signing_fails():
    """NEGCTL: an outcome member the signature never covered is invalid."""
    def tamper(receipt: dict) -> None:
        receipt["result_digest"] = "sha256:" + "d" * 64

    assert _run("plain")["result"]["verdict"] == VERDICT_VERIFIED
    result = _run("plain", tamper)["result"]
    assert result["verdict"] == VERDICT_UNVERIFIED, result["axes"]
    assert result["failure_class"] == FAILURE_INVALID


def test_optional_members_are_not_listed_as_unsigned():
    note = _run("all_three")["note"]
    assert "result_digest" not in note and "invocation_ref" not in note


@pytest.mark.parametrize("value", ("", "a'b\"c"))
def test_shared_runner_extra_output_hook_negctl(value):
    """NEGCTL: omitting the hook drops the caller's planted output."""
    extra = {"probe": {"value": value}}
    out = run_flat_receipt(_FIXTURE, "plain", extra_output=repr(extra))
    assert out.get("probe") == extra["probe"]
    assert "probe" not in run_flat_receipt(_FIXTURE, "plain")
