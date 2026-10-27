"""Caller cutoffs on the offline verifier, judged under a socket ban.
NEGCTL: the cut cases with their pins omitted keep main's verdict."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from asqav.verifier.verify_receipt import (
    CUTOFF_AXES,
    FAILURE_INVALID,
    FAILURE_UNVERIFIABLE,
    POLICY_DIGEST_RESOLUTION,
    REVOKED_KEY_STATUSES,
    VERDICT_UNVERIFIED,
    VERDICT_VERIFIED,
)
from tests.test_verifier_required_extensions_574 import _signed
from tests.test_verify_receipt import _trusted_envelope, _valid_payload

HOUR = timedelta(hours=1)
SRC = Path(__file__).resolve().parents[1] / "src"
VERIFIER_MODULE = "asqav.verifier.verify_receipt"
MARKER = "NETWORK-ATTEMPT"


def _socket_ban(patch, record) -> None:
    import socket

    def refuse(name, target):
        def blocked(*args, **kwargs):
            record(name, target(args))
            raise OSError(f"socket ban: {name} refused")

        return blocked

    patch(socket.socket, "connect", refuse("connect", lambda a: a[1]))
    patch(socket.socket, "connect_ex", refuse("connect_ex", lambda a: a[1]))
    patch(socket, "create_connection", refuse("create_connection", lambda a: a[0]))
    patch(socket, "getaddrinfo", refuse("getaddrinfo", lambda a: a[:2]))


#: Child preamble: this tree's src first (the installed asqav plugin would
#: shadow it in process), then the ban, each attempt marked on stderr.
BOOT = "\n".join(
    (
        "import sys",
        f"sys.path.insert(0, {str(SRC)!r})",
        inspect.getsource(_socket_ban),
        f"_socket_ban(setattr, lambda *a: print({MARKER!r}, *a, file=sys.stderr))",
    )
)
STRUCTURED = "\n".join(
    (
        BOOT,
        "import json",
        f"from {VERIFIER_MODULE} import run_structured",
        "a = json.load(sys.stdin)",
        "tsa = [bytes.fromhex(a['tsa'])]",
        "out = run_structured(a['envelope'], a['jwks'], None, trusted_tsa_keys=tsa, **a['pins'])",
        "print(json.dumps(out))",
    )
)


def _minted(key=None, **payload_overrides):
    """A receipt that verifies offline, its jwks (key entry updated by key) and
    its issued_at instant."""
    payload, sig, jwks = _signed(**payload_overrides)
    jwks["keys"][0].update(key or {})
    envelope, tsa_pk = _trusted_envelope(payload, sig, kid="k1")
    issued = datetime.fromisoformat(payload["issued_at"].replace("Z", "+00:00"))
    return envelope, jwks, tsa_pk, issued


def _verify(envelope, jwks, tsa_pk, **pins):
    """run_structured of this tree under the socket ban, out of process."""
    pins = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in pins.items()}
    args = {"envelope": envelope, "jwks": jwks, "tsa": tsa_pk.hex(), "pins": pins}
    proc = subprocess.run(
        [sys.executable, "-I", "-c", STRUCTURED],
        input=json.dumps(args),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    attempts = [line for line in proc.stderr.splitlines() if line.startswith(MARKER)]
    assert attempts == []
    return json.loads(proc.stdout)


def _axis(out, pin):
    return next(a for a in out["axes"] if a["name"] == CUTOFF_AXES[pin])


def _unjudged(out):
    return {e["id"] for e in out["coverage"]["checks_not_evaluated"]}


def _held_pins(envelope, issued):
    return {
        "policy_digest": envelope["payload"]["policy_digest"],
        "at_time": issued + timedelta(minutes=1),
        "status_cutoff": issued + HOUR,
    }


@pytest.mark.parametrize("pin", sorted(CUTOFF_AXES))
def test_held_pin_passes_its_axis(pin) -> None:
    envelope, jwks, tsa_pk, issued = _minted()
    out = _verify(envelope, jwks, tsa_pk, **{pin: _held_pins(envelope, issued)[pin]})
    assert (_axis(out, pin)["result"], out["verdict"]) == ("PASS", VERDICT_VERIFIED), out["axes"]
    assert (POLICY_DIGEST_RESOLUTION in _unjudged(out)) == (pin != "policy_digest")


def test_unpinned_run_keeps_policy_resolution_unjudged() -> None:
    envelope, jwks, tsa_pk, _ = _minted()
    out = _verify(envelope, jwks, tsa_pk, policy_digest=None, at_time=None, status_cutoff=None)
    assert out["verdict"] == VERDICT_VERIFIED
    assert POLICY_DIGEST_RESOLUTION in _unjudged(out)
    assert CUTOFF_AXES["policy_digest"] not in [a["name"] for a in out["axes"]]


def test_mismatched_policy_pin_fails_invalid() -> None:
    envelope, jwks, tsa_pk, _ = _minted()
    out = _verify(envelope, jwks, tsa_pk, policy_digest="sha256:" + "4" * 64)
    assert _axis(out, "policy_digest")["result"] == "FAIL"
    assert out["verdict"] != VERDICT_VERIFIED
    assert out["failure_class"] == FAILURE_INVALID


def test_policy_pin_on_removed_payload_is_unverifiable() -> None:
    envelope, jwks, tsa_pk, _ = _minted()
    pinned = envelope["payload"]["policy_digest"]
    envelope["payload"] = None
    out = _verify(envelope, jwks, tsa_pk, policy_digest=pinned)
    assert (out["verdict"], out["failure_class"]) == (VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE)
    assert POLICY_DIGEST_RESOLUTION in _unjudged(out)


def test_at_time_before_issue_is_not_verified() -> None:
    envelope, jwks, tsa_pk, issued = _minted()
    out = _verify(envelope, jwks, tsa_pk, at_time=issued - HOUR)
    assert _axis(out, "at_time")["result"] == "FAIL"
    assert out["verdict"] != VERDICT_VERIFIED


def test_at_time_replaces_the_wall_clock_for_expiry() -> None:
    envelope, jwks, tsa_pk, issued = _minted(expires_at="2099-01-01T00:00:00Z")
    held = _verify(envelope, jwks, tsa_pk, at_time=issued)
    lapsed = _verify(envelope, jwks, tsa_pk, at_time="2099-01-02T00:00:00Z")
    expiry = [next(a for a in o["axes"] if a["name"] == "expiry")["result"] for o in (held, lapsed)]
    assert expiry == ["PASS", "FAIL"]


@pytest.mark.parametrize("status", sorted(REVOKED_KEY_STATUSES))
def test_status_cutoff_past_revoked_at_fails_key_status(status) -> None:
    envelope, jwks, tsa_pk, issued = _minted()
    jwks["keys"][0].update(status=status, revoked_at=(issued + HOUR).isoformat())
    out = _verify(envelope, jwks, tsa_pk, status_cutoff=issued + 2 * HOUR)
    assert _axis(out, "status_cutoff")["result"] == "FAIL"
    assert (out["verdict"], out["failure_class"]) == (VERDICT_UNVERIFIED, FAILURE_INVALID)
    held = _verify(envelope, jwks, tsa_pk, status_cutoff=issued + HOUR / 2)
    unpinned = _verify(envelope, jwks, tsa_pk)
    assert _axis(held, "status_cutoff") == _axis(unpinned, "status_cutoff")


def test_key_without_status_under_cutoff_is_unverifiable() -> None:
    envelope, jwks, tsa_pk, issued = _minted()
    del jwks["keys"][0]["status"]
    out = _verify(envelope, jwks, tsa_pk, status_cutoff=issued + HOUR)
    assert _axis(out, "status_cutoff")["result"] == "SKIPPED"
    assert (out["verdict"], out["failure_class"]) == (VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE)


@pytest.mark.parametrize("status", sorted(REVOKED_KEY_STATUSES))
def test_revoked_key_without_revoked_at_never_passes_a_cutoff(status) -> None:
    envelope, jwks, tsa_pk, issued = _minted(key={"status": status})
    out = _verify(envelope, jwks, tsa_pk, status_cutoff=issued + HOUR)
    assert _axis(out, "status_cutoff")["result"] != "PASS"
    assert out["verdict"] == VERDICT_UNVERIFIED


#: Cut cases: each verifies on main unpinned, and cannot be judged offline once
#: its pin is set. Builders return (minted receipt, pins).
CUT_CASES = {
    "policy-digest-removed": lambda: (
        _minted(policy_digest=None),
        {"policy_digest": _valid_payload("org-legit", "agt_one")["policy_digest"]},
    ),
    "at-time-unparseable": lambda: (_minted(), {"at_time": "not-an-instant"}),
    "status-cutoff-unparseable": lambda: (_minted(), {"status_cutoff": "not-an-instant"}),
    "revoked-at-unparseable": lambda: (
        _minted(key={"revoked_at": "not-an-instant"}),
        {"status_cutoff": "2026-01-01T00:00:00Z"},
    ),
}


def _assert_cut_unverifiable(case: str, pinned: bool) -> None:
    (envelope, jwks, tsa_pk, _), pins = CUT_CASES[case]()
    out = _verify(envelope, jwks, tsa_pk, **(pins if pinned else {}))
    assert (out["verdict"], out["failure_class"]) == (VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE)


@pytest.mark.parametrize("case", sorted(CUT_CASES))
def test_caller_cutoffs_unverifiable_offline_under_socket_ban(case) -> None:
    _assert_cut_unverifiable(case, pinned=True)


@pytest.mark.parametrize("case", sorted(CUT_CASES))
def test_negctl_cut_cases_without_pins_keep_mains_verdict(case) -> None:
    with pytest.raises(AssertionError):
        _assert_cut_unverifiable(case, pinned=False)
