"""Caller cutoffs on the offline verifier, judged under a socket ban."""
# NEGCTL: the cut cases unpinned keep main's verified verdict.
# NEGCTL: a planted urlopen trips the ban; the CLI fetch is recorded.

from __future__ import annotations

import base64
import inspect
import json
import subprocess
import sys
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from asqav.verifier.verify_receipt import (
    CUTOFF_AXES,
    FAILURE_INVALID,
    FAILURE_UNVERIFIABLE,
    JWKS_URL,
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
JWKS_HOST = urllib.parse.urlsplit(JWKS_URL).hostname


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
#: A mutant of the ban proof: one real urlopen before the shipped call.
PLANTED = "\n".join(
    (
        BOOT,
        "import json, urllib.request",
        f"import {VERIFIER_MODULE} as m",
        "real = m.run_structured",
        "def planted(*args, **kwargs):",
        "    try:",
        "        urllib.request.urlopen('http://example.invalid')",
        "    except OSError as exc:",
        "        print(f'planted attempt refused: {exc}', file=sys.stderr)",
        "    return real(*args, **kwargs)",
        "m.run_structured = planted",
        "a = json.load(sys.stdin)",
        "tsa = [bytes.fromhex(a['tsa'])]",
        "out = m.run_structured(a['envelope'], a['jwks'], None, trusted_tsa_keys=tsa, **a['pins'])",
        "print(json.dumps(out))",
    )
)
CLI = "\n".join(
    (
        BOOT,
        "import json, runpy, sys",
        "a = json.load(sys.stdin)",
        "sys.argv = ['verify_receipt'] + a['argv']",
        f"runpy.run_module({VERIFIER_MODULE!r}, run_name='__main__')",
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


def _attempts(stderr: str) -> list:
    out = []
    for line in stderr.splitlines():
        parts = line.split(maxsplit=2)
        if parts and parts[0] == MARKER and len(parts) > 1:
            out.append((parts[1], parts[2] if len(parts) > 2 else ""))
    return out


def _assert_no_egress(attempts: list) -> None:
    assert not attempts, f"{MARKER}: network attempt(s) recorded: {attempts}"


def _verify(envelope, jwks, tsa_pk, script=STRUCTURED, **pins):
    """run_structured of this tree under the socket ban, out of process."""
    pins = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in pins.items()}
    args = {"envelope": envelope, "jwks": jwks, "tsa": tsa_pk.hex(), "pins": pins}
    proc = subprocess.run(
        [sys.executable, "-I", "-c", script],
        input=json.dumps(args),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    _assert_no_egress(_attempts(proc.stderr))
    return json.loads(proc.stdout)


def _run_cli(argv, *, demand_offline=True):
    proc = subprocess.run(
        [sys.executable, "-I", "-c", CLI],
        input=json.dumps({"argv": argv}),
        capture_output=True,
        text=True,
        timeout=120,
    )
    attempts = _attempts(proc.stderr)
    if demand_offline:
        _assert_no_egress(attempts)
    return proc, attempts


def _pin_flag(pin: str) -> str:
    return "--" + pin.replace("_", "-")


def _pin_flags(pins: dict) -> list:
    flags = []
    for pin, value in pins.items():
        flags += [_pin_flag(pin), value.isoformat() if isinstance(value, datetime) else str(value)]
    return flags


def _cli_args(tmp_path, envelope, jwks, tsa_pk, pins, *, jwks_arg=True, offline=True):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(envelope))
    tsa = tmp_path / "tsa.b64"
    tsa.write_text(base64.b64encode(tsa_pk).decode())
    args = ["--receipt", str(receipt), "--tsa-key", str(tsa)]
    if jwks_arg:
        path = tmp_path / "jwks.json"
        path.write_text(json.dumps(jwks))
        args += ["--jwks", str(path)]
    if offline:
        args.append("--offline")
    return args + _pin_flags(pins)


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
    assert POLICY_DIGEST_RESOLUTION in _unjudged(out)


def test_unpinned_run_keeps_policy_resolution_and_verdict() -> None:
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


@pytest.mark.parametrize("status", ("", "unknown", "decommissioned"))
def test_status_cutoff_never_rescues_a_non_revoked_skip(status) -> None:
    envelope, jwks, tsa_pk, issued = _minted()
    jwks["keys"][0].update(status=status, revoked_at=(issued + HOUR).isoformat())
    out = _verify(envelope, jwks, tsa_pk, status_cutoff=issued + HOUR / 2)
    assert _axis(out, "status_cutoff")["result"] == "SKIPPED"
    assert (out["verdict"], out["failure_class"]) == (VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE)


@pytest.mark.parametrize("status", sorted(REVOKED_KEY_STATUSES))
def test_status_cutoff_past_a_pre_issuance_revocation_fails_key_status(status) -> None:
    envelope, jwks, tsa_pk, issued = _minted()
    jwks["keys"][0].update(status=status, revoked_at=(issued - HOUR).isoformat())
    out = _verify(envelope, jwks, tsa_pk, status_cutoff=issued + HOUR)
    assert _axis(out, "status_cutoff")["result"] == "FAIL"
    assert (out["verdict"], out["failure_class"]) == (VERDICT_UNVERIFIED, FAILURE_INVALID)


def test_status_cutoff_keeps_an_anchored_post_issuance_revocation_row() -> None:
    envelope, jwks, tsa_pk, issued = _minted()
    jwks["keys"][0].update(status="revoked", revoked_at=(issued + HOUR).isoformat())
    unpinned = _verify(envelope, jwks, tsa_pk)
    pinned = _verify(envelope, jwks, tsa_pk, status_cutoff=issued + 2 * HOUR)
    assert _axis(pinned, "status_cutoff") == _axis(unpinned, "status_cutoff")
    assert _axis(pinned, "status_cutoff")["result"] == "PASS"


def test_status_cutoff_before_the_revocation_stays_skipped_without_an_anchor() -> None:
    envelope, jwks, tsa_pk, issued = _minted()
    envelope.pop("anchors")
    jwks["keys"][0].update(status="revoked", revoked_at=(issued + HOUR).isoformat())
    unpinned = _verify(envelope, jwks, tsa_pk)
    pinned = _verify(envelope, jwks, tsa_pk, status_cutoff=issued + HOUR / 2)
    assert _axis(unpinned, "status_cutoff")["result"] == "SKIPPED"
    assert _axis(pinned, "status_cutoff")["result"] == "SKIPPED"
    assert (pinned["verdict"], pinned["failure_class"]) == (VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE)


def test_status_cutoff_before_the_revocation_passes_under_an_anchored_cutoff() -> None:
    envelope, jwks, tsa_pk, issued = _minted()
    jwks["keys"][0].update(status="revoked", revoked_at=(issued + HOUR).isoformat())
    out = _verify(envelope, jwks, tsa_pk, status_cutoff=issued + HOUR / 2)
    assert _axis(out, "status_cutoff")["result"] == "PASS"
    assert out["verdict"] == VERDICT_VERIFIED


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
    (envelope, jwks, tsa_pk, _), _pins = CUT_CASES[case]()
    out = _verify(envelope, jwks, tsa_pk)
    assert out["verdict"] == VERDICT_VERIFIED
    assert out["failure_class"] is None


def test_planted_network_attempt_trips_the_socket_ban() -> None:
    """NEGCTL: the ban _verify relies on catches a real urlopen in its child."""
    envelope, jwks, tsa_pk, issued = _minted()
    with pytest.raises(AssertionError, match=MARKER):
        _verify(envelope, jwks, tsa_pk, script=PLANTED, status_cutoff=issued + HOUR)


@pytest.mark.parametrize("case", ("policy-digest-removed", "revoked-at-unparseable"))
def test_cli_cutoffs_offline_make_no_network_call(tmp_path, case) -> None:
    (envelope, jwks, tsa_pk, _), pins = CUT_CASES[case]()
    proc, attempts = _run_cli(_cli_args(tmp_path, envelope, jwks, tsa_pk, pins))
    assert attempts == []
    assert proc.returncode != 0
    assert f"failure_class: {FAILURE_UNVERIFIABLE}" in proc.stdout


@pytest.mark.parametrize("pin", ("at_time", "status_cutoff"))
def test_cli_rejects_an_unparseable_instant(tmp_path, pin) -> None:
    envelope, jwks, tsa_pk, _ = _minted()
    args = _cli_args(tmp_path, envelope, jwks, tsa_pk, {pin: "not-an-instant"})
    proc, attempts = _run_cli(args)
    assert attempts == []
    assert proc.returncode != 0
    assert f"{_pin_flag(pin)} 'not-an-instant' is not an ISO-8601 instant" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_at_time_judges_the_receipt_at_the_pin(tmp_path) -> None:
    envelope, jwks, tsa_pk, issued = _minted()
    args = _cli_args(tmp_path, envelope, jwks, tsa_pk, {"at_time": issued - HOUR})
    flagged, attempts = _run_cli(args)
    assert attempts == []
    assert flagged.returncode != 0
    assert f"[FAIL] {CUTOFF_AXES['at_time']} " in flagged.stdout
    unfixed, _ = _run_cli(_cli_args(tmp_path, envelope, jwks, tsa_pk, {}))
    assert unfixed.returncode == 0


def test_cli_fetch_without_offline_is_caught_by_the_ban(tmp_path) -> None:
    """NEGCTL: without --jwks/--offline the ban records the fetch."""
    envelope, jwks, tsa_pk, _ = _minted()
    pins = {"policy_digest": envelope["payload"]["policy_digest"]}
    args = _cli_args(tmp_path, envelope, jwks, tsa_pk, pins, jwks_arg=False, offline=False)
    proc, attempts = _run_cli(args, demand_offline=False)
    assert proc.returncode != 0
    assert any(name == "create_connection" and JWKS_HOST in target for name, target in attempts), attempts
    with pytest.raises(AssertionError, match="network attempt"):
        _assert_no_egress(attempts)
