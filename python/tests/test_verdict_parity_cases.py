"""Every Python verifier returns the manifest verdict on the signed float receipts (747).

The TypeScript half, typescript/tests/verifier-verdict-parity.test.ts, reads the same manifest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["ASQAV_API_URL"] = "http://127.0.0.1:9"

import asqav  # noqa: E402
from asqav.counterparty import verify_counterparty_binding  # noqa: E402
from asqav.verifier import pack_verify as pv  # noqa: E402
from asqav.verifier import verify_receipt as vr  # noqa: E402
from asqav.verifier.oracle.runner import run_one  # noqa: E402

VERIFIER_DIR = Path(__file__).resolve().parents[2] / "verifier"
CASES_DIR = VERIFIER_DIR / "verdict-parity-cases"
MANIFEST = json.loads((CASES_DIR / "manifest.json").read_text())["cases"]
CLI_EXIT = {("verified", None): 0, ("unverified", "invalid"): 1, ("unverified", "unverifiable"): 2}


def _text(case: Path, name: str) -> str | None:
    path = case / name
    return path.read_text() if path.exists() else None


def _standalone(case: Path) -> tuple[str, str | None]:
    origin = _text(case, "originating_envelope.json")
    prev = _text(case, "predecessor.json")
    try:
        pred = vr._parse_object(prev, "predecessor") if prev else None
        prev_payload = pred.get("payload", pred) if pred else None
        out = vr.run_structured(
            vr._parse_object(_text(case, "receipt.json"), "receipt"),
            vr._parse_object(_text(case, "jwks.json"), "jwks"),
            prev_payload,
            trusted_tsa_keys=vr._load_tsa_keys([str(case / "tsa_trust.pem")]),
            counterparty=vr._parse_object(origin, "counterparty") if origin else None,
        )
    except vr.VerifierInputError:
        return "unverified", "unverifiable"
    return out["verdict"], out["failure_class"]


def _loaded(case: Path, name: str):
    text = _text(case, name)
    return json.loads(text) if text else None


def _structured_dict(case: Path) -> tuple[str, str | None]:
    # The public function on json.loads dicts, whole predecessor included.
    out = vr.run_structured(
        _loaded(case, "receipt.json"), _loaded(case, "jwks.json"), _loaded(case, "predecessor.json"),
        trusted_tsa_keys=vr._load_tsa_keys([str(case / "tsa_trust.pem")]),
        counterparty=_loaded(case, "originating_envelope.json"),
    )
    return out["verdict"], out["failure_class"]


PACK_STATUS = {pv.VERIFIED: ("verified", None), pv.INVALID: ("unverified", "invalid"),
               pv.NOT_OFFLINE: ("unverified", "unverifiable")}


def _pack_entry(case: Path) -> tuple[str, str | None]:
    out = pv.verify_entry(
        _loaded(case, "receipt.json"), _loaded(case, "jwks.json"), _loaded(case, "predecessor.json"),
        tsa_keys=vr._load_tsa_keys([str(case / "tsa_trust.pem")]),
    )
    return PACK_STATUS[out["status"]]


def _standalone_cli(case: Path) -> int:
    args = [sys.executable, "-I", vr.__file__, "--offline", "--receipt", str(case / "receipt.json"),
            "--jwks", str(case / "jwks.json"), "--tsa-key", str(case / "tsa_trust.pem")]
    if (case / "originating_envelope.json").exists():
        args += ["--counterparty", str(case / "originating_envelope.json")]
    if (case / "predecessor.json").exists():
        args += ["--predecessor", str(case / "predecessor.json")]
    env = {**os.environ, "ASQAV_API_URL": "http://127.0.0.1:9"}
    return subprocess.run(args, capture_output=True, text=True, env=env, timeout=120).returncode


def _offline_dict(case: Path) -> tuple[str, str | None]:
    origin = _text(case, "originating_envelope.json")
    prev = _text(case, "predecessor.json")
    out = asqav.verify_receipt_offline(
        json.loads(_text(case, "receipt.json")), json.loads(_text(case, "jwks.json")),
        predecessor=json.loads(prev) if prev else None,
        originating_envelope=json.loads(origin) if origin else None,
    )
    return out["verdict"], out["failure_class"]


def test_manifest_covers_every_zone_with_controls() -> None:
    lexemes = {c["lexeme"] for c in MANIFEST}
    assert {"1e-07", "7.635507284768651e+16", "-0.0", "500.0", "0.5", "5e2", "-0"} <= lexemes
    assert all("binding" in c for c in MANIFEST if c["dir"].startswith("counterparty-"))
    assert sum(c["dir"].endswith("-tamper") for c in MANIFEST) == 10
    # Integers past 2**53, signed and unsigned, receipts and bindings alike.
    unsafe = {c["dir"] for c in MANIFEST if c.get("unsafe_integer")}
    parts = ("receipt-unsafe-int-signed", "receipt-unsafe-int-unsigned",
             "receipt-unsafe-int-predecessor", "counterparty-")
    for part in parts:
        assert any(d.startswith(part) for d in unsafe), part
    assert {c["outcome"] for c in MANIFEST} == {"verified", "unverified"}
    dirs = sorted(p.name for p in CASES_DIR.iterdir() if p.is_dir())
    assert dirs == sorted(c["dir"] for c in MANIFEST)


def test_fixtures_equal_generator_output() -> None:
    gen = VERIFIER_DIR / "generate_verdict_parity_cases.py"
    env = {**os.environ, "ASQAV_API_URL": "http://127.0.0.1:9"}
    proc = subprocess.run([sys.executable, "-I", str(gen), "--check"], capture_output=True,
                          text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("row", MANIFEST, ids=[c["dir"] for c in MANIFEST])
def test_every_python_verifier_returns_the_manifest_outcome(row: dict) -> None:
    case = CASES_DIR / row["dir"]
    want = (row["outcome"], row["failure_class"])
    oracle = run_one(case, "asqav-native", row["outcome"], "", row["failure_class"] or "")
    got = {
        "oracle": (oracle.actual_verdict, oracle.actual_failure_class or None),
        "verify_receipt.py": _standalone(case),
        "verify_receipt_offline": _offline_dict(case),
        "run_structured(dict)": _structured_dict(case),
    }
    if not (case / "originating_envelope.json").exists():
        got["pack_verify.verify_entry"] = _pack_entry(case)  # takes no origin
    assert got == dict.fromkeys(got, want), (row["dir"], row["lexeme"], oracle.detail[:300])
    assert _standalone_cli(case) == CLI_EXIT[want], row["dir"]


BINDINGS = [c for c in MANIFEST if "binding" in c]


@pytest.mark.parametrize("row", BINDINGS, ids=[c["dir"] for c in BINDINGS])
def test_binding_label_is_the_sdk_verdict(row: dict) -> None:
    case = CASES_DIR / row["dir"]
    ack = json.loads(_text(case, "receipt.json"))
    origin = json.loads(_text(case, "originating_envelope.json"))
    assert verify_counterparty_binding(ack, origin).label == row["binding"]


# The oracle CLI takes no origin, so only rows whose receipt holds the integer.
UNSAFE = [c for c in MANIFEST if c.get("unsafe_integer") and "-origin-" not in c["dir"]]


@pytest.mark.parametrize("row", UNSAFE, ids=[c["dir"] for c in UNSAFE])
def test_oracle_cli_refuses_an_integer_past_2_53(row: dict) -> None:
    case = CASES_DIR / row["dir"]
    args = [sys.executable, "-m", "asqav.verifier.oracle", str(case / "receipt.json"),
            "--keys", str(case / "jwks.json")]
    if (case / "predecessor.json").exists():
        args += ["--predecessor", str(case / "predecessor.json")]
    env = {**os.environ, "ASQAV_API_URL": "http://127.0.0.1:9",
           "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    proc = subprocess.run(args, capture_output=True, text=True, env=env, timeout=120)
    # Exit 2 is unverifiable; a crash exits 1, which reads as invalid.
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
