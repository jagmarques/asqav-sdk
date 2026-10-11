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
    try:
        out = vr.run_structured(
            vr._parse_object(_text(case, "receipt.json"), "receipt"),
            vr._parse_object(_text(case, "jwks.json"), "jwks"),
            trusted_tsa_keys=vr._load_tsa_keys([str(case / "tsa_trust.pem")]),
            counterparty=vr._parse_object(origin, "counterparty") if origin else None,
        )
    except vr.VerifierInputError:
        return "unverified", "unverifiable"
    return out["verdict"], out["failure_class"]


def _standalone_cli(case: Path) -> int:
    args = [sys.executable, "-I", vr.__file__, "--offline", "--receipt", str(case / "receipt.json"),
            "--jwks", str(case / "jwks.json"), "--tsa-key", str(case / "tsa_trust.pem")]
    if (case / "originating_envelope.json").exists():
        args += ["--counterparty", str(case / "originating_envelope.json")]
    env = {**os.environ, "ASQAV_API_URL": "http://127.0.0.1:9"}
    return subprocess.run(args, capture_output=True, text=True, env=env, timeout=120).returncode


def _offline_dict(case: Path) -> tuple[str, str | None]:
    origin = _text(case, "originating_envelope.json")
    out = asqav.verify_receipt_offline(
        json.loads(_text(case, "receipt.json")), json.loads(_text(case, "jwks.json")),
        originating_envelope=json.loads(origin) if origin else None,
    )
    return out["verdict"], out["failure_class"]


def test_manifest_covers_every_zone_with_controls() -> None:
    lexemes = {c["lexeme"] for c in MANIFEST}
    assert {"1e-07", "7.635507284768651e+16", "-0.0", "500.0", "0.5", "5e2", "-0"} <= lexemes
    assert all("binding" in c for c in MANIFEST if c["dir"].startswith("counterparty-"))
    assert sum(c["dir"].endswith("-tamper") for c in MANIFEST) == 10
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
    }
    assert got == dict.fromkeys(got, want), (row["dir"], row["lexeme"], oracle.detail[:300])
    assert _standalone_cli(case) == CLI_EXIT[want], row["dir"]


BINDINGS = [c for c in MANIFEST if "binding" in c]


@pytest.mark.parametrize("row", BINDINGS, ids=[c["dir"] for c in BINDINGS])
def test_binding_label_is_the_sdk_verdict(row: dict) -> None:
    case = CASES_DIR / row["dir"]
    ack = json.loads(_text(case, "receipt.json"))
    origin = json.loads(_text(case, "originating_envelope.json"))
    assert verify_counterparty_binding(ack, origin).label == row["binding"]
