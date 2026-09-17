"""The oracle README's asqav-native axis claim, checked against the axes the
oracle actually returns.

The README once said the oracle runs only the signature, chain and
structure-presence axes and does NOT run clock-skew. Execution shows it runs
twelve axes with skew live in-line (a frozen clock fails skew and flips the
verdict to unverified/invalid). This module executes the oracle on two
asqav-native vectors and asserts the README's enumeration equals the names
that come back, so the claim cannot drift again.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

from asqav.verifier import verify_receipt as vr
from asqav.verifier.oracle import ADAPTERS
from asqav.verifier.oracle.core import verify as oracle_verify

REPO = Path(__file__).parent.parent.parent
README = REPO / "python" / "src" / "asqav" / "verifier" / "oracle" / "README.md"
README_TEXT = README.read_text(encoding="utf-8")

#: The machine-readable enumeration the asqav-native bullet must carry.
DECLARED_LIST = re.compile(r"runs these (\d+) axes, in order:(.*?)(?:\.|$)", re.S)
BACKTICKED = re.compile(r"`{1,2}([a-z_]+)`{1,2}")

#: A claim that something is not evaluated, run, checked or covered.
DENIAL = re.compile(r"\b(not|never|without)\b[^.|\n]{0,80}?\b(evaluat|check|cover|run)\w*", re.I)


def _vector(name: str) -> tuple[dict, dict, dict | None]:
    vd = REPO / "verifier" / "conformance-vectors" / name
    receipt = json.loads((vd / "receipt.json").read_text(encoding="utf-8"))
    jwks = json.loads((vd / "jwks.json").read_text(encoding="utf-8"))
    pred_file = vd / "predecessor.json"
    pred = json.loads(pred_file.read_text(encoding="utf-8")) if pred_file.exists() else None
    return receipt, jwks, pred


def _oracle_axes() -> list[str]:
    seen: list[str] | None = None
    for name in ("asqav-01-genesis-permit", "asqav-03-chain-link"):
        receipt, jwks, pred = _vector(name)
        res = oracle_verify(receipt, ADAPTERS, key_provider=jwks, predecessor=pred)
        assert res.verdict == "verified", [(a.axis, a.result) for a in res.axes]
        names = [a.axis for a in res.axes]
        if seen is None:
            seen = names
        assert names == seen, f"{name} reports a different axis order"
    assert seen is not None
    return seen


def _declared_axes() -> tuple[int, list[str]]:
    match = DECLARED_LIST.search(README_TEXT)
    assert match is not None, "oracle README enumerates no asqav-native axes"
    return int(match.group(1)), BACKTICKED.findall(match.group(2))


def _native_bullet() -> str:
    at = README_TEXT.index("- **asqav-native**")
    end = README_TEXT.index("- **aerf**")
    assert at < end
    # Collapse wrapping first: a denial split across two source lines must
    # still read as one sentence, or the check below passes vacuously.
    return re.sub(r"\s+", " ", README_TEXT[at:end])


class _FrozenPast(_dt.datetime):
    """A wall clock behind the vector's issued_at, so only skew fails."""

    @classmethod
    def now(cls, tz=None):  # noqa: ARG003 - signature mirrors datetime.now
        return _dt.datetime(1990, 1, 1, tzinfo=_dt.timezone.utc)


def test_parser_fires_and_is_not_vacuous() -> None:
    sample = "runs these 2 axes, in order: `structure`, `signature`."
    match = DECLARED_LIST.search(sample)
    assert match is not None
    assert int(match.group(1)) == 2
    assert BACKTICKED.findall(match.group(2)) == ["structure", "signature"]
    assert DECLARED_LIST.search("runs only those three axes") is None
    assert DENIAL.search("does NOT run the clock-skew axis") is not None
    assert DENIAL.search("runs the clock-skew axis live") is None


def test_readme_enumeration_equals_the_axes_the_oracle_returns() -> None:
    returned = _oracle_axes()
    count, declared = _declared_axes()
    assert declared == returned
    assert count == len(returned)


def test_readme_carries_no_skew_denial() -> None:
    bullet = _native_bullet()
    assert "skew" in bullet.lower(), "asqav-native bullet never names skew"
    units = [u for u in re.split(r"[|\n]|(?<=\.)\s", bullet) if u.strip()]
    denials = [u for u in units if "skew" in u.lower() and DENIAL.search(u)]
    assert denials == [], denials


def test_oracle_skew_is_live_and_decides_the_verdict(monkeypatch) -> None:
    receipt, jwks, _ = _vector("asqav-01-genesis-permit")
    assert oracle_verify(receipt, ADAPTERS, key_provider=jwks).verdict == "verified"
    monkeypatch.setattr(vr, "datetime", _FrozenPast)
    res = oracle_verify(receipt, ADAPTERS, key_provider=jwks)
    off = [a for a in res.axes if a.result != "PASS"]
    assert [a.axis for a in off] == ["skew"]
    assert "ahead of wall clock" in off[0].note
    assert res.verdict == "unverified"
    assert res.failure_class == "invalid"
