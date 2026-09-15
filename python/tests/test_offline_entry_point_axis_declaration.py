"""Every shipped statement of the offline APIs' axis vocabulary, checked against
the axes those APIs actually return.

Three surfaces enumerate that vocabulary: the ``verify_receipt_offline``
docstring, the ``verifyReceiptOffline`` JSDoc and ``docs/offline-verification.md``.
This module executes the offline entry point and the standalone verifier, then
asserts all three enumerations equal the names that come back, so a coverage claim
cannot drift away from the coverage. The clock-shift case pins the one axis the
three surfaces disagreed about: ``skew`` runs in-line offline and decides the
verdict.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

import pytest

import asqav
from asqav.verifier import verify_receipt as vr

REPO = Path(__file__).parent.parent.parent
DOC_FILE = REPO / "docs" / "offline-verification.md"
TS_FILE = REPO / "typescript" / "src" / "index.ts"
VECTOR = REPO / "verifier" / "conformance-vectors" / "asqav-01-genesis-permit"

DOC_TEXT = DOC_FILE.read_text(encoding="utf-8")
TS_TEXT = TS_FILE.read_text(encoding="utf-8")

AXES_HEADING = "## Axes both offline APIs report"
DIFF_HEADING = "### Where the standalone verifier's vocabulary differs"

#: One marker phrase, so the docstring and the JSDoc parse with the same regex.
DECLARED_LIST = re.compile(r"Reports (\d+) axes in this order:(.*?)\.", re.S)
BACKTICKED = re.compile(r"`{1,2}([a-z_]+)`{1,2}")
DOC_OFFLINE_COUNT = re.compile(r"report these (\d+) axes")
DOC_STANDALONE_COUNT = re.compile(r"report (\d+) axes on the same")

#: A claim that something is not evaluated, run, checked or covered.
DENIAL = re.compile(r"\b(not|never|without)\b[^.|\n]{0,80}?\b(evaluat|check|cover|run)\w*", re.I)


def _receipt_and_jwks() -> tuple[dict, dict]:
    return (
        json.loads((VECTOR / "receipt.json").read_text(encoding="utf-8")),
        json.loads((VECTOR / "jwks.json").read_text(encoding="utf-8")),
    )


def _offline_axes() -> list[str]:
    receipt, jwks = _receipt_and_jwks()
    result = asqav.verify_receipt_offline(receipt, jwks)
    assert result["verdict"] == "verified", result["axes"]
    return [axis["name"] for axis in result["axes"]]


def _standalone_axes() -> list[str]:
    receipt, jwks = _receipt_and_jwks()
    return [axis["name"] for axis in vr.run_structured(receipt, jwks)["axes"]]


def _doc_section(heading: str) -> str:
    """The body under one exact heading, up to the next heading of any level."""
    lines = DOC_TEXT.split("\n")
    if heading not in lines:
        return ""
    start = lines.index(heading) + 1
    for offset in range(start, len(lines)):
        if lines[offset].startswith("#"):
            return "\n".join(lines[start:offset])
    return "\n".join(lines[start:])


def _doc_table_rows(heading: str) -> list[list[str]]:
    """Data-row cells of the first table under one heading, header dropped."""
    rows: list[list[str]] = []
    in_table = False
    for line in _doc_section(heading).split("\n"):
        if not line.startswith("|"):
            if in_table:
                break
            continue
        in_table = True
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        rows.append(cells)
    return rows[1:]


def _first_column_names(heading: str) -> list[str]:
    names = []
    for row in _doc_table_rows(heading):
        match = BACKTICKED.match(row[0])
        assert match is not None, f"first cell is not an axis name: {row[0]!r}"
        names.append(match.group(1))
    return names


def _declared_axes(text: str, where: str) -> tuple[int, list[str]]:
    match = DECLARED_LIST.search(text)
    assert match is not None, f"{where} enumerates no axes"
    return int(match.group(1)), BACKTICKED.findall(match.group(2))


def _ts_jsdoc(symbol: str) -> str:
    """The JSDoc block attached to one exported TypeScript function."""
    at = TS_TEXT.index(f"export function {symbol}(")
    opened = TS_TEXT.rindex("/**", 0, at)
    closed = TS_TEXT.index("*/", opened)
    assert TS_TEXT[closed + 2 : at].strip() == "", f"{symbol} carries no attached JSDoc"
    return TS_TEXT[opened:closed]


def _units(text: str) -> list[str]:
    return [unit for unit in re.split(r"[|\n]|(?<=\.)\s", text) if unit.strip()]


def _skew_denials(text: str) -> list[str]:
    return [unit for unit in _units(text) if "skew" in unit.lower() and DENIAL.search(unit)]


class _FrozenPast(_dt.datetime):
    """A wall clock behind the vector's issued_at, so only the skew axis fails."""

    @classmethod
    def now(cls, tz=None):  # noqa: ARG003 - signature mirrors datetime.now
        return _dt.datetime(2000, 1, 1, tzinfo=_dt.timezone.utc)


def test_doc_parser_fires_and_is_not_vacuous():
    assert _doc_section("## A heading this file does not carry") == ""
    assert _doc_table_rows("## A heading this file does not carry") == []
    assert _skew_denials("The skew axis is not evaluated here") != []
    assert _skew_denials("The skew axis is evaluated here") == []
    assert len(_doc_table_rows(AXES_HEADING)) == 12
    assert len(_doc_table_rows(DIFF_HEADING)) == 3


def test_documented_axis_table_equals_the_axes_the_offline_api_returns():
    returned = _offline_axes()
    assert _first_column_names(AXES_HEADING) == returned
    stated = DOC_OFFLINE_COUNT.search(DOC_TEXT)
    assert stated is not None, f"{DOC_FILE.name} states no axis count"
    assert int(stated.group(1)) == len(returned)


def test_python_docstring_axis_list_equals_the_axes_the_offline_api_returns():
    returned = _offline_axes()
    count, declared = _declared_axes(asqav.verify_receipt_offline.__doc__ or "", "docstring")
    assert declared == returned
    assert count == len(returned)


def test_typescript_jsdoc_axis_list_equals_the_axes_the_offline_api_returns():
    returned = _offline_axes()
    jsdoc = _ts_jsdoc("verifyReceiptOffline")
    count, declared = _declared_axes(jsdoc, "verifyReceiptOffline JSDoc")
    assert declared == returned
    assert count == len(returned)


def test_documented_vocabulary_difference_equals_the_measured_difference():
    offline, standalone = set(_offline_axes()), set(_standalone_axes())
    measured = {
        name: (
            "reported" if name in offline else "not reported",
            "reported" if name in standalone else "not reported",
        )
        for name in offline ^ standalone
    }
    documented = {}
    for row in _doc_table_rows(DIFF_HEADING):
        match = BACKTICKED.match(row[0])
        assert match is not None, f"first cell is not an axis name: {row[0]!r}"
        documented[match.group(1)] = (row[1], row[2])
    assert documented == measured
    stated = DOC_STANDALONE_COUNT.search(DOC_TEXT)
    assert stated is not None, f"{DOC_FILE.name} states no standalone axis count"
    assert int(stated.group(1)) == len(_standalone_axes())


def test_skew_is_evaluated_offline_and_decides_the_verdict(monkeypatch):
    receipt, jwks = _receipt_and_jwks()
    assert asqav.verify_receipt_offline(receipt, jwks)["verdict"] == "verified"
    monkeypatch.setattr(vr, "datetime", _FrozenPast)
    result = asqav.verify_receipt_offline(receipt, jwks)
    off_axes = [axis for axis in result["axes"] if axis["result"] != "PASS"]
    assert [axis["name"] for axis in off_axes] == ["skew"]
    assert off_axes[0]["note"].endswith(f"ahead of wall clock (> {vr.SKEW_BOUND_SECONDS}s)")
    assert result["verdict"] == "unverified"
    assert result["failure_class"] == "invalid"


@pytest.mark.parametrize(
    "where",
    ["docstring", "jsdoc", "doc"],
)
def test_no_surface_denies_that_skew_is_evaluated(where):
    text = {
        "docstring": asqav.verify_receipt_offline.__doc__ or "",
        "jsdoc": _ts_jsdoc("verifyReceiptOffline"),
        "doc": DOC_TEXT,
    }[where]
    assert _skew_denials(text) == []


def test_every_surface_states_the_skew_bound_the_code_applies():
    bound = f"{vr.SKEW_BOUND_SECONDS} seconds"
    assert bound in (asqav.verify_receipt_offline.__doc__ or "")
    assert bound in _ts_jsdoc("verifyReceiptOffline")
    assert bound in DOC_TEXT
