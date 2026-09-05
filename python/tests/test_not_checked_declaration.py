"""Non-coverage is declared in the output, on every result, passing ones included.

A verifier that reports only what it checked lets a reader mistake silence for
coverage. Prose in a README does not travel with the result and is not machine
readable, so the boundary of the claim has to sit beside the claim.

These tests pin three things: that the declaration is on every return path this
function has, that it survives a caller mutating it, and that a newly added
return path cannot quietly ship without it.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from asqav.verifier.verify_receipt import (
    NOT_CHECKED,
    coverage_declaration,
    not_checked_declaration,
    run_structured,
)

_VECTORS = pathlib.Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"
_REQUIRED_KEYS = {"check", "requirement", "reason", "condition"}


def _vector(name: str):
    d = _VECTORS / name
    return json.loads((d / "receipt.json").read_text()), json.loads((d / "jwks.json").read_text())


def test_declaration_is_non_empty_and_well_formed():
    """Every entry names a check and says why it is not performed."""
    assert NOT_CHECKED, "an empty declaration claims total coverage"
    for entry in NOT_CHECKED:
        assert set(entry) == _REQUIRED_KEYS, entry
        assert entry["check"] and entry["check"] == entry["check"].lower()
        assert entry["reason"].strip(), entry
        # `condition` is deliberately nullable: None means never performed.
        assert entry["condition"] is None or entry["condition"].strip()


def test_declaration_names_are_unique():
    """Two entries under one name would let a reader miss the second."""
    names = [e["check"] for e in NOT_CHECKED]
    assert len(names) == len(set(names)), names


def test_present_on_a_rejected_input():
    """The earliest bail-out still declares what it does not check."""
    result = run_structured("not an envelope", {})
    assert result["verdict"] == "unverified"
    assert len(result["not_checked"]) == len(NOT_CHECKED)


def test_present_on_a_fully_evaluated_receipt_whose_signature_verified():
    """The path that can report `verified` carries the declaration.

    This vector runs the real ML-DSA-65 verify, so the signature axis passing is
    evidence the crypto ran rather than that it was skipped.
    """
    receipt, jwks = _vector("asqav-06-mldsa65-payload-prod")
    result = run_structured(receipt, jwks)
    signature = next(a for a in result["axes"] if a["name"] == "signature")
    if signature["result"] == "SKIPPED":
        pytest.skip("dilithium-py absent; the signature axis cannot run here")
    assert signature["result"] == "PASS"
    assert len(result["not_checked"]) == len(NOT_CHECKED)


def test_a_caller_cannot_narrow_what_later_results_declare():
    """Each result gets its own copy, so mutation does not reach the constant."""
    first = run_structured("not an envelope", {})
    first["not_checked"].clear()
    first_entry = not_checked_declaration()[0]
    first_entry["reason"] = "mutated"
    assert len(run_structured("not an envelope", {})["not_checked"]) == len(NOT_CHECKED)
    assert not_checked_declaration()[0]["reason"] != "mutated"


def test_every_return_path_of_run_structured_declares_it():
    """The anti-regression gate: a new early return cannot omit the declaration.

    Parsed from the source rather than exercised, because the point is to catch a
    return path no test happens to reach - which is exactly the path that would
    ship undeclared.
    """
    source = pathlib.Path(
        __import__("asqav.verifier.verify_receipt", fromlist=["_"]).__file__
    ).read_text()
    tree = ast.parse(source)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_structured"
    )
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert returns, "run_structured has no return statements; the gate is vacuous"
    for node in returns:
        assert isinstance(node.value, ast.Dict), "a non-dict return cannot carry the declaration"
        keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        assert "not_checked" in keys, (
            f"run_structured returns at line {node.lineno} without not_checked"
        )


#: The twelve axes after the structure gate on the normal path, pinned verbatim so
#: a reorder of the verifier's sequence fails here instead of editing the pin.
_AXES_AFTER_STRUCTURE = (
    "nonce", "issuer_key", "issuer_bind", "key_status", "signature", "key_binding",
    "counterparty", "payload_digest", "chain", "anchors", "skew", "expiry",
)


def test_coverage_block_on_a_full_run():
    """A receipt that runs the whole sequence reports stopped_at null and the table."""
    receipt, jwks = _vector("asqav-01-genesis-permit")
    result = run_structured(receipt, jwks)
    # The pin is the live axis sequence itself: the coverage constant must track it.
    assert [a["name"] for a in result["axes"]] == ["structure", *_AXES_AFTER_STRUCTURE]
    cov = result["coverage"]
    assert cov["stopped_at"] is None
    entries = cov["checks_not_evaluated"]
    assert [e["id"] for e in entries] == [e["check"] for e in NOT_CHECKED]
    for entry in entries:
        assert entry["reason"] == "not_implemented", entry
        assert entry["status"] == "not_implemented", entry
        assert list(entry)[:3] == ["id", "reason", "status"], entry
        assert entry["requirement"] and "condition" in entry, entry


def test_coverage_block_on_the_early_returns():
    """The structure gate stops evaluation; the other axes read not_reached."""
    for bad in ("not an envelope", {"payload": "not a dict"}):
        result = run_structured(bad, {})
        cov = result["coverage"]
        assert cov["stopped_at"] == "structure", bad
        entries = cov["checks_not_evaluated"]
        not_impl = [e for e in entries if e["reason"] == "not_implemented"]
        not_reached = [e for e in entries if e["reason"] == "not_reached"]
        # not_implemented entries come first, in NOT_CHECKED table order.
        assert [e["id"] for e in not_impl] == [e["check"] for e in NOT_CHECKED]
        assert [e["id"] for e in not_reached] == list(_AXES_AFTER_STRUCTURE)
        for entry in not_reached:
            assert entry["status"] == "implemented", entry
            assert list(entry) == ["id", "reason", "status"], entry


def test_coverage_not_implemented_ids_match_the_table_in_order():
    """Parity: the block's not_implemented ids ARE this language's table, in order."""
    for axes in ([], [{"name": "expiry"}]):
        entries = coverage_declaration(axes)["checks_not_evaluated"]
        not_impl = [e["id"] for e in entries if e["reason"] == "not_implemented"]
        assert not_impl == [e["check"] for e in NOT_CHECKED]


def test_a_caller_cannot_narrow_the_coverage_block_of_later_results():
    """Each result gets its own block; mutation does not reach the helper."""
    first = run_structured("not an envelope", {})
    first["coverage"]["checks_not_evaluated"].clear()
    fresh = run_structured("not an envelope", {})["coverage"]
    assert len(fresh["checks_not_evaluated"]) == len(NOT_CHECKED) + len(_AXES_AFTER_STRUCTURE)


def test_every_return_path_of_run_structured_carries_coverage():
    """The anti-regression gate for the coverage block, parsed from source.

    Same construction as the not_checked gate above: a return path no test
    reaches is exactly the one that would ship without the block.
    """
    source = pathlib.Path(
        __import__("asqav.verifier.verify_receipt", fromlist=["_"]).__file__
    ).read_text()
    tree = ast.parse(source)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_structured"
    )
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert returns, "run_structured has no return statements; the gate is vacuous"
    for node in returns:
        assert isinstance(node.value, ast.Dict), "a non-dict return cannot carry coverage"
        keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        assert "coverage" in keys, (
            f"run_structured returns at line {node.lineno} without coverage"
        )
