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
