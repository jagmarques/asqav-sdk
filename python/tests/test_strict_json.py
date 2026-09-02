"""Strict JSON ingest (criterion 419): duplicate members fail closed.

Every receipt- and record-parsing path rejects a duplicated JSON member name at
ANY nesting depth as a terminal parse failure, before any hashing,
canonicalisation, or signature check. Last-wins ingest would hash the bytes an
attacker kept and drop the ones they replaced; these tests pin the rejection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asqav import strict_json
from asqav.verifier import verify_receipt as vr

_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"


    # A duplicated member name at the top level raises the terminal error.
def test_top_level_duplicate_member_rejected() -> None:
    with pytest.raises(strict_json.DuplicateMemberError):
        strict_json.loads('{"payload": {"a": 1}, "payload": {"a": 2}}')


    # A duplicated member name nested inside an object raises too.
def test_nested_duplicate_member_rejected() -> None:
    with pytest.raises(strict_json.DuplicateMemberError):
        strict_json.loads('{"payload": {"digest": {"hash": "x", "hash": "y"}}}')


    # Depth does not weaken the gate: a duplicate five levels down is terminal.
def test_deeply_nested_duplicate_member_rejected() -> None:
    text = '{"a": {"b": {"c": {"d": [{"e": 1, "e": 2}]}}}}'
    with pytest.raises(strict_json.DuplicateMemberError):
        strict_json.loads(text)


    # Duplicates inside an array element are rejected at any depth.
def test_duplicate_member_inside_array_element_rejected() -> None:
    with pytest.raises(strict_json.DuplicateMemberError):
        strict_json.loads('{"list": [{"k": 1}, {"k": 2, "k": 3}]}')


    # The same member name in SIBLING objects is legal; only one object scope matters.
def test_same_name_in_sibling_objects_is_allowed() -> None:
    out = strict_json.loads('{"list": [{"k": 1}, {"k": 2}]}')
    assert out == {"list": [{"k": 1}, {"k": 2}]}


    # Clean documents parse exactly like the stdlib decoder.
def test_clean_documents_parse_like_stdlib() -> None:
    samples = [
        "{}",
        '{"a": 1, "b": [true, false, null], "c": {"d": "e"}}',
        '{"nested": [[[[1]]]], "mix": [{"x": [2, {"y": 3}]}]}',
        '[1, 2, {"a": {"b": [3]}}]',
    ]
    for text in samples:
        assert strict_json.loads(text) == json.loads(text), text


    # The object_pairs_hook is reserved for duplicate rejection, never overridden.
def test_loads_refuses_a_caller_object_pairs_hook() -> None:
    with pytest.raises(ValueError):
        strict_json.loads("{}", object_pairs_hook=dict)


    # The standalone verifier parses with the same gate (self-contained copy).
def test_standalone_verifier_rejects_duplicate_members() -> None:
    with pytest.raises(vr.VerifierInputError, match="duplicate JSON member name"):
        vr._parse_object('{"payload": 1, "payload": 2}', "receipt.json")


    # A duplicate key in the standalone path raises before any hashing runs.
def test_standalone_duplicate_member_is_terminal_before_hashing() -> None:
    text = '{"payload": {"type": "protectmcp:decision"}, "payload": {"type": "evil"}}'
    with pytest.raises(vr.DuplicateMemberError):
        json.loads(text, object_pairs_hook=vr._reject_duplicate_members)


    # The OTel GenAI door recovers a receipt only when its JSON is duplicate-free.
def test_doors_otel_receipt_rejects_duplicate_members() -> None:
    from asqav import doors

    attrs = {doors.OTEL_RECEIPT_ATTR: '{"a": 1, "a": 2}'}
    with pytest.raises(strict_json.DuplicateMemberError):
        doors.receipt_from_otel_genai_attributes(attrs)


    # Both corpus duplicate-member vectors are terminal parse failures.
@pytest.mark.parametrize("vec", ["asqav-11-dup-member-toplevel", "asqav-13-dup-member-nested"])
def test_corpus_duplicate_member_vectors_never_parse(vec: str) -> None:
    raw = (_CORPUS / vec / "receipt.json").read_text()
    with pytest.raises(strict_json.DuplicateMemberError):
        strict_json.loads(raw)


    # A mutated receipt with a duplicated member never verifies through the runner.
@pytest.mark.parametrize("vec", ["asqav-11-dup-member-toplevel", "asqav-13-dup-member-nested"])
def test_corpus_duplicate_member_vectors_never_verify(vec: str) -> None:
    from asqav.verifier.oracle.runner import run_one

    outcome = run_one(_CORPUS / vec, "asqav-native", "unverified", "duplicate_member", "unverifiable")
    assert outcome.ok
    assert outcome.actual_verdict == "unverified"
    assert outcome.actual_failure_class == "unverifiable"
    assert "terminal parse failure before any hashing" in outcome.detail


# --- Integers outside +/-2**53, the second thing the strict door refuses (finding 8) ---


    # An integer with no exact double would canonicalise two ways across the SDKs.
@pytest.mark.parametrize(
    "literal",
    ["9007199254740993", "-9007199254740993", "1000000000000000000000"],
)
def test_integer_outside_the_canonical_range_rejected(literal: str) -> None:
    with pytest.raises(strict_json.UnsafeIntegerError):
        strict_json.loads('{"n": %s}' % literal)


    # 2**53 itself is exactly representable and is pinned canonical by the upstream corpus.
@pytest.mark.parametrize(
    "literal",
    ["9007199254740991", "9007199254740992", "-9007199254740992", "0"],
)
def test_integer_inside_the_canonical_range_accepted(literal: str) -> None:
    assert strict_json.loads('{"n": %s}' % literal) == {"n": int(literal)}


    # The conformant workaround the draft's section 4 tells callers to use.
def test_the_same_value_as_a_json_string_is_accepted() -> None:
    assert strict_json.loads('{"n": "9007199254740993"}') == {"n": "9007199254740993"}


    # Nesting depth does not matter; the parse hook sees every literal.
def test_nested_unsafe_integer_rejected() -> None:
    with pytest.raises(strict_json.UnsafeIntegerError):
        strict_json.loads('{"a": {"b": [1, 2, {"c": 9007199254740993}]}}')


    # The standalone verifier ships as one file and carries its own copy of the hook.
def test_standalone_verifier_rejects_an_unsafe_integer() -> None:
    with pytest.raises(vr.VerifierInputError, match="canonical integer range"):
        vr._parse_object('{"n": 9007199254740993}', "receipt")
    assert vr._parse_object('{"n": 9007199254740992}', "receipt") == {"n": 9007199254740992}
