"""The vector-to-requirement mapping is current, derived, and publishes its gaps.

A mapping that drifts from the corpus is worse than none, because it reads as a
coverage claim while describing a corpus that no longer exists. And a mapping
without its unmapped half can be made to look complete by growing the corpus,
which is the failure mode the second half exists to prevent.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAP = ROOT / "verifier" / "conformance-vectors" / "requirement-map.json"
GENERATOR = ROOT / "verifier" / "build_requirement_map.py"


@pytest.fixture(scope="module")
def document():
    assert MAP.exists(), f"{MAP} is not published"
    return json.loads(MAP.read_text())


def _builder():
    spec = importlib.util.spec_from_file_location("build_requirement_map", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _builder()


def _assert_axis_scope_matches(corpus: pathlib.Path, document: dict) -> None:
    """Each manifest-native vector declares the map's exercised/not split."""
    manifest = json.loads((corpus / "manifest.json").read_text())
    native = [e["dir"] for e in manifest if e["format"] == BUILDER.NATIVE_FORMAT]
    assert len(native) == len(document["vectors"])
    for name in native:
        vec = document["vectors"][name]
        expected = json.loads((corpus / name / "expected.json").read_text())
        assert BUILDER.AXIS_SCOPE_KEY in expected, name
        assert expected[BUILDER.AXIS_SCOPE_KEY] == {
            BUILDER.AXIS_SCOPE_EXERCISED: vec["exercises"],
            BUILDER.AXIS_SCOPE_NOT_APPLICABLE: vec["not_exercised"],
        }, name


def test_axis_scope_matches_requirement_map(document):
    """A vector's declared axis scope is the one the verifier run derived."""
    _assert_axis_scope_matches(BUILDER.VECTORS, document)


def test_axis_scope_declaring_an_exercised_axis_not_applicable_fails(document, tmp_path):
    """NEGCTL: asqav-24 waiving REQ-ANCHOR, which it exercises, is drift."""
    keep = {"manifest.json", "expected.json"}
    corpus = tmp_path / BUILDER.VECTORS.name
    shutil.copytree(
        BUILDER.VECTORS,
        corpus,
        ignore=lambda d, names: [n for n in names if n not in keep and "." in n],
    )
    _assert_axis_scope_matches(corpus, document)
    name, req = "asqav-24-anchor-block-hash-prod", "REQ-ANCHOR"
    assert req in BUILDER.REQUIREMENTS
    assert req in document["vectors"][name]["exercises"]
    path = corpus / name / "expected.json"
    expected = json.loads(path.read_text())
    expected[BUILDER.AXIS_SCOPE_KEY][BUILDER.AXIS_SCOPE_NOT_APPLICABLE].append(req)
    path.write_text(json.dumps(expected))
    with pytest.raises(AssertionError, match=name):
        _assert_axis_scope_matches(corpus, document)


def _assert_verdict_scope_and_anchor_flag(document: dict) -> None:
    """Each verdict names its axis scope; no vector claims trusted anchors."""
    for name, vec in document["vectors"].items():
        scope = vec[BUILDER.OBSERVED_VERDICT_SCOPE_KEY]
        assert scope == BUILDER.OBSERVED_VERDICT_SCOPE, name
        assert vec[BUILDER.TRUSTED_ANCHOR_MATERIAL_KEY] is False, name


def test_observed_verdict_axis_scoped_and_anchor_material_flag(document):
    """asqav-24 ships anchor material beside its receipt, which is not trust."""
    _assert_verdict_scope_and_anchor_flag(document)
    vec = document["vectors"]["asqav-24-anchor-block-hash-prod"]
    assert vec[BUILDER.CORPUS_ANCHOR_MATERIAL_KEY]
    assert vec[BUILDER.TRUSTED_ANCHOR_MATERIAL_KEY] is False


def test_trusted_flag_or_unscoped_verdict_fails_the_check(document):
    """NEGCTL: a True anchor flag or a dropped scope label is caught."""
    name = "asqav-24-anchor-block-hash-prod"
    flagged = copy.deepcopy(document)
    flagged["vectors"][name][BUILDER.TRUSTED_ANCHOR_MATERIAL_KEY] = True
    with pytest.raises(AssertionError, match=name):
        _assert_verdict_scope_and_anchor_flag(flagged)
    unscoped = copy.deepcopy(document)
    unscoped["vectors"][name][BUILDER.OBSERVED_VERDICT_SCOPE_KEY] = "full-profile"
    with pytest.raises(AssertionError, match=name):
        _assert_verdict_scope_and_anchor_flag(unscoped)


def test_regenerating_reproduces_the_published_map(document, tmp_path):
    """The committed map is what the generator produces from today's corpus."""
    before = MAP.read_text()
    declared = {p: p.read_text() for p in BUILDER.VECTORS.glob("*/expected.json")}
    result = subprocess.run(
        [sys.executable, str(GENERATOR)], capture_output=True, text=True, cwd=ROOT
    )
    rewritten = {p: text for p, text in declared.items() if p.read_text() != text}
    for path, text in rewritten.items():
        path.write_text(text)
    assert not rewritten, f"stale axis_scope in {sorted(p.parent.name for p in rewritten)}"
    assert result.returncode == 0, result.stderr
    assert "axis_scope rewritten : 0 " in result.stdout
    after = MAP.read_text()
    if after != before:
        MAP.write_text(before)
        pytest.fail(
            "requirement-map.json is stale: the corpus moved without the map being "
            "regenerated. Run python3 verifier/build_requirement_map.py and commit."
        )


def _assert_verified_vectors_exercise_signature(document: dict) -> None:
    """A verified native vector never waives the signature check."""
    for name, vec in document["vectors"].items():
        if vec.get("declared_outcome") in _renderer()._VERIFIED_OUTCOMES:
            assert BUILDER.SIGNATURE_REQUIREMENT in vec["exercises"], name


def test_verified_native_vectors_exercise_the_signature(document):
    """asqav-52's flat hash-mode receipt is read off the oracle verifying it."""
    _assert_verified_vectors_exercise_signature(document)
    vec = document["vectors"]["asqav-52-outer-flat-hash-unsigned-members"]
    assert vec[BUILDER.DERIVED_BY_KEY] == BUILDER.DERIVED_BY_ORACLE
    outer = "REQ-OUTER-COPIES"
    assert outer in BUILDER.REQUIREMENTS
    assert outer in vec["exercises"]


def test_verified_vector_waiving_the_signature_fails_the_check(document):
    """NEGCTL: asqav-52 with REQ-SIGNATURE moved to not_exercised is refused."""
    name = "asqav-52-outer-flat-hash-unsigned-members"
    waived = copy.deepcopy(document)
    waived["vectors"][name]["exercises"].remove(BUILDER.SIGNATURE_REQUIREMENT)
    with pytest.raises(AssertionError, match=name):
        _assert_verified_vectors_exercise_signature(waived)


def test_the_unmapped_list_is_published_even_when_it_is_empty(document):
    """The key must exist; an absent key would read as 'no gaps' by omission."""
    assert "unmapped_requirements" in document
    assert isinstance(document["unmapped_requirements"], list)
    assert "unmapped_note" in document


def test_every_unmapped_requirement_says_why(document):
    """A gap without a reason is not actionable, so it does not count as declared."""
    for req in document["unmapped_requirements"]:
        detail = document["unmapped_detail"][req]
        assert detail["axis"]
        assert detail["results_seen_across_vectors"], req
        # Whatever the reason, no vector may have exercised it, or it is not unmapped.
        assert "PASS" not in detail["results_seen_across_vectors"], req
        assert "FAIL" not in detail["results_seen_across_vectors"], req


def test_coverage_and_unmapped_partition_the_requirement_set(document):
    """No requirement may be silently absent from both halves."""
    requirements = set(document["requirements"])
    covered = {r for r, hits in document["coverage"].items() if hits}
    unmapped = set(document["unmapped_requirements"])
    assert covered | unmapped == requirements
    assert not (covered & unmapped)


def test_interop_fixtures_are_not_counted_as_profile_coverage(document):
    """Another specification's fixture is not evidence about this profile."""
    for name, entry in document["interop_fixtures"].items():
        assert entry["exercises_profile_requirements"] is False, name
    for req, hits in document["coverage"].items():
        for vector in hits:
            assert vector in document["vectors"], f"{req} credits a non-asqav vector {vector}"


def test_seq_chain_vector_chain_axis_is_pass_in_the_committed_map(document):
    """asqav-17's chain link rederives, and the published axis evidence says so.

    The builder hands run_structured the predecessor's payload, not its
    envelope; a chain FAIL in this entry would mean the map derived REQ-CHAIN
    coverage from a false chain break over the envelope bytes.
    """
    axes = document["axis_results"]["asqav-17-seq-contiguous"]
    assert axes["chain"] == "PASS"


def test_anchor_material_vector_exercises_req_anchor(document):
    """asqav-24 ships public anchor material, so the committed map shows the axis PASS.

    The builder hands run_structured the vector's tsa_trust.pem and
    bitcoin_headers.json; without that pass-through the anchors axis SKIPs and
    REQ-ANCHOR stays unmapped.
    """
    axes = document["axis_results"]["asqav-24-anchor-block-hash-prod"]
    assert axes["anchors"] == "PASS"
    assert document["coverage"]["REQ-ANCHOR"] == [
        "asqav-24-anchor-block-hash-prod", "asqav-31-counterparty-scope-match",
        "asqav-32-counterparty-anchors-included", "asqav-33-counterparty-scope-absent",
        "asqav-34-counterparty-scope-unknown", "asqav-35-counterparty-peer-swapped",
        "asqav-36-counterparty-origin-forged", "asqav-37-counterparty-divergent-terms",
        "asqav-38-counterparty-received-only", "asqav-39-counterparty-cross-replay",
        "asqav-40-counterparty-self-binding", "asqav-41-counterparty-omitted-peer",
        "asqav-42-counterparty-same-binding-second-ack", "asqav-43-counterparty-unknown-member",
    ]
    assert document["unmapped_requirements"] == []


README = ROOT / "verifier" / "conformance-vectors" / "README.md"
RENDERER = ROOT / "verifier" / "render_vector_list.py"
#: The claim asqav-24's notes made while the corpus shipped the material.
NO_ANCHOR_MATERIAL_CLAIM = "does not ship"
ANCHORED_VECTOR = "asqav-24-anchor-block-hash-prod"


def _renderer():
    spec = importlib.util.spec_from_file_location("render_vector_list", RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_readme_rows_match_map(readme_text: str, document: dict) -> int:
    """Return the scoped row count; a bare-verified row needs full coverage."""
    verified = _renderer()._VERIFIED_OUTCOMES
    scoped = 0
    for line in readme_text.splitlines():
        row = re.match(r"- `([^`]+)`: (\S+)", line)
        if not row or row.group(1) not in document["vectors"]:
            continue
        name, word = row.groups()
        if word in verified:
            vec = document["vectors"][name]
            assert not vec["not_exercised"], name
            corpus_only = not vec[BUILDER.TRUSTED_ANCHOR_MATERIAL_KEY]
            assert not (vec[BUILDER.CORPUS_ANCHOR_MATERIAL_KEY] and corpus_only), name
        elif line.startswith(f"- `{name}`: {_renderer().AXIS_SCOPED_LABEL} "):
            scoped += 1
    return scoped


def test_corpus_readme_states_axis_scope(document):
    """No partial vector reads bare verified; asqav-24 claims no gap."""
    text = README.read_text()
    assert _assert_readme_rows_match_map(text, document) > 0
    assert _renderer().AXIS_SCOPED_LABEL in text
    manifest = json.loads((BUILDER.VECTORS / "manifest.json").read_text())
    notes = next(e["notes"] for e in manifest if e["dir"] == ANCHORED_VECTOR)
    expected = json.loads((BUILDER.VECTORS / ANCHORED_VECTOR / "expected.json").read_text())
    material = document["vectors"][ANCHORED_VECTOR][BUILDER.CORPUS_ANCHOR_MATERIAL_KEY]
    for claim_text in (notes, expected["notes"]):
        assert NO_ANCHOR_MATERIAL_CLAIM not in claim_text
        assert all(name in claim_text for name in material), claim_text
        assert BUILDER.TRUSTED_ANCHOR_MATERIAL_KEY in claim_text
    assert f"- `{ANCHORED_VECTOR}`: {_renderer().AXIS_SCOPED_LABEL} " in text


def test_bare_verified_row_on_a_partial_vector_fails_the_readme_check(document):
    """NEGCTL: asqav-01 rewritten to bare verified is refused."""
    text = README.read_text()
    renderer = _renderer()
    name = "asqav-01-genesis-permit"
    assert document["vectors"][name]["not_exercised"]
    scoped_row = next(r for r in text.splitlines() if r.startswith(f"- `{name}`: "))
    assert renderer.AXIS_SCOPED_LABEL in scoped_row
    bare = text.replace(scoped_row, f"- `{name}`: verified")
    with pytest.raises(AssertionError, match=name):
        _assert_readme_rows_match_map(bare, document)
