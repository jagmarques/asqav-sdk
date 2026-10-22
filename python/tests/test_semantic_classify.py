"""Failure classification for the oracle (574 D5a). ``oracle.semantic_classify`` derives each class from 418
(invalid = resolved key + binding failed; unverifiable = recompute could not complete) plus the axis contract
proving FAIL implies resolved inputs. These tests pin it: literals match the core surface; always-invalid axes
map FAIL to invalid; note-dependent branches and the default map as 418 requires; corpus vectors keep their
verdicts on the new path. A flat port (all FAILs invalid, or all unverifiable) fails them.
"""
from __future__ import annotations

import pytest

from asqav.verifier.oracle import crypto
from asqav.verifier.oracle import semantic_classify as sc
from asqav.verifier.oracle.core import FAILURE_INVALID, FAILURE_UNVERIFIABLE
from asqav.verifier.oracle.runner import run_corpus

_CORPUS = __import__("pathlib").Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"


def test_literals_match_the_core_surface() -> None:
    assert sc.FAILURE_INVALID == FAILURE_INVALID
    assert sc.FAILURE_UNVERIFIABLE == FAILURE_UNVERIFIABLE


@pytest.mark.parametrize(
    "axis",
    sorted(
        {
            "signature",
            "key_status",
            "issuer_bind",
            "key_binding",
            "counterparty",
            "payload_digest",
            "outer_copies",
            "nonce",
            "seq",
            "parent_signature",
            "pdp_signature",
            "anchors",
        }
    ),
)
def test_proven_binding_failures_are_invalid(axis: str) -> None:
    assert sc.classify(axis, crypto.FAIL, "proven break") == FAILURE_INVALID


@pytest.mark.parametrize(
    ("axis", "note", "failure_class"),
    [
        ("chain", "chain break: expected ab.. got cd..", FAILURE_INVALID),
        ("chain", "predecessor is a different receipt format", FAILURE_INVALID),
        ("chain", "predecessor payload not canonicalisable", FAILURE_UNVERIFIABLE),
        ("skew", "unparseable issued_at 'x'", FAILURE_UNVERIFIABLE),
        ("skew", "issued_at 9999s ahead of wall clock (> 300s)", FAILURE_INVALID),
        ("structure", "unsupported ACTA alg ed25519", FAILURE_INVALID),
        ("structure", "key_purpose mismatch", FAILURE_INVALID),
        ("structure", "signing-key DID != issuer DID", FAILURE_INVALID),
        ("structure", "proof @context is not a prefix", FAILURE_INVALID),
        ("structure", "unregistered type foo", FAILURE_INVALID),
        ("structure", "malformed type 42", FAILURE_INVALID),
        ("structure", "missing required fields: sig", FAILURE_UNVERIFIABLE),
        ("expiry", "unreadable expires_at 'x'", FAILURE_UNVERIFIABLE),
        ("expiry", "expires_at 2020-01-01 lapsed 5s ago", FAILURE_INVALID),
        ("issuer_key", "anything at all", FAILURE_UNVERIFIABLE),
        ("brand_new_axis", "anything at all", FAILURE_UNVERIFIABLE),
    ],
)
def test_note_branches_and_default(axis: str, note: str, failure_class: str) -> None:
    assert sc.classify(axis, crypto.FAIL, note) == failure_class


@pytest.mark.parametrize("result", [crypto.PASS, crypto.SKIPPED])
def test_pass_and_skip_never_prove_a_break(result: str) -> None:
    assert sc.classify("signature", result, "any") == (
        None if result == crypto.PASS else FAILURE_UNVERIFIABLE
    )


def test_corpus_verdicts_preserved_through_the_new_path() -> None:
    from asqav.verifier.oracle.runner import _tolerated as _runner_tolerated

    results = run_corpus(_CORPUS)
    bad = [r.dir for r in results if not _runner_tolerated(r)]
    assert not bad, f"regressed vectors: {bad}"
    assert len(results) == 118


def test_fail_implies_resolved_inputs_for_gated_axes() -> None:
    # Axes that cannot resolve inputs SKIP upstream, so a FAIL reaching
    # the table ran against resolved inputs (axis code proves each case).
    assert sc.classify("signature", crypto.FAIL, "signature verification FAILED") == FAILURE_INVALID
    assert sc.classify("signature", crypto.SKIPPED, "no key: unknown kid") == FAILURE_UNVERIFIABLE
    assert sc.classify("key_binding", crypto.SKIPPED, "no signing key resolved") == FAILURE_UNVERIFIABLE
    assert sc.classify("counterparty", crypto.SKIPPED, "binding not resolvable offline") == FAILURE_UNVERIFIABLE
