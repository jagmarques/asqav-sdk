"""Agreement comparanda (488 C12, C29): all-match, single-dimension flips, none.

C29 binds the verified envelope: kid/signature derive from the receipt bytes
through the adapter seam (envelope_of), never caller-stated; compare enforces
them inside agreement_version; bind_envelope merges caller terms with the
bytes-derived envelope and rejects a contradicting stated envelope."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from asqav.verifier.oracle import ADAPTERS
from asqav.verifier.oracle.agreement import (
    COMPARANDA,
    AgreementRef,
    bind_envelope,
    compare_assessments,
    envelope_of,
)

_CTX = {
    "evaluator": "c12-probe",
    "trust_roots": ["root-a", "root-b"],
    "revocation_snapshot": "snap-1",
    "evaluation_time": "2029-06-01T00:00:00Z",
}
_AXES = {"expiry": "PASS", "skew": "PASS"}
_AGR = AgreementRef.of_terms("agr-1", "2029-01", {"clause": "pay on delivery"})


def _view(**over: object) -> dict:
    base: dict = {
        "doc_sha256": "dd" * 32,
        "axes": dict(_AXES),
        "context": dict(_CTX),
        "agreement": _AGR,
    }
    for key, value in over.items():
        if key == "context":
            base["context"] = {**_CTX, **value}  # type: ignore[dict-item]
        else:
            base[key] = value
    return base


def test_identical_views_match_on_all_seven() -> None:
    result = compare_assessments(_view(), _view())
    assert result == {name: True for name in COMPARANDA}


def test_terms_change_flips_only_agreement_version() -> None:
    other = _view(agreement=AgreementRef.of_terms("agr-1", "2029-01", {"clause": "pay upfront"}))
    result = compare_assessments(_view(), other)
    assert result["agreement_version"] is False
    assert all(v for k, v in result.items() if k != "agreement_version")


def test_later_evaluation_time_flips_only_verification_time() -> None:
    other = _view(context={"evaluation_time": "2031-01-01T00:00:00Z"})
    result = compare_assessments(_view(), other)
    assert result["verification_time"] is False
    assert all(v for k, v in result.items() if k != "verification_time")


def test_missing_agreement_is_a_mismatch_not_a_crash() -> None:
    result = compare_assessments(_view(agreement=None), _view())
    assert result["agreement_version"] is False
    assert result["evidence_bytes"] is True


def test_reads_camel_case_contexts_from_the_other_language() -> None:
    snake_view = _view()
    snake_view["context"] = {
        "evaluator": "c12-probe",
        "trustRoots": ["root-b", "root-a"],
        "revocationSnapshot": "snap-1",
        "evaluationTime": "2029-06-01T00:00:00Z",
    }
    assert compare_assessments(_view(), snake_view) == {name: True for name in COMPARANDA}


_NON_ASCII_DIGEST = "3a357c459fc615cbc788440180f136108db0051615defa7f819bf38b8b72f3f7"


def test_non_ascii_terms_hash_matches_the_other_language() -> None:
    ref = AgreementRef.of_terms("agr-1", "2029-01", {"clause": "páy on delivery ✓"})
    assert ref.terms_hash == _NON_ASCII_DIGEST


def test_cross_casing_agreement_dicts_compare_by_value() -> None:
    camel = {"agreementId": "agr-1", "predicateVersion": "2029-01", "termsHash": _AGR.terms_hash}
    same = compare_assessments(_view(), _view(agreement=camel))
    assert same["agreement_version"] is True
    different = dict(camel, termsHash="DIFFERENT")
    assert compare_assessments(_view(), _view(agreement=different))["agreement_version"] is False


def test_both_absent_agreements_match() -> None:
    assert compare_assessments(_view(agreement=None), _view(agreement=None))[
        "agreement_version"
    ] is True


_FIX = Path(__file__).resolve().parents[2] / "verifier" / "signed-conflict-fixtures"
_CORPUS = Path(__file__).resolve().parents[2] / "verifier" / "conformance-vectors"

_ALLOW_KID = "did:key:z6MktqPnbGGsq5TR4h9dYRWtkoSQtiZt9CXfUFNcfxA7Fi5x#z6MktqPnbGGsq5TR4h9dYRWtkoSQtiZt9CXfUFNcfxA7Fi5x"
_ALLOW_SIG = "e7ef246a853f05b7731bec610f91481a147589e751ef642d15b4f3c5c32062eddf1277413023aa9b677b293d751e81722e356df2eeb211d1eaaa3d0f8421c904"
_DENY_SIG = "831d9f99e3fdd33580257453aaea01ac1a5b677cd96955886aaa6da812b47408d904d2b3a51c96ba7de4611c7b113ad7b702aa8142652f8f80b8770ff010c006"
_GENESIS_KID = "asqav-oracle-vec-key"
_GENESIS_SIG = "332a20933137e4daa5e4e23830e95631cc7facbb04a1d8bb5013bc358a189e531df6cdfbac191fe814c206f28b7261d71205cfae904d21bcf265f2c130f8d903"


def _doc(name: str) -> dict:
    return json.loads((_FIX / f"{name}.json").read_bytes())


def _genesis_doc() -> dict:
    return json.loads((_CORPUS / "asqav-01-genesis-permit" / "receipt.json").read_bytes())


def test_envelope_of_reads_kid_and_signature_from_bytes() -> None:
    allow = envelope_of(_doc("conflict-allow"), ADAPTERS)
    assert (allow.kid, allow.signature) == (_ALLOW_KID, _ALLOW_SIG)
    assert (allow.agreement_id, allow.predicate_version, allow.terms_hash) == (None, None, None)
    genesis = envelope_of(_genesis_doc(), ADAPTERS)
    assert (genesis.kid, genesis.signature) == (_GENESIS_KID, _GENESIS_SIG)


def test_envelope_of_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        envelope_of({"hello": "world"}, ADAPTERS)


def _env_view(doc_sha: str, agreement: object) -> dict:
    return {"doc_sha256": doc_sha, "axes": dict(_AXES), "context": dict(_CTX),
            "agreement": agreement}


def test_kid_flip_breaks_only_agreement_version() -> None:
    same, other = envelope_of(_doc("conflict-allow"), ADAPTERS), envelope_of(
        _genesis_doc(), ADAPTERS)
    result = compare_assessments(_env_view("ab" * 32, same), _env_view("ab" * 32, other))
    assert result["agreement_version"] is False
    assert all(v for k, v in result.items() if k != "agreement_version")


def test_signature_flip_breaks_only_agreement_version() -> None:
    # Same principal (same kid), conflicting proof (different signature).
    allow = envelope_of(_doc("conflict-allow"), ADAPTERS)
    deny = envelope_of(_doc("conflict-deny"), ADAPTERS)
    assert allow.kid == deny.kid and allow.signature != deny.signature
    result = compare_assessments(_env_view("ab" * 32, allow), _env_view("ab" * 32, deny))
    assert result["agreement_version"] is False
    assert all(v for k, v in result.items() if k != "agreement_version")


def test_bind_envelope_merges_terms_with_bytes_envelope() -> None:
    terms = AgreementRef.of_terms("agr-1", "2029-01", {"clause": "pay on delivery"})
    bound = bind_envelope(terms, _doc("conflict-allow"), ADAPTERS)
    assert (bound.agreement_id, bound.predicate_version, bound.terms_hash) == (
        terms.agreement_id, terms.predicate_version, terms.terms_hash)
    assert (bound.kid, bound.signature) == (_ALLOW_KID, _ALLOW_SIG)
    bare = bind_envelope(None, _doc("conflict-allow"), ADAPTERS)
    assert (bare.agreement_id, bare.terms_hash, bare.kid) == (None, None, _ALLOW_KID)


def test_bind_envelope_rejects_forged_envelope() -> None:
    forged = {"kid": "attacker-key", "signature": _ALLOW_SIG}
    with pytest.raises(ValueError):
        bind_envelope(forged, _doc("conflict-allow"), ADAPTERS)
    forged_sig = {"kid": _ALLOW_KID, "signature": "00" * 64}
    with pytest.raises(ValueError):
        bind_envelope(forged_sig, _doc("conflict-allow"), ADAPTERS)
    # A matching stated envelope is redundant but harmless.
    honest = {"kid": _ALLOW_KID, "signature": _ALLOW_SIG}
    assert bind_envelope(honest, _doc("conflict-allow"), ADAPTERS).kid == _ALLOW_KID
