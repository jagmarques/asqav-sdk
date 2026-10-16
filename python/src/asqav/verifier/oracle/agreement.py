"""Agreement/predicate reference and comparanda comparison (488 C12, C29).

One of C5's seven comparanda has no object anywhere: the cross-organization
agreement/predicate version. Decision (recorded 2026-10-16, reversible):
reference terms by id + version + sha256 of canonical terms bytes; never
embedded, never minting authority. The compare step reports per-comparandum
equality only; classification stays a later slice.

C29 binds the verified envelope: AgreementRef carries the kid + signature
the bytes were verified under, derived from those bytes through the adapter
seam (never caller-stated). A stated envelope that contradicts the bytes
raises instead of classifying (the C24 digest rule, applied to the envelope).
Terms stay caller-attached: the cross-org terms object does not exist yet."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _canonical(obj: object) -> bytes:
    # ensure_ascii=False: UTF-8 bytes match TypeScript JSON.stringify output,
    # so non-ASCII terms hash identically in both languages.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class AgreementRef:
    """Pointer to the predicate terms both sides claim to apply.

    kid/signature name the envelope the bytes were verified under; they
    arrive only through envelope_of/bind_envelope (bytes-derived), never
    from of_terms (terms stay caller-attached)."""

    agreement_id: str | None = None
    predicate_version: str | None = None
    terms_hash: str | None = None
    kid: str | None = None
    signature: str | None = None

    @classmethod
    def of_terms(cls, agreement_id: str, predicate_version: str, terms: object) -> "AgreementRef":
        """Bind canonical terms bytes; hash is sha256 hex."""
        digest = hashlib.sha256(_canonical(terms)).hexdigest()
        return cls(agreement_id=agreement_id, predicate_version=predicate_version, terms_hash=digest)


def _norm_agreement(agreement: object | None) -> dict:
    """Normalize an AgreementRef, bilingual dict, or None to plain fields."""
    if agreement is None:
        return {"agreement_id": None, "predicate_version": None,
                "terms_hash": None, "kid": None, "signature": None}
    if isinstance(agreement, dict):
        # Bilingual keys: agreements cross languages as JSON.
        # kid/signature spell the same in both; keep the lookup shape.
        return {
            "agreement_id": agreement.get("agreement_id", agreement.get("agreementId")),
            "predicate_version": agreement.get("predicate_version", agreement.get("predicateVersion")),
            "terms_hash": agreement.get("terms_hash", agreement.get("termsHash")),
            "kid": agreement.get("kid"),
            "signature": agreement.get("signature"),
        }
    return {
        "agreement_id": agreement.agreement_id,
        "predicate_version": agreement.predicate_version,
        "terms_hash": agreement.terms_hash,
        "kid": agreement.kid,
        "signature": agreement.signature,
    }


def envelope_of(doc: dict, adapters: list[Any]) -> AgreementRef:
    """Derive the verified envelope from parsed receipt bytes (488 C29).

    Dispatches through the adapter seam (detect + extract_signature), so
    the kid/signature are read from the bytes being verified, not stated.
    Fail-closed: no detecting adapter or a malformed envelope raises."""
    from asqav.verifier.oracle.core import detect

    adapter = detect(doc, adapters)
    if adapter is None:
        raise ValueError("no adapter detects this receipt; envelope unbound")
    try:
        material = adapter.extract_signature(doc)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"envelope extraction failed: {exc}") from exc
    return AgreementRef(kid=material.kid, signature=material.sig.hex())


def bind_envelope(agreement: object | None, doc: dict, adapters: list[Any]) -> AgreementRef:
    """Merge caller terms with the bytes-derived envelope (488 C29).

    Terms (id/version/hash) stay caller-attached; kid/signature always
    come from the bytes. A stated envelope contradicting the bytes raises
    instead of classifying; a matching or absent stated envelope is fine."""
    derived = envelope_of(doc, adapters)
    stated = _norm_agreement(agreement)
    for field in ("kid", "signature"):
        if stated[field] is not None and stated[field] != getattr(derived, field):
            raise ValueError(f"stated {field} contradicts the verified envelope")
    return AgreementRef(
        agreement_id=stated["agreement_id"],
        predicate_version=stated["predicate_version"],
        terms_hash=stated["terms_hash"],
        kid=derived.kid,
        signature=derived.signature,
    )


COMPARANDA = (
    "evidence_bytes",
    "agreement_version",
    "content_hashes",
    "evaluator",
    "trust_roots",
    "revocation_snapshot",
    "verification_time",
)


def compare_assessments(first: dict, second: dict) -> dict[str, bool]:
    """Per-comparandum equality for two retained assessment views.

    Each view: {"doc_sha256", "axes", "context": {"evaluator",
    "trust_roots", "revocation_snapshot", "evaluation_time" or snake/camel
    equivalents}, "agreement": AgreementRef | dict | None}."""
    f_ctx, s_ctx = first.get("context") or {}, second.get("context") or {}

    def _get(ctx: dict, snake: str, camel: str) -> object:
        # Absent and None read the same: fall back like TS `??`, so an
        # explicit null never shadows the sibling key in either language.
        value = ctx.get(snake)
        return ctx.get(camel) if value is None else value

    def _agr(view: dict) -> dict:
        return _norm_agreement(view.get("agreement"))

    f_agr, s_agr = _agr(first), _agr(second)
    f_roots = sorted(_get(f_ctx, "trust_roots", "trustRoots") or [])
    s_roots = sorted(_get(s_ctx, "trust_roots", "trustRoots") or [])
    f_axes = hashlib.sha256(_canonical(first.get("axes") or {})).hexdigest()
    s_axes = hashlib.sha256(_canonical(second.get("axes") or {})).hexdigest()
    return {
        "evidence_bytes": first.get("doc_sha256") == second.get("doc_sha256"),
        "agreement_version": (
            f_agr.get("agreement_id") == s_agr.get("agreement_id")
            and f_agr.get("predicate_version") == s_agr.get("predicate_version")
            and f_agr.get("terms_hash") == s_agr.get("terms_hash")
            and f_agr.get("kid") == s_agr.get("kid")
            and f_agr.get("signature") == s_agr.get("signature")
        ),
        "content_hashes": f_axes == s_axes,
        "evaluator": _get(f_ctx, "evaluator", "evaluator")
        == _get(s_ctx, "evaluator", "evaluator"),
        "trust_roots": f_roots == s_roots,
        "revocation_snapshot": _get(f_ctx, "revocation_snapshot", "revocationSnapshot")
        == _get(s_ctx, "revocation_snapshot", "revocationSnapshot"),
        "verification_time": _get(f_ctx, "evaluation_time", "evaluationTime")
        == _get(s_ctx, "evaluation_time", "evaluationTime"),
    }
