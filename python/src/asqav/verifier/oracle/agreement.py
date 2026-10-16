"""Agreement/predicate reference and comparanda comparison (488 C12).

One of C5's seven comparanda has no object anywhere: the cross-organization
agreement/predicate version. Decision (recorded 2026-10-16, reversible):
reference terms by id + version + sha256 of canonical terms bytes; never
embedded, never minting authority. The compare step reports per-comparandum
equality only; classification stays a later slice."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class AgreementRef:
    """Pointer to the predicate terms both sides claim to apply."""

    agreement_id: str | None = None
    predicate_version: str | None = None
    terms_hash: str | None = None

    @classmethod
    def of_terms(cls, agreement_id: str, predicate_version: str, terms: object) -> "AgreementRef":
        """Bind canonical terms bytes; hash is sha256 hex."""
        digest = hashlib.sha256(_canonical(terms)).hexdigest()
        return cls(agreement_id=agreement_id, predicate_version=predicate_version, terms_hash=digest)


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
        return ctx.get(snake, ctx.get(camel))

    def _agr(view: dict) -> dict:
        agr = view.get("agreement")
        if agr is None:
            return {}
        return agr if isinstance(agr, dict) else {
            "agreement_id": agr.agreement_id,
            "predicate_version": agr.predicate_version,
            "terms_hash": agr.terms_hash,
        }

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
