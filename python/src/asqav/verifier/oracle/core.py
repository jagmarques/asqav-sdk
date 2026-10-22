"""The shared verification core - format detection, dispatch, and the verdict.
``verify(doc, ...)`` detects the format, then drives the adapter through the shared
axes: structure, signature, chain. It proves only what the bytes prove - a valid
signature over the canonical bytes, a reproducible chain link, and structural
presence at time T - never the behaviour or correctness of the recorded action."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from asqav import strict_json

from ..verify_receipt import NOT_CHECKED, not_checked_declaration
from . import crypto
from .adapter import FormatAdapter, VerificationContext
from .semantic import fold_independent

# Failure classification lives in ``semantic_classify`` (criterion 574
# D5a): derived from criterion 418, not ported from another table.
from .semantic_classify import classify as classify_failure

#: Public verdict vocabulary (criteria 418/438). The per-axis PASS/FAIL/SKIPPED
#: tokens stay internal; the surface a caller reads speaks these three only.
VERDICT_VERIFIED = "verified"
VERDICT_VERIFIED_KEYED = "verified_keyed"
VERDICT_UNVERIFIED = "unverified"

#: Failure classes carried by every unverified verdict (criterion 418); the two
#: are never collapsed - a proven binding failure is not an incomplete check.
FAILURE_INVALID = "invalid"
FAILURE_UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class AxisResult:
    """One verification axis outcome.
    Fields: axis (structure / signature / chain); result (PASS / FAIL / SKIPPED,
    internal token); note (human-readable report detail); failure_class (invalid /
    unverifiable for a non-PASS axis, None on PASS)."""

    axis: str
    result: str
    note: str
    failure_class: str | None = None


@dataclass
class VerifyResult:
    """The aggregate outcome of verifying one receipt.
    ``verdict`` is ``verified`` only when every non-skipped axis passed AND the
    signature was checked; ``verified_keyed`` when the digest is keyed (internally
    consistent, not third-party re-derivable). A skipped signature downgrades to
    ``unverified``/``unverifiable``, never verified, as in the standalone verifier.
    Expiry never folds the verdict (426); defaults fail closed (unfolded = unverified)."""

    fmt: str
    axes: list[AxisResult] = field(default_factory=list)
    verdict: str = VERDICT_UNVERIFIED
    failure_class: str | None = FAILURE_UNVERIFIABLE
    #: In-body origin attestation (v:2 ``signer``), surfaced from the signed
    #: payload. None when the receipt carries none (v:1). Never gates the verdict.
    signer: str | None = None
    #: Earliest axis in report order that did not PASS, else None. Reported so a
    #: disagreement on WHICH check failed first is as visible as on the verdict.
    first_failing_edge: str | None = None
    #: Resolved invocation context, so a disagreement attributes to an input.
    context: VerificationContext | None = None
    #: The checks this result does not claim, beside the claim it makes.
    not_checked: list = field(default_factory=list)
    #: Coverage block beside not_checked; snake_case keys mirror the TS twin.
    coverage: dict = field(default_factory=dict)

        # Return the result for one axis, or None if it was not run.
    def axis(self, name: str) -> AxisResult | None:
        return next((a for a in self.axes if a.axis == name), None)


#: The axis order verify() reports on the normal path, mirroring the TS twin.
ORACLE_AXIS_ORDER = (
    "structure", "signature", "chain", "seq", "expiry", "nonce",
    "key_binding", "counterparty", "payload_digest", "skew", "outer_copies",
    "key_status", "issuer_bind",
)


def _oracle_coverage(axis_names: list[str]) -> dict:
    """Coverage block for one result's axes; same shape as the TS twin."""
    entries = [
        {
            "id": row["check"],
            "reason": "not_implemented",
            "status": "not_implemented",
            "requirement": row["requirement"],
            "condition": row["condition"],
        }
        for row in NOT_CHECKED
    ]
    if axis_names and axis_names[-1] == ORACLE_AXIS_ORDER[-1]:
        return {"stopped_at": None, "checks_not_evaluated": entries}
    stopped_at = "structure"
    entries += [
        {"id": name, "reason": "not_reached", "status": "implemented"}
        for name in ORACLE_AXIS_ORDER[1:]
        if name not in axis_names
    ]
    return {"stopped_at": stopped_at, "checks_not_evaluated": entries}


#: Fixed leading axis order every adapter walks, before its format extras.
#: Pinned so a reorder fails a gate instead of renaming which edge is "first".
AXIS_ORDER_PREFIX = ("structure", "signature", "chain", "seq")


def first_failing_edge(axes: list[AxisResult]) -> str | None:
    """Name the earliest axis in report order that drove the verdict away from verified.
    Walks the axes in the fixed reported order, so the answer never depends on which
    check ran first. Exclusions mirror the semantic table: expiry never folds the
    verdict (426); a SKIPPED chain is tolerated where any other SKIPPED blocks, else
    an expired verified receipt would name an edge. A corpus-wide gate asserts None
    for exactly the verified verdicts. SKIPPED counts: an unchecked axis stopped the reasoning clean."""
    for a in axes:
        if a.result == crypto.FAIL and a.axis != "expiry":
            return a.axis
        if a.result == crypto.SKIPPED and a.axis != "chain":
            return a.axis
    return None


    # The verdict aggregation lives in ``semantic`` (criterion 574 D5):
    # independently derived from 418/438/426, not ported from this fold.
def detect(doc: dict, adapters: list[FormatAdapter]) -> FormatAdapter | None:
    if not isinstance(doc, dict):
        # A non-object receipt (array, string, number, null) matches no format;
        # the adapters assume a dict, so guard here rather than crash in detect().
        return None
    return next((a for a in adapters if a.detect(doc)), None)


def _axis(axis: str, result: str, note: str) -> AxisResult:
    return AxisResult(axis, result, note, classify_failure(axis, result, note))


#: Note for "anchors": null, malformed; absent or [] is the conformant spelling.
#: Verbatim in all three engines; the cross-language anchor-shape tests pin it.
_ANCHORS_NULL_NOTE = (
    "anchors is null: malformed; absent or [] is the conformant spelling of no anchors"
)


    # Adapter schema plus the anchors shape rule, which no schema sees because
    # anchors sits beside the signed payload.
def _structure_axis(ad: FormatAdapter, doc: dict) -> AxisResult:
    res, note = ad.schema(doc)
    if res == crypto.PASS and "anchors" in doc and doc["anchors"] is None:
        return _axis("structure", crypto.FAIL, _ANCHORS_NULL_NOTE)
    return _axis("structure", res, note)


def _signature_axis(ad: FormatAdapter, doc: dict, key_provider: Any) -> AxisResult:
    sm = ad.extract_signature(doc)
    pk, note = ad.resolve_key(doc, key_provider)
    if pk is None:
        return _axis("signature", crypto.SKIPPED, f"no key: {note}")
    msg = ad.signing_input(doc)
    res, why = crypto.verify_signature(sm.alg, pk, msg, sm.sig)
    return _axis("signature", res, why)


def _chain_axis(
    ad: FormatAdapter, doc: dict, adapters: list[FormatAdapter], predecessor: dict | None
) -> AxisResult:
    step = ad.chain_step(doc)
    if step.is_genesis:
        return _axis("chain", crypto.PASS, "genesis receipt (no predecessor link)")
    if predecessor is None:
        return _axis("chain", crypto.SKIPPED, "no predecessor supplied")
    # A chain link must stay in one format; a cross-format predecessor is invalid.
    pred_ad = detect(predecessor, adapters)
    if pred_ad is None or pred_ad.name != ad.name:
        return _axis("chain", crypto.FAIL, "predecessor is a different receipt format")
    actual = step.recompute(predecessor)
    if actual == step.prev_field:
        return _axis("chain", crypto.PASS, "chain link rederives from predecessor")
    exp = str(step.prev_field)[:16]
    return _axis("chain", crypto.FAIL, f"chain break: expected {exp}.. got {actual[:16]}..")


def _seq_axis(
    ad: FormatAdapter, doc: dict, adapters: list[FormatAdapter], predecessor: dict | None
) -> AxisResult:
    # A gap is omission evidence. Never SKIPPED: the semantic fold blocks on
    # a non-chain SKIPPED, so a receipt with no counter regresses.
    seq = ad.seq_of(doc)
    if seq is None:
        return _axis("seq", crypto.PASS, "no seq member; receipt is not part of a counted series")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
        return _axis("seq", crypto.FAIL, f"malformed seq: {seq!r}")
    if predecessor is None:
        return _axis("seq", crypto.PASS, f"seq {seq}; no predecessor supplied")
    # A counter only means anything within one format's own series.
    pred_ad = detect(predecessor, adapters)
    if pred_ad is None or pred_ad.name != ad.name:
        return _axis("seq", crypto.PASS, f"seq {seq}; predecessor is a different receipt format")
    prev = ad.seq_of(predecessor)
    if prev is None:
        return _axis("seq", crypto.PASS, f"seq {seq}; predecessor carries no seq")
    if isinstance(prev, bool) or not isinstance(prev, int) or prev < 1:
        return _axis("seq", crypto.FAIL, f"malformed predecessor seq: {prev!r}")
    if seq == prev + 1:
        return _axis("seq", crypto.PASS, f"seq {seq} follows predecessor {prev}")
    if seq <= prev:
        return _axis("seq", crypto.FAIL, f"seq not monotonic: {seq} after predecessor {prev}")
    return _axis(
        "seq",
        crypto.FAIL,
        f"seq gap: {seq - prev - 1} receipt(s) withheld between {prev} and {seq}",
    )


#: Max nesting the recursive JCS encoder tolerates, mirrors the standalone cap.
MAX_NESTING_DEPTH = 200

_TOO_DEEP_NOTE = f"receipt nesting exceeds the supported depth (> {MAX_NESTING_DEPTH} levels)"


def _exceeds_depth(obj: Any, max_depth: int) -> bool:
    """True when ``obj`` nests deeper than ``max_depth``, walked with an explicit stack.

    No recursion here, so the check itself never overflows before it can cap a
    receipt the JCS canonicaliser (signing_input, chain recompute) would crash on.
    """
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > max_depth:
            return True
        if isinstance(cur, dict):
            stack.extend((v, depth + 1) for v in cur.values())
        elif isinstance(cur, list):
            stack.extend((v, depth + 1) for v in cur)
    return False


_UNSAFE_INTEGER_NOTE = "an integer outside +/-2**53 has no exact double; raw-text ingest refuses it"


    # Same refusal strict ingest gives raw text, so a parsed dict cannot pass it.
def _holds_unsafe_integer(*inputs: Any) -> bool:
    try:
        for value in inputs:
            strict_json.assert_canonical_integers(value)
    except strict_json.UnsafeIntegerError:
        return True
    return False


    # One-axis unverified result for a pre-axis refusal (depth, profile range).
def _early_unverified(
    fmt: str, note: str, context: VerificationContext | None = None,
) -> VerifyResult:
    axes = [_axis("structure", crypto.FAIL, note)]
    verdict, failure_class = fold_independent(axes, keyed=False)
    return VerifyResult(
        fmt=fmt,
        axes=axes,
        verdict=verdict,
        failure_class=failure_class,
        first_failing_edge=first_failing_edge(axes),
        context=context,
        not_checked=not_checked_declaration(),
        coverage=_oracle_coverage([a.axis for a in axes]),
    )


    # Verify one parsed receipt and return a structured ``VerifyResult``.
def verify(
    doc: dict,
    adapters: list[FormatAdapter],
    key_provider: Any = None,
    predecessor: dict | None = None,
    *,
    context: VerificationContext | None = None,
) -> VerifyResult:
    ad = detect(doc, adapters)
    resolved = context or VerificationContext()
    if ad is None:
        axes = [_axis("structure", crypto.FAIL, "no adapter recognises this receipt")]
        verdict, failure_class = fold_independent(axes, keyed=False)
        return VerifyResult(
            fmt="unknown",
            axes=axes,
            verdict=verdict,
            failure_class=failure_class,
            first_failing_edge=first_failing_edge(axes),
            context=resolved,
            not_checked=not_checked_declaration(),
            coverage=_oracle_coverage([a.axis for a in axes]),
        )
    # An over-nested receipt would crash the recursive JCS encoder. Cap it here and
    # report unverified/unverifiable, never verified, matching the standalone gate.
    if _exceeds_depth(doc, MAX_NESTING_DEPTH) or (
        predecessor is not None and _exceeds_depth(predecessor, MAX_NESTING_DEPTH)
    ):
        return _early_unverified(ad.name, _TOO_DEEP_NOTE, resolved)
    if _holds_unsafe_integer(doc, predecessor, key_provider, resolved.originating_envelope):
        return _early_unverified(ad.name, _UNSAFE_INTEGER_NOTE, resolved)
    pred_ad = detect(predecessor, adapters) if predecessor is not None else None
    refusal = ad.profile_precheck(
        doc,
        predecessor=predecessor,
        predecessor_fmt=pred_ad.name if pred_ad is not None else None,
        counterparty=resolved.originating_envelope,
    )
    if refusal is not None:
        return _early_unverified(ad.name, refusal, resolved)

    axes = [
        _structure_axis(ad, doc),
        _signature_axis(ad, doc, key_provider),
        _chain_axis(ad, doc, adapters, predecessor),
        _seq_axis(ad, doc, adapters, predecessor),
    ]
    axes.extend(_axis(name, res, note) for name, res, note in
                ad.extra_axes_with_context(doc, key_provider, resolved))

    # Expiry reports on its own axis and never folds the verdict (criterion 426);
    # a keyed digest reports verified_keyed, never plain verified (criterion 438).
    verdict, failure_class = fold_independent(axes, keyed=ad.keyed_digest(doc))
    signer = ad.attestation(doc).get("signer")
    return VerifyResult(
        fmt=ad.name,
        axes=axes,
        verdict=verdict,
        failure_class=failure_class,
        signer=signer,
        first_failing_edge=first_failing_edge(axes),
        context=resolved,
        not_checked=not_checked_declaration(),
        coverage=_oracle_coverage([a.axis for a in axes]),
    )


    # Lowercase hex SHA-256 - the chain primitive every format shares.
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
