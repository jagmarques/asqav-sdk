"""The shared verification core - format detection, dispatch, and the verdict.

``verify(doc, ...)`` detects the format, then drives the adapter through the
shared axes: structure, signature, chain. It proves only what the bytes prove -
a valid signature over the canonical bytes, a reproducible chain link, and
structural presence at time T. It never attests the behaviour or correctness of
the recorded action.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from . import crypto
from .adapter import FormatAdapter

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

    Fields:
      axis: which check (structure / signature / chain).
      result: PASS / FAIL / SKIPPED (internal token).
      note: human-readable detail for the report.
      failure_class: invalid / unverifiable for a non-PASS axis, None on PASS.
    """

    axis: str
    result: str
    note: str
    failure_class: str | None = None


@dataclass
class VerifyResult:
    """The aggregate outcome of verifying one receipt.

    ``verdict`` is ``verified`` only when every non-skipped axis passed AND the
    signature was actually checked; ``verified_keyed`` when the digest is keyed
    (internally consistent but not third-party re-derivable). A skipped signature
    downgrades to ``unverified`` with ``failure_class=unverifiable``, never a
    verified, mirroring the standalone verifier. The expiry axis never folds the
    verdict (426). Defaults fail closed: an unfolded result reads unverified.
    """

    fmt: str
    axes: list[AxisResult] = field(default_factory=list)
    verdict: str = VERDICT_UNVERIFIED
    failure_class: str | None = FAILURE_UNVERIFIABLE
    #: In-body origin attestation (v:2 ``signer``), surfaced from the signed
    #: payload. None when the receipt carries none (v:1). Never gates the verdict.
    signer: str | None = None
    #: Earliest axis in report order that did not PASS, else None. Reported so two verifiers
    #: disagreeing about WHICH check failed first is as visible as disagreeing about the verdict.
    first_failing_edge: str | None = None

        # Return the result for one axis, or None if it was not run.
    def axis(self, name: str) -> AxisResult | None:
        return next((a for a in self.axes if a.axis == name), None)


#: Axes whose FAIL proves a cryptographic/policy binding failure (invalid).
_INVALID_FAIL_AXES = frozenset(
    {
        "signature",
        "anchors",
        "issuer_bind",
        "key_status",
        "key_binding",
        "counterparty",
        "payload_digest",
        "nonce",
        "parent_signature",
        "pdp_signature",
        # A gap proves receipts were withheld and a malformed counter was signed as-is;
        # both are proven defects, not a recompute that could not finish.
        "seq",
    }
)


def axis_failure_class(axis: str, result: str, note: str) -> str | None:
    """Map one axis outcome to its failure class (criterion 418).

    PASS carries none; SKIPPED means recomputation could not complete
    (unverifiable); a FAIL is invalid when a binding was proven broken and
    unverifiable when the receipt's own bytes stopped the recompute. A FAIL this
    table does not name reads unverifiable: an unclassified failure is never
    reported as a proven binding failure.
    """
    if result == crypto.PASS:
        return None
    if result == crypto.SKIPPED:
        return FAILURE_UNVERIFIABLE
    if axis in _INVALID_FAIL_AXES:
        return FAILURE_INVALID
    if axis == "chain":
        # A mismatched link or a cross-format predecessor is a proven break; a
        # predecessor the canonicaliser cannot walk stops the recompute instead.
        if note.startswith("chain break:") or note == "predecessor is a different receipt format":
            return FAILURE_INVALID
        return FAILURE_UNVERIFIABLE
    if axis == "skew":
        if note.startswith("unparseable issued_at"):
            return FAILURE_UNVERIFIABLE
        return FAILURE_INVALID
    if axis == "structure":
        if note.startswith("unsupported ACTA alg") or note.startswith(
            "unsupported signature algorithm"
        ):
            return FAILURE_INVALID
        if note.startswith("key_purpose mismatch"):
            return FAILURE_INVALID
        if "signing-key DID != issuer DID" in note:
            return FAILURE_INVALID
        if note.startswith("proof @context is not a prefix"):
            return FAILURE_INVALID
        return FAILURE_UNVERIFIABLE
    if axis == "expiry":
        if note.startswith("unreadable expires_at"):
            return FAILURE_UNVERIFIABLE
        return FAILURE_INVALID
    # issuer_key / ingest and anything unlisted: the recompute could not complete.
    return FAILURE_UNVERIFIABLE


#: Fixed leading axis order every adapter walks, before its format-specific extras.
#: Pinned so a reorder fails a gate instead of quietly renaming which edge is "first".
AXIS_ORDER_PREFIX = ("structure", "signature", "chain", "seq")


def first_failing_edge(axes: list[AxisResult]) -> str | None:
    """Name the earliest axis in report order that drove the verdict away from verified.

    Walks the axes as reported, which is the fixed order `verify` builds, so the
    answer never depends on which check happened to be evaluated first internally.

    The two exclusions mirror `fold_verdict` exactly, and are the whole reason
    this is not simply "the first non-PASS axis": the expiry axis reports on its
    own and never folds the verdict (criterion 426), and a SKIPPED chain is
    tolerated where any other SKIPPED blocks. Without them a verified receipt
    that is merely expired would name an edge, reporting a failure that did not
    happen. Kept in step by a corpus-wide gate asserting this returns None for
    exactly the verified verdicts.

    SKIPPED counts otherwise: an axis that could not be checked is where the
    chain of reasoning stopped being clean, and naming only a FAIL would let an
    unverifiable receipt report no failing edge at all.
    """
    for a in axes:
        if a.result == crypto.FAIL and a.axis != "expiry":
            return a.axis
        if a.result == crypto.SKIPPED and a.axis != "chain":
            return a.axis
    return None


    # Fold per-axis outcomes into the public verdict + failure class (418/438).
def fold_verdict(axes: list[AxisResult], keyed: bool) -> tuple[str, str | None]:
    failed = [a for a in axes if a.result == crypto.FAIL and a.axis != "expiry"]
    blocking_skip = any(a.result == crypto.SKIPPED and a.axis != "chain" for a in axes)
    if failed:
        # A proven binding failure dominates a malformed-member failure: the
        # receipt is invalid on the strongest ground the axes established.
        classes = {a.failure_class for a in failed}
        failure_class = FAILURE_INVALID if FAILURE_INVALID in classes else FAILURE_UNVERIFIABLE
        return VERDICT_UNVERIFIED, failure_class
    if blocking_skip:
        return VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE
    if keyed:
        return VERDICT_VERIFIED_KEYED, None
    return VERDICT_VERIFIED, None


    # Return the first adapter whose structural fingerprint matches ``doc``.
def detect(doc: dict, adapters: list[FormatAdapter]) -> FormatAdapter | None:
    if not isinstance(doc, dict):
        # A non-object receipt (array, string, number, null) matches no format;
        # the adapters assume a dict, so guard here rather than crash in detect().
        return None
    return next((a for a in adapters if a.detect(doc)), None)


def _axis(axis: str, result: str, note: str) -> AxisResult:
    return AxisResult(axis, result, note, axis_failure_class(axis, result, note))


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
    # A chain link must stay within one format; a cross-format predecessor is not a valid link.
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
    # A gap is omission evidence. Never SKIPPED: fold_verdict blocks on a non-chain
    # SKIPPED, so a receipt with no counter would regress to unverified.
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


#: Max nesting the recursive JCS encoder tolerates, mirrors the standalone verifier cap.
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


    # Verify one parsed receipt and return a structured ``VerifyResult``.
def verify(
    doc: dict,
    adapters: list[FormatAdapter],
    key_provider: Any = None,
    predecessor: dict | None = None,
) -> VerifyResult:
    ad = detect(doc, adapters)
    if ad is None:
        axes = [_axis("structure", crypto.FAIL, "no adapter recognises this receipt")]
        verdict, failure_class = fold_verdict(axes, keyed=False)
        return VerifyResult(
            fmt="unknown",
            axes=axes,
            verdict=verdict,
            failure_class=failure_class,
            first_failing_edge=first_failing_edge(axes),
        )
    # An over-nested receipt would crash the recursive JCS encoder. Cap it here and
    # report unverified/unverifiable, never a verified, matching the standalone gate.
    if _exceeds_depth(doc, MAX_NESTING_DEPTH) or (
        predecessor is not None and _exceeds_depth(predecessor, MAX_NESTING_DEPTH)
    ):
        axes = [_axis("structure", crypto.FAIL, _TOO_DEEP_NOTE)]
        verdict, failure_class = fold_verdict(axes, keyed=False)
        return VerifyResult(
            fmt=ad.name,
            axes=axes,
            verdict=verdict,
            failure_class=failure_class,
            first_failing_edge=first_failing_edge(axes),
        )

    axes = [
        _axis("structure", *ad.schema(doc)),
        _signature_axis(ad, doc, key_provider),
        _chain_axis(ad, doc, adapters, predecessor),
        _seq_axis(ad, doc, adapters, predecessor),
    ]
    axes.extend(_axis(name, res, note) for name, res, note in ad.extra_axes(doc, key_provider))

    # Expiry reports on its own axis and never folds the verdict (criterion 426);
    # a keyed digest reports verified_keyed, never plain verified (criterion 438).
    verdict, failure_class = fold_verdict(axes, keyed=ad.keyed_digest(doc))
    signer = ad.attestation(doc).get("signer")
    return VerifyResult(
        fmt=ad.name,
        axes=axes,
        verdict=verdict,
        failure_class=failure_class,
        signer=signer,
        first_failing_edge=first_failing_edge(axes),
    )


    # Lowercase hex SHA-256 - the chain primitive every format shares.
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
