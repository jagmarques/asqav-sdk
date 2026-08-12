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
from .taxonomy import (
    CLASSIFICATION,
    INVALID,
    NO_REASON,
    PASS,
    UNVERIFIABLE,
    derive_verdict,
)


@dataclass(frozen=True)
class AxisResult:
    """One verification axis outcome.

    Fields:
      axis: which check (structure / signature / chain).
      result: PASS / INVALID / UNVERIFIABLE / SKIPPED (criterion 418).
      note: human-readable detail for the report.
      reason_code: the closed failure-class token, "none" when nothing failed.
    """

    axis: str
    result: str
    note: str
    reason_code: str = NO_REASON

        # The criterion 418 class of this axis: "invalid" / "unverifiable" / None.
    @property
    def classification(self) -> str | None:
        return CLASSIFICATION.get(self.result)


@dataclass
class VerifyResult:
    """The aggregate outcome of verifying one receipt.

    ``verdict`` is PASS only when every applicable axis passed and no
    recomputation was left incomplete (criterion 418): one INVALID axis
    dominates, otherwise any UNVERIFIABLE axis downgrades, and a receipt is
    never PASS while a recomputation failed. The expiry axis never folds it
    (426). ``classification`` mirrors the verdict on the wire.
    """

    fmt: str
    axes: list[AxisResult] = field(default_factory=list)
    verdict: str = UNVERIFIABLE
    #: In-body origin attestation (v:2 ``signer``), surfaced from the signed
    #: payload. None when the receipt carries none (v:1). Never gates the verdict.
    signer: str | None = None

        # Return the result for one axis, or None if it was not run.
    def axis(self, name: str) -> AxisResult | None:
        return next((a for a in self.axes if a.axis == name), None)

        # The criterion 418 wire token for the verdict: valid/invalid/unverifiable.
    @property
    def classification(self) -> str:
        return CLASSIFICATION[self.verdict]


    # Return the first adapter whose structural fingerprint matches ``doc``.
def detect(doc: dict, adapters: list[FormatAdapter]) -> FormatAdapter | None:
    if not isinstance(doc, dict):
        # A non-object receipt (array, string, number, null) matches no format;
        # the adapters assume a dict, so guard here rather than crash in detect().
        return None
    return next((a for a in adapters if a.detect(doc)), None)


def _signature_axis(ad: FormatAdapter, doc: dict, key_provider: Any) -> AxisResult:
    sm = ad.extract_signature(doc)
    pk, note, reason = ad.resolve_key(doc, key_provider)
    if pk is None:
        # The signature cannot be recomputed without its key; never a PASS
        return AxisResult("signature", UNVERIFIABLE, f"no key: {note}", reason)
    msg = ad.signing_input(doc)
    res, why, reason = crypto.verify_signature(sm.alg, pk, msg, sm.sig)
    return AxisResult("signature", res, why, reason)


def _chain_axis(
    ad: FormatAdapter, doc: dict, adapters: list[FormatAdapter], predecessor: dict | None
) -> AxisResult:
    step = ad.chain_step(doc)
    if step.is_genesis:
        return AxisResult("chain", PASS, "genesis receipt (no predecessor link)")
    if predecessor is None:
        # Recomputation cannot complete without the predecessor; blocks PASS
        return AxisResult(
            "chain", UNVERIFIABLE, "no predecessor supplied", "chain_predecessor_missing"
        )
    # A chain link must stay within one format; a cross-format predecessor is not a valid link.
    pred_ad = detect(predecessor, adapters)
    if pred_ad is None or pred_ad.name != ad.name:
        return AxisResult(
            "chain", INVALID, "predecessor is a different receipt format", "chain_mismatch"
        )
    try:
        actual = step.recompute(predecessor)
    except (ValueError, TypeError, RecursionError) as exc:
        return AxisResult(
            "chain", UNVERIFIABLE, f"predecessor not canonicalisable: {exc}",
            "canonicalization_failed",
        )
    if actual == step.prev_field:
        return AxisResult("chain", PASS, "chain link rederives from predecessor")
    exp = str(step.prev_field)[:16]
    return AxisResult(
        "chain", INVALID, f"chain break: expected {exp}.. got {actual[:16]}..", "chain_mismatch"
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
        # No format claims this receipt, so no check can even start: the two
        # failure classes stay distinct - this is the unverifiable one
        return VerifyResult(
            fmt="unknown",
            axes=[AxisResult(
                "structure", UNVERIFIABLE, "no adapter recognises this receipt",
                "format_unrecognized",
            )],
            verdict=UNVERIFIABLE,
        )
    # An over-nested receipt would crash the recursive JCS encoder. Cap it here and
    # report UNVERIFIABLE, never a PASS, matching the standalone verifier shape gate.
    if _exceeds_depth(doc, MAX_NESTING_DEPTH) or (
        predecessor is not None and _exceeds_depth(predecessor, MAX_NESTING_DEPTH)
    ):
        return VerifyResult(
            fmt=ad.name,
            axes=[AxisResult("structure", UNVERIFIABLE, _TOO_DEEP_NOTE, "canonicalization_failed")],
            verdict=UNVERIFIABLE,
        )

    axes = [
        AxisResult("structure", *ad.schema(doc)),
        _signature_axis(ad, doc, key_provider),
        _chain_axis(ad, doc, adapters, predecessor),
    ]
    axes.extend(
        AxisResult(name, res, note, reason)
        for name, res, note, reason in ad.extra_axes(doc, key_provider)
    )

    # Expiry reports on its own axis and never folds the verdict (criterion 426);
    # INVALID dominates UNVERIFIABLE and either one blocks PASS (criterion 418).
    # SKIPPED axes do not apply and never block; UNVERIFIABLE ones always do.
    verdict = derive_verdict((a.axis, a.result) for a in axes)
    signer = ad.attestation(doc).get("signer")
    return VerifyResult(fmt=ad.name, axes=axes, verdict=verdict, signer=signer)


    # Lowercase hex SHA-256 - the chain primitive every format shares.
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
