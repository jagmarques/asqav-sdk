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


@dataclass(frozen=True)
class AxisResult:
    """One verification axis outcome.

    Fields:
      axis: which check (structure / signature / chain).
      result: PASS / FAIL / SKIPPED.
      note: human-readable detail for the report.
    """

    axis: str
    result: str
    note: str


@dataclass
class VerifyResult:
    """The aggregate outcome of verifying one receipt.

    ``verdict`` is PASS only when every non-skipped axis passed AND the signature
    was actually checked; a skipped signature downgrades to INCOMPLETE, never a
    PASS, mirroring the standalone verifier.
    """

    fmt: str
    axes: list[AxisResult] = field(default_factory=list)
    verdict: str = "INCOMPLETE"
    #: In-body origin attestation (v:2 ``signer``), surfaced from the signed
    #: payload. None when the receipt carries none (v:1). Never gates the verdict.
    signer: str | None = None

    def axis(self, name: str) -> AxisResult | None:
        """Return the result for one axis, or None if it was not run."""
        return next((a for a in self.axes if a.axis == name), None)


def detect(doc: dict, adapters: list[FormatAdapter]) -> FormatAdapter | None:
    """Return the first adapter whose structural fingerprint matches ``doc``."""
    if not isinstance(doc, dict):
        # A non-object receipt (array, string, number, null) matches no format;
        # the adapters assume a dict, so guard here rather than crash in detect().
        return None
    return next((a for a in adapters if a.detect(doc)), None)


def _signature_axis(ad: FormatAdapter, doc: dict, key_provider: Any) -> AxisResult:
    sm = ad.extract_signature(doc)
    pk, note = ad.resolve_key(doc, key_provider)
    if pk is None:
        return AxisResult("signature", crypto.SKIPPED, f"no key: {note}")
    msg = ad.signing_input(doc)
    res, why = crypto.verify_signature(sm.alg, pk, msg, sm.sig)
    return AxisResult("signature", res, why)


def _chain_axis(
    ad: FormatAdapter, doc: dict, adapters: list[FormatAdapter], predecessor: dict | None
) -> AxisResult:
    step = ad.chain_step(doc)
    if step.is_genesis:
        return AxisResult("chain", crypto.PASS, "genesis receipt (no predecessor link)")
    if predecessor is None:
        return AxisResult("chain", crypto.SKIPPED, "no predecessor supplied")
    # A chain link must stay within one format; a cross-format predecessor is not a valid link.
    pred_ad = detect(predecessor, adapters)
    if pred_ad is None or pred_ad.name != ad.name:
        return AxisResult("chain", crypto.FAIL, "predecessor is a different receipt format")
    actual = step.recompute(predecessor)
    if actual == step.prev_field:
        return AxisResult("chain", crypto.PASS, "chain link rederives from predecessor")
    exp = str(step.prev_field)[:16]
    return AxisResult("chain", crypto.FAIL, f"chain break: expected {exp}.. got {actual[:16]}..")


def verify(
    doc: dict,
    adapters: list[FormatAdapter],
    key_provider: Any = None,
    predecessor: dict | None = None,
) -> VerifyResult:
    """Verify one parsed receipt and return a structured ``VerifyResult``."""
    ad = detect(doc, adapters)
    if ad is None:
        return VerifyResult(
            fmt="unknown",
            axes=[AxisResult("structure", crypto.FAIL, "no adapter recognises this receipt")],
            verdict="FAIL",
        )

    axes = [
        AxisResult("structure", *ad.schema(doc)),
        _signature_axis(ad, doc, key_provider),
        _chain_axis(ad, doc, adapters, predecessor),
    ]
    axes.extend(AxisResult(name, res, note) for name, res, note in ad.extra_axes(doc, key_provider))

    has_fail = any(a.result == crypto.FAIL for a in axes)
    # Any skipped axis except a missing-predecessor chain blocks PASS: an unchecked
    # signature / counter-sign / PDP layer means INCOMPLETE, never a hiding PASS.
    blocking_skip = any(a.result == crypto.SKIPPED and a.axis != "chain" for a in axes)
    if has_fail:
        verdict = "FAIL"
    elif blocking_skip:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"
    signer = ad.attestation(doc).get("signer")
    return VerifyResult(fmt=ad.name, axes=axes, verdict=verdict, signer=signer)


def sha256_hex(data: bytes) -> str:
    """Lowercase hex SHA-256 - the chain primitive every format shares."""
    return hashlib.sha256(data).hexdigest()
