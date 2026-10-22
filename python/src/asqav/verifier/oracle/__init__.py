"""Universal neutral verifier - verify agent receipts across formats.
The oracle EXTENDS the standalone Asqav verifier (``asqav/verifier/verify_receipt.py``)
to verify rival and sibling receipt formats behind one seam. It proves only what a
verifier can prove from the bytes: a valid signature over the canonical bytes, a
reproducible hash-chain link and structural presence at time T, never behaviour or
correctness of the recorded action. Public surface: ``__all__``."""
# ``FormatAdapter`` is the 6-method per-format seam; ``VerifyResult`` and
# ``AxisResult`` are the structured outcome.
from __future__ import annotations

from .adapter import ChainStep, FormatAdapter, SignatureMaterial, VerificationContext
from .adapters.acta import ActaAdapter
from .adapters.aerf import AerfAdapter
from .adapters.agentreceipts import AgentReceiptsAdapter
from .adapters.asqav_native import AsqavNativeAdapter
from .adapters.authproof import AuthproofAdapter
from .adapters.pipelock import PipelockEvidenceAdapter
from .adapters.w3c_vc import W3cVcAdapter
from .core import AxisResult, VerifyResult, verify

#: Fingerprints are exclusive, so registration order is not load-bearing.
ADAPTERS: list[FormatAdapter] = [
    AsqavNativeAdapter(),
    AerfAdapter(),
    ActaAdapter(),
    AgentReceiptsAdapter(),
    W3cVcAdapter(),
    AuthproofAdapter(),
    PipelockEvidenceAdapter(),
]

__all__ = [
    "ADAPTERS",
    "ActaAdapter",
    "AerfAdapter",
    "AgentReceiptsAdapter",
    "AsqavNativeAdapter",
    "AuthproofAdapter",
    "AxisResult",
    "ChainStep",
    "FormatAdapter",
    "PipelockEvidenceAdapter",
    "SignatureMaterial",
    "VerifyResult",
    "VerificationContext",
    "W3cVcAdapter",
    "verify",
]
