"""Universal neutral verifier for agent receipts across formats.
It extends the standalone Asqav verifier to rival formats behind one seam."""
# Formats plug in through ``FormatAdapter``; outcomes come back as
# ``VerifyResult`` with per-axis ``AxisResult`` entries.
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
