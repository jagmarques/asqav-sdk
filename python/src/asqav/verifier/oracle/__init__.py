"""Universal neutral verifier - verify agent receipts across formats.

The oracle EXTENDS the standalone Asqav verifier (``asqav/verifier/verify_receipt.py``)
to verify rival and sibling receipt formats behind one seam. It proves only what
a verifier can prove from the bytes: a valid signature over the canonical bytes,
a reproducible hash-chain link, and structural presence at time T. It never
attests behaviour or correctness of the recorded action.

Public surface:
  - ``FormatAdapter``  : the 6-method per-format seam.
  - ``ADAPTERS``       : ordered registry the dispatcher walks for detection.
  - ``verify``         : verify one parsed receipt; returns a ``VerifyResult``.
  - ``VerifyResult`` / ``AxisResult`` : structured outcome.
"""
from __future__ import annotations

from .adapter import ChainStep, FormatAdapter, SignatureMaterial
from .adapters.acta import ActaAdapter
from .adapters.aerf import AerfAdapter
from .adapters.agentreceipts import AgentReceiptsAdapter
from .adapters.asqav_native import AsqavNativeAdapter
from .adapters.authproof import AuthproofAdapter
from .adapters.pipelock import PipelockEvidenceAdapter
from .core import AxisResult, VerifyResult, verify

#: Detection fingerprints are mutually exclusive, so registration order is not load-bearing.
ADAPTERS: list[FormatAdapter] = [
    AsqavNativeAdapter(),
    AerfAdapter(),
    ActaAdapter(),
    AgentReceiptsAdapter(),
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
    "verify",
]
