"""Universal neutral verifier - verify agent receipts across formats.

The oracle EXTENDS the standalone Asqav verifier (``verifier/verify_receipt.py``)
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

import os
import sys

# The adapters import the standalone ``verify_receipt`` as a top-level module. Put the
# verifier directory on the path before importing them so the oracle works both from the
# source tree and from the installed ``asqav.verifier`` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .adapter import ChainStep, FormatAdapter, SignatureMaterial  # noqa: E402
from .adapters.acta import ActaAdapter  # noqa: E402
from .adapters.aerf import AerfAdapter  # noqa: E402
from .adapters.agentreceipts import AgentReceiptsAdapter  # noqa: E402
from .adapters.asqav_native import AsqavNativeAdapter  # noqa: E402
from .adapters.authproof import AuthproofAdapter  # noqa: E402
from .core import AxisResult, VerifyResult, verify  # noqa: E402

#: Detection fingerprints are mutually exclusive, so registration order is not load-bearing.
ADAPTERS: list[FormatAdapter] = [
    AsqavNativeAdapter(),
    AerfAdapter(),
    ActaAdapter(),
    AgentReceiptsAdapter(),
    AuthproofAdapter(),
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
    "SignatureMaterial",
    "VerifyResult",
    "verify",
]
