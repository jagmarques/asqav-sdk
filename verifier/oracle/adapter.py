"""The format-adapter seam - six methods that describe one receipt format.

Everything format-specific lives behind this seam; everything shared (the JCS
canonicaliser, the Ed25519 / ML-DSA verify, the SHA-256 chain walk, the vector
runner) is infrastructure the core drives. A new format is one ``FormatAdapter``
subclass, no change to the core.

The single most error-prone divergence between formats is the signing input: some
sign DIRECTLY over the canonical bytes, others hash-then-sign, and each strips a
different field set before canonicalising. ``signing_input`` encodes that per
format so the core never has to know.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SignatureMaterial:
    """The bytes and metadata needed to check one issuer signature.

    Fields:
      sig: raw signature bytes (already decoded from hex / base64url / multibase).
      alg: algorithm token routed through ``crypto.verify_signature``.
      kid: key identifier used by ``resolve_key`` to find the public key.
    """

    sig: bytes
    alg: str
    kid: str


@dataclass(frozen=True)
class ChainStep:
    """The hash-chain linkage for one receipt.

    Fields:
      prev_field: the value of the format's previous-hash field, or None when the
        format omits it on genesis (AERF) - distinct from an explicit-null genesis.
      is_genesis: True when this receipt declares itself first on its chain.
      recompute: callable(predecessor_doc) -> hex digest, encoding which bytes the
        format hashes (e.g. AERF hashes the signable payload, sig excluded).
    """

    prev_field: str | None
    is_genesis: bool
    recompute: Any


class FormatAdapter(ABC):
    """One receipt format's binding to the shared verification core.

    Subclasses implement the six methods below. ``name`` labels the format in
    results and the vector manifest. ``detect`` is a cheap structural fingerprint
    (no crypto) the dispatcher uses to pick the adapter for a parsed receipt.
    """

    #: Stable format label surfaced in results and the vector manifest.
    name: str = "abstract"

    @abstractmethod
    def detect(self, doc: dict) -> bool:
        """Cheap structural test: does this adapter own ``doc``? No crypto."""

    @abstractmethod
    def extract_signature(self, doc: dict) -> SignatureMaterial:
        """Pull the issuer signature bytes plus alg and kid out of ``doc``."""

    @abstractmethod
    def resolve_key(self, doc: dict, key_provider: Any) -> tuple[bytes | None, str]:
        """Return ``(public_key_bytes_or_None, note)`` for the receipt's signer.

        ``key_provider`` is format-shaped: a JWKS dict for Asqav-native, a
        ``{key_id: pem_or_raw}`` map for AERF, or None when the key is embedded.
        """

    @abstractmethod
    def signing_input(self, doc: dict) -> bytes:
        """Return the exact bytes the signature covers.

        Encodes the per-format rule: which fields are stripped, the canonical
        form, and whether the result is pre-hashed or signed directly.
        """

    @abstractmethod
    def chain_step(self, doc: dict) -> ChainStep:
        """Describe how this receipt links to its predecessor."""

    @abstractmethod
    def schema(self, doc: dict) -> tuple[str, str]:
        """Structural check; returns ``(result, note)`` with PASS / FAIL."""
