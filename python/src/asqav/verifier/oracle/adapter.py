"""The format-adapter seam: six methods that describe one receipt format.
Format-specific code lives here; the core drives only shared infrastructure."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class VerificationContext:
    """Caller-supplied comparison input; never key material."""

    originating_envelope: Any = None
    #: Wall-clock override the time axes evaluate against; None = now.
    evaluation_time: Any = None
    #: Trust roots the caller evaluated against (attribution only).
    trust_roots: Any = None
    #: Revocation snapshot the caller evaluated against (attribution only).
    revocation_snapshot: Any = None
    #: Policy revision the caller evaluated against (attribution only).
    policy_revision: Any = None
    #: Trust-policy identity the caller evaluated against (attribution only).
    trust_policy_id: Any = None
    #: Profile revision the caller evaluated against (attribution only).
    profile_revision: Any = None
    #: Evidence cutoff the caller evaluated against (attribution only).
    evidence_knowledge_cutoff: Any = None
    #: Event time the caller evaluated against (attribution only).
    event_time: Any = None
    #: Effective time the caller evaluated against (attribution only).
    effective_time: Any = None
    #: Who ran the evaluation (attribution only).
    evaluator: Any = None


def resolve_evaluation_now(evaluation_time: Any) -> datetime | None:
    """Coerce the context hint to a datetime; unparseable means wall clock."""
    parsed: datetime | None = None
    if evaluation_time is None or isinstance(evaluation_time, datetime):
        parsed = evaluation_time
    elif isinstance(evaluation_time, str):
        try:
            parsed = datetime.fromisoformat(evaluation_time.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed is not None and parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass(frozen=True)
class SignatureMaterial:
    """The bytes and metadata needed to check one issuer signature.
sig is raw bytes, alg routes the verifier, kid finds the key."""

    sig: bytes
    alg: str
    kid: str


@dataclass(frozen=True)
class ChainStep:
    """The hash-chain linkage for one receipt.
prev_field None on AERF genesis; is_genesis and recompute walk the chain."""

    prev_field: str | None
    is_genesis: bool
    recompute: Any


class FormatAdapter(ABC):
    """One receipt format's binding to the shared verification core.
Subclasses implement six methods; ``detect`` is a structural fingerprint."""

    #: Stable format label surfaced in results and the vector manifest.
    name: str = "abstract"

    @abstractmethod
    def detect(self, doc: dict) -> bool:
        # Cheap structural test: does this adapter own ``doc``? No crypto.
        pass

    @abstractmethod
    def extract_signature(self, doc: dict) -> SignatureMaterial:
        # Pull the issuer signature bytes plus alg and kid out of ``doc``.
        pass

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
        # Describe how this receipt links to its predecessor.
        pass

    @abstractmethod
    def schema(self, doc: dict) -> tuple[str, str]:
        # Structural check; returns ``(result, note)`` with PASS / FAIL.
        pass

    def extra_axes(self, doc: dict, key_provider: Any) -> list[tuple[str, str, str]]:
        """Format-specific axes beyond structure, signature and chain.
Return ``(axis_name, result, note)`` per required check; missing layer FAILs."""
        return []

    def seq_of(self, doc: dict) -> Any:
        # Raw so the axis can report a malformed counter; extra_axes never sees the
        # predecessor continuity needs, so the counter is read here instead.
        return None

    def extra_axes_with_context(
        self, doc: dict, key_provider: Any, context: VerificationContext,
    ) -> list[tuple[str, str, str]]:
        """Preserve adapters implementing the two-argument extra-axis hook."""
        return self.extra_axes(doc, key_provider)

    def attestation(self, doc: dict) -> dict[str, Any]:
        """In-body attestation fields surfaced in the verdict (e.g. ``signer``).

        Not a check. A format carrying an in-signed-body origin attestation
        returns it here so the verdict exposes it. Default is none.
        """
        return {}

    def keyed_digest(self, doc: dict) -> bool:
        """True when the receipt's digest is keyed (criterion 438).

        A keyed digest (e.g. HMAC-SHA256) is internally consistent but not
        third-party re-derivable, so a fully-checked receipt reports
        ``verified_keyed``, never plain ``verified``. Default is False.
        """
        return False

    def profile_precheck(
        self, doc: dict, predecessor: Any = None, predecessor_fmt: str | None = None,
        counterparty: Any = None,
    ) -> str | None:
        """Refusal note or None; default no-op, the core notes it early."""
        return None
