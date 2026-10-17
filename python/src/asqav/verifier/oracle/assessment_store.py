"""Retained assessment log with reassessment labelling (488 C10).
Append-only hash-chained entries; originals never revised, later runs append."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


def _canonical(obj: object) -> bytes:
    """Deterministic bytes for hashing (UTF-8 canonical JSON)."""
    # ensure_ascii=False: UTF-8 bytes match the TS JSON.stringify side,
    # so non-ASCII inputs hash identically in both languages (488 C15).
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class AssessmentEntry:
    """One retained assessment; immutable once appended."""

    doc_sha256: str
    context: dict
    axes: dict
    verdict: str
    failure_class: str | None
    recorded_at: str | None = None
    kind: str = "original"
    supersedes: str | None = None
    # Why a reassessment exists (488 C10, e.g. key-compromised,
    # policy-updated); originals keep None. Defaults keep old logs loading.
    reason: str | None = None
    prev_hash: str = field(default="GENESIS")
    entry_hash: str = field(default="")
    # Bytes-derived envelope ref (488 C39a), outside the hash
    # body so old logs still load and chain-hashes never move.
    agreement: dict | None = None

    def hashed(self) -> "AssessmentEntry":
        """Return this entry with entry_hash computed over its content."""
        # agreement and reason ride outside the chain: old entries hashed
        # body without them, and both re-derive from the chained digest.
        body = {k: v for k, v in asdict(self).items()
                if k not in ("entry_hash", "agreement", "reason")}
        digest = hashlib.sha256(_canonical(body)).hexdigest()
        # The ref and the label ride along unchained; the digest covers the rest.
        return AssessmentEntry(**{**body, "agreement": self.agreement,
                                  "reason": self.reason, "entry_hash": digest})


class AssessmentStore:
    """Append-only assessment log with replay and reassessment labelling."""

    def __init__(self) -> None:
        self._entries: list[AssessmentEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def record(
        self,
        doc_sha256: str,
        context: dict,
        axes: dict,
        verdict: str,
        failure_class: str | None,
        recorded_at: str | None = None,
        agreement: dict | None = None,
    ) -> AssessmentEntry:
        """Append an original assessment; originals are never modified."""
        if self.replay(doc_sha256) is not None:
            raise ValueError(f"duplicate original for {doc_sha256[:16]}; reassess instead")
        prev = self._entries[-1].entry_hash if self._entries else "GENESIS"
        entry = AssessmentEntry(
            doc_sha256=doc_sha256,
            context=dict(context),
            axes=dict(axes),
            verdict=verdict,
            failure_class=failure_class,
            recorded_at=recorded_at,
            agreement=dict(agreement) if agreement is not None else None,
            prev_hash=prev,
        ).hashed()
        self._entries.append(entry)
        return entry

    def replay(self, doc_sha256: str) -> AssessmentEntry | None:
        """Return the original assessment for these input bytes, if retained."""
        for entry in self._entries:
            if entry.doc_sha256 == doc_sha256 and entry.kind == "original":
                return entry
        return None

    def current(self, doc_sha256: str) -> AssessmentEntry | None:
        """Latest assessment for these bytes: reassessment wins over the original.

        Readers take current(); replay() stays the contemporaneous original,
        so a known-wrong frozen verdict is never served as the answer."""
        seen: AssessmentEntry | None = None
        for entry in self._entries:
            if entry.doc_sha256 == doc_sha256 and entry.kind in ("original", "reassessment"):
                seen = entry
        return seen

    def reassess(
        self,
        doc_sha256: str,
        context: dict,
        axes: dict,
        verdict: str,
        failure_class: str | None,
        recorded_at: str | None = None,
        agreement: dict | None = None,
        reason: str | None = None,
    ) -> AssessmentEntry:
        """Append a reassessment naming the original it supersedes."""
        original = self.replay(doc_sha256)
        if original is None:
            raise ValueError(f"no retained original for {doc_sha256[:16]}; record first")
        prev = self._entries[-1].entry_hash if self._entries else "GENESIS"
        entry = AssessmentEntry(
            doc_sha256=doc_sha256,
            context=dict(context),
            axes=dict(axes),
            verdict=verdict,
            failure_class=failure_class,
            recorded_at=recorded_at,
            kind="reassessment",
            supersedes=original.entry_hash,
            agreement=dict(agreement) if agreement is not None else None,
            reason=reason,
            prev_hash=prev,
        ).hashed()
        self._entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        """True when every entry hash recomputes and every link holds."""
        prev = "GENESIS"
        for entry in self._entries:
            if entry.prev_hash != prev:
                return False
            if entry.hashed().entry_hash != entry.entry_hash:
                return False
            prev = entry.entry_hash
        return True

    def to_jsonl(self) -> str:
        """Serialize retained entries, one object per line."""
        return "\n".join(json.dumps(asdict(e), sort_keys=True) for e in self._entries)

    @classmethod
    def from_jsonl(cls, text: str) -> "AssessmentStore":
        """Restore retained entries; raises ValueError on a broken chain."""
        store = cls()
        for line in text.splitlines():
            if line.strip():
                store._entries.append(AssessmentEntry(**json.loads(line)))
        if not store.verify_chain():
            raise ValueError("assessment log chain broken")
        originals: dict[str, str] = {}
        for entry in store._entries:
            if entry.kind == "original":
                if entry.supersedes is not None:
                    raise ValueError("original must not supersede")
                if entry.doc_sha256 in originals:
                    raise ValueError("duplicate original in restored log")
                originals[entry.doc_sha256] = entry.entry_hash
            elif entry.kind != "reassessment":
                raise ValueError(f"unknown entry kind {entry.kind!r}")
        for entry in store._entries:
            if entry.kind == "reassessment":
                if originals.get(entry.doc_sha256) != entry.supersedes:
                    raise ValueError("reassessment links a missing original")
        return store
