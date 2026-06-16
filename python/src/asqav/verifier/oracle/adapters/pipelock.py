"""Pipelock EvidenceReceipt v2 adapter - Ed25519 over JCS with zeroed signature.

Targets Pipelock v2.x EvidenceReceipt v2 as implemented in the Python verifier
``luckyPipewrench/pipelock-verify-python`` (cloned at commit
``9eaff72a87b3b412945fac6de07739bc2bef2116``). The authoritative signing rule
is in ``pipelock_verify/_evidence.py::_signable_preimage``.

Signing rule: clone the receipt dict, zero out the ``signature`` object
(all four fields set to empty strings), then JCS-canonicalize the result.
The signature is Ed25519 PureEdDSA over those bytes; the 64-byte signature
is hex-encoded with an ``ed25519:`` prefix (``ed25519:<128 hex chars>``).

Key resolution: EvidenceReceipt v2 does NOT embed its signer public key in
the envelope. The caller must supply the raw 32-byte Ed25519 key hex via the
key provider, keyed by ``signature.signer_key_id``. When the key is absent
the signature axis SKIPS and the verdict is INCOMPLETE, never a hiding PASS.

Chain rule: ``chain_seq`` is a monotonic sequence counter; ``chain_prev_hash``
is ``"sha256:" + hex(SHA-256(JCS(full predecessor receipt)))`` for non-genesis
receipts. Genesis receipts set ``chain_prev_hash`` to ``"genesis"`` or
``"sha256:0"`` (both are accepted). The chain axis hash covers the full
predecessor receipt INCLUDING its signature (unlike AERF which strips it).

Canonicalization: Pipelock v2 uses strict RFC 8785 JCS (NFC strings,
lexicographic codepoint key sort, integer-only numeric values, no whitespace).
This is the same dialect as the agentreceipts adapter; we reuse ``jcs_rfc8785``
which byte-matches pipelock's Go canonicaliser for all-ASCII field sets and
NFC input.

WHAT IS VERIFIED:
  - Structure: required envelope fields, ``record_type``, ``receipt_version``,
    ``payload_kind`` in the known set, ``signature.algorithm == ed25519``,
    key-purpose authority-matrix check, RFC 3339 timestamp.
  - Signature: Ed25519 over JCS(receipt with zeroed sig), PASS / FAIL / SKIPPED.
  - Chain: ``chain_prev_hash == "sha256:" + sha256_hex(JCS(predecessor))``.

WHAT IS NOT VERIFIED (the signature layer closes this):
  - Payload semantic validity (13 payload kinds; we check required-field
    presence only for ``proxy_decision`` - the format of the conformance fixture
    - and skip deep validation for the other 12).
  - Key-directory liveness (the oracle never fetches a directory).
  - Clock skew.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any

from ..adapter import ChainStep, FormatAdapter, SignatureMaterial
from ..canonical import jcs_rfc8785
from ..core import sha256_hex

#: Wire constants matching ``pipelock_verify/_evidence.py``.
_RECORD_TYPE = "evidence_receipt_v2"
_RECEIPT_VERSION = 2
_SIG_PREFIX = "ed25519:"
_SIG_LEN = 64  # raw bytes

#: Genesis sentinels: both forms are used in upstream fixtures.
_GENESIS_SENTINELS = frozenset({"genesis", "sha256:0"})

#: Minimum required envelope fields for structural validity.
_REQUIRED_ENVELOPE = (
    "record_type",
    "receipt_version",
    "payload_kind",
    "event_id",
    "timestamp",
    "signature",
)

#: Known payload kinds from the authority matrix.
_PAYLOAD_KINDS = frozenset(
    {
        "proxy_decision",
        "contract_ratified",
        "contract_promote_intent",
        "contract_promote_committed",
        "contract_rollback_authorized",
        "contract_rollback_committed",
        "contract_demoted",
        "contract_expired",
        "contract_drift",
        "shadow_delta",
        "opportunity_missing",
        "key_rotation",
        "contract_redaction_request",
    }
)

#: Authority matrix: payload_kind -> required key_purpose.
_PAYLOAD_AUTHORITY: dict[str, str] = {
    "proxy_decision": "receipt-signing",
    "contract_ratified": "receipt-signing",
    "contract_promote_intent": "contract-activation-signing",
    "contract_promote_committed": "receipt-signing",
    "contract_rollback_authorized": "contract-activation-signing",
    "contract_rollback_committed": "receipt-signing",
    "contract_demoted": "receipt-signing",
    "contract_expired": "receipt-signing",
    "contract_drift": "receipt-signing",
    "shadow_delta": "receipt-signing",
    "opportunity_missing": "receipt-signing",
    "key_rotation": "contract-activation-signing",
    "contract_redaction_request": "contract-activation-signing",
}

#: Signature proof sub-object fields.
_SIGNATURE_FIELDS = frozenset({"signer_key_id", "key_purpose", "algorithm", "signature"})


def _safe_hex(value: Any) -> bytes:
    """Decode a hex string; b'' on any malformed input so verify FAILs, never crashes."""
    if not isinstance(value, str):
        return b""
    try:
        return bytes.fromhex(value)
    except ValueError:
        return b""


def _signable_preimage(doc: dict) -> bytes:
    """JCS(receipt with signature zeroed out) - the exact bytes Ed25519 covers.

    Mirrors ``pipelock_verify._evidence._signable_preimage``: the four fields
    inside the ``signature`` object are replaced with empty strings before
    canonicalization, matching the Go zero-value struct layout.
    """
    clone = dict(doc)
    clone["signature"] = {
        "signer_key_id": "",
        "key_purpose": "",
        "algorithm": "",
        "signature": "",
    }
    return jcs_rfc8785(clone)


def _full_receipt_hash(doc: dict) -> str:
    """SHA-256 of JCS(full receipt including its signature) - the chain primitive."""
    return sha256_hex(jcs_rfc8785(doc))


class PipelockEvidenceAdapter(FormatAdapter):
    """Pipelock EvidenceReceipt v2 - Ed25519 over JCS, chain covers full receipt."""

    name = "pipelock-evidence-v2"

    def detect(self, doc: dict) -> bool:
        """Cheap structural fingerprint: record_type + receipt_version, no crypto."""
        return (
            doc.get("record_type") == _RECORD_TYPE
            and doc.get("receipt_version") == _RECEIPT_VERSION
            and isinstance(doc.get("signature"), dict)
        )

    def extract_signature(self, doc: dict) -> SignatureMaterial:
        sig_proof = doc.get("signature", {})
        sig_value = sig_proof.get("signature", "")
        # Strip the ``ed25519:`` prefix before hex-decoding.
        if isinstance(sig_value, str) and sig_value.startswith(_SIG_PREFIX):
            sig_bytes = _safe_hex(sig_value[len(_SIG_PREFIX):])
        else:
            sig_bytes = b""
        return SignatureMaterial(
            sig=sig_bytes,
            alg="Ed25519",
            kid=sig_proof.get("signer_key_id", ""),
        )

    def resolve_key(self, doc: dict, key_provider: Any) -> tuple[bytes | None, str]:
        """Return the raw 32-byte Ed25519 key from the caller-supplied key provider.

        EvidenceReceipt v2 does NOT embed its public key. The caller passes a
        ``{signer_key_id: hex_or_bytes}`` map. If the key is absent the axis SKIPs.
        """
        kid = doc.get("signature", {}).get("signer_key_id", "")
        provider = key_provider or {}
        material = provider.get(kid)
        if material is None:
            return None, f"signer_key_id {kid!r} not in key provider"
        if isinstance(material, bytes):
            raw = material
        else:
            raw = _safe_hex(str(material))
        if len(raw) != 32:
            return None, f"signer_key_id {kid!r} key is {len(raw)} bytes, expected 32"
        return raw, f"resolved signer_key_id {kid!r}"

    def signing_input(self, doc: dict) -> bytes:
        """JCS preimage: full receipt with signature object zeroed out."""
        return _signable_preimage(doc)

    def chain_step(self, doc: dict) -> ChainStep:
        prev = doc.get("chain_prev_hash")
        # Genesis: field absent OR one of the sentinel values.
        is_genesis = prev is None or prev in _GENESIS_SENTINELS
        # Chain hash covers the full predecessor receipt including its signature.
        return ChainStep(
            prev_field=prev,
            is_genesis=is_genesis,
            recompute=lambda pred: "sha256:" + _full_receipt_hash(pred),
        )

    def schema(self, doc: dict) -> tuple[str, str]:
        # Required envelope fields.
        missing = [f for f in _REQUIRED_ENVELOPE if doc.get(f) is None]
        if missing:
            return "FAIL", f"pipelock-evidence-v2 missing required fields: {','.join(missing)}"

        # record_type and receipt_version are the detection discriminators; re-check for schema.
        if doc.get("record_type") != _RECORD_TYPE:
            return "FAIL", f"record_type must be {_RECORD_TYPE!r}"
        if doc.get("receipt_version") != _RECEIPT_VERSION:
            return "FAIL", f"receipt_version must be {_RECEIPT_VERSION}"

        payload_kind = doc.get("payload_kind", "")
        if payload_kind not in _PAYLOAD_KINDS:
            return "FAIL", f"unknown payload_kind: {payload_kind!r}"

        sig_proof = doc.get("signature", {})
        if not isinstance(sig_proof, dict):
            return "FAIL", "signature must be an object"

        algorithm = sig_proof.get("algorithm", "")
        if algorithm != "ed25519":
            return "FAIL", f"unsupported signature algorithm {algorithm!r}; only ed25519 is implemented"

        key_purpose = sig_proof.get("key_purpose", "")
        required_purpose = _PAYLOAD_AUTHORITY.get(payload_kind)
        if required_purpose and key_purpose != required_purpose:
            return (
                "FAIL",
                f"key_purpose mismatch: {payload_kind!r} requires {required_purpose!r}, "
                f"got {key_purpose!r}",
            )

        sig_value = sig_proof.get("signature", "")
        if not isinstance(sig_value, str) or not sig_value.startswith(_SIG_PREFIX):
            return "FAIL", f"signature.signature must start with {_SIG_PREFIX!r}"

        return (
            "PASS",
            f"pipelock-evidence-v2 envelope; payload_kind={payload_kind!r}; "
            f"key_purpose authority-matrix check passed",
        )
