"""Client-side digests and validators for the fields a signing call carries."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def _compute_action_ref(action_type: str, context: dict[str, Any] | None) -> str:
    """Client-side ``sha256:<hex>`` of the canonical Action object.

    The Action shape mirrors `core/canonical.py:hash_action` on the cloud
    so the SDK and cloud agree byte-for-byte under JCS encoding.
    """
    from ..canonicalize import canonicalize_action

    return "sha256:" + hashlib.sha256(canonicalize_action(action_type, context)).hexdigest()


#: Wire format for caller-supplied sha256 digest fields (matches helper output).
_SHA256_HEX_RE = re.compile(r"^sha256:[a-f0-9]{64}$")

#: Bare 64-hex, no prefix; the controls_evaluated quorum.attestation_hash form.
_BARE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Server-built control keys an enumerable controls_evaluated block may carry;
#: mirrors the cloud false_control_attestation_guard allowlist (conformance.py).
_ALLOWED_CONTROL_KEYS: frozenset[str] = frozenset(
    {
        "emergency_halt",
        "delegation_scope",
        "quorum",
        "mandate",
        "policy",
        "content_scan",
        "result",
    }
)


#: Wire format for ``tool_fingerprint``: 32 bare lowercase hex (SHA-256[:32]);
#: unlike the ``sha256:<hex>`` form of the other digests.
_TOOL_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{32}$")


def _compute_tool_fingerprint(tool_name: str, tool_schema: dict[str, Any] | None) -> str:
    """32-hex fingerprint over `{tool_name, schema}` (NSA CSI 6030316-26).
    First 32 hex chars of the JCS (RFC 8785) SHA-256, bare."""
    from ..canonicalize import canonicalize

    payload = canonicalize({"tool_name": tool_name, "schema": tool_schema or {}})
    return hashlib.sha256(payload).hexdigest()[:32]


def _compute_config_manifest_digest(manifest: dict[str, Any]) -> str:
    """Compute the canonical ``sha256:<hex>`` of a configuration manifest.
    JCS (RFC 8785) digest; NSA CSI 6030316-26 audits capability creep."""
    from ..canonicalize import canonicalize

    return "sha256:" + hashlib.sha256(canonicalize(manifest)).hexdigest()


def _compute_cve_inventory_digest(cve_list: list[Any]) -> str:
    """Compute the canonical ``sha256:<hex>`` of a CVE inventory snapshot.
    JCS (RFC 8785) digest of the record list (NSA CSI 6030316-26)."""
    from ..canonicalize import canonicalize

    return "sha256:" + hashlib.sha256(canonicalize(cve_list)).hexdigest()
