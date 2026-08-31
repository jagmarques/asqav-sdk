"""Asqav-native adapter - the existing verify_receipt logic behind the seam.

Two real Asqav wire shapes route here, kept mutually exclusive at detection:

  - COMPLIANCE mode: the 3-key ``{payload, signature, anchors}`` envelope whose
    ``payload`` carries ``previousReceiptHash`` / ``issuer_id`` (the protectmcp:*
    shape). ML-DSA-65 (or Ed25519 for the test vectors) signs the canonical bytes
    of ``payload`` DIRECTLY (no pre-hash, no field strip); the chain hash is the
    SHA-256 of the predecessor payload's canonical bytes.
  - HASH mode: the default ``/sign`` output - a FLAT receipt with ``mode:"hash"``,
    ``payload:null``, a ``signature_b64`` (or ``signature``), and a ``hash`` over
    the action bytes. The signing input is the flat 11-field object the cloud
    rebuilds, not the receipt itself; ``_hash_mode_signing_input`` reconstructs it.

Scope: this adapter verifies the issuer signature, the hash-chain link, and
structural presence of the required fields. It does NOT check anchor liveness or
issued_at clock skew - the standalone ``verify_receipt`` carries those axes.

Key resolution and the compliance-mode structure check delegate to
``verify_receipt`` so the two surfaces stay byte-for-byte identical; the oracle
does not re-implement them. Canonicalisation goes through ``asqav_jcs``, asserted
byte-identical to ``verify_receipt.canonical_json`` by the cloud-parity test.
"""
from __future__ import annotations

from typing import Any

from asqav.verifier import verify_receipt as _vr

from ..adapter import ChainStep, FormatAdapter, SignatureMaterial
from ..canonical import asqav_jcs
from ..core import sha256_hex
from .acta import _is_lower_hex

#: Field set the cloud's hash-mode signer canonicalises, in _build_signing_message order.
_HASH_MODE_FIELDS = (
    "v",
    "mode",
    "hash",
    "hash_algo",
    "metadata",
    "server_timestamp",
    "action_id",
    "agent_id",
    "org_id",
    "policy_digest",
    "policy_decision",
)


#: Claims belonging to the signed payload; hash mode signs the flat fields only,
#: so a thumbprint pasted onto one binds nothing.
_UNSIGNED_CLAIM_FIELDS = ("issuer_id", "previousReceiptHash", "key_thumbprint")


def _is_hash_mode(doc: dict) -> bool:
    """True for a flat hash-mode signature receipt (mode=hash, null payload, a sig).

    Routing reads the shape of the signed unit and nothing else: a dict
    ``payload`` is the compliance signed unit, any other shape routes here. No
    field an attacker can add or omit moves a doc between the paths, so an added
    field cannot select a weaker axis set. A doc displaying claims outside its
    signed unit FAILs the structure axis instead of being rerouted.
    """
    if doc.get("mode") != "hash" or isinstance(doc.get("payload"), dict):
        return False
    return bool(doc.get("signature_b64") or doc.get("signature"))


def _resolved_key_material(entry: dict | None) -> tuple[Any, bytes | None]:
    """alg plus raw public-key bytes of the resolved directory entry.

    The directory publishes the key as standard base64 under ``public_key``; an
    AKP thumbprint is taken over the same bytes in unpadded base64url, so the
    decode has to happen before the digest, never after.
    """
    if not isinstance(entry, dict):
        return None, None
    try:
        return entry.get("alg"), _vr._b64decode(entry.get("public_key", ""))
    except Exception:
        return entry.get("alg"), None


def _payload(doc: dict) -> dict:
    """Normalise to the signed payload regardless of envelope nesting.

    An explicit ``payload: null`` falls back to the doc, matching the TypeScript
    payloadOf, so a caller never has to guard against None.
    """
    env = _vr.normalise_envelope(doc)
    payload = env.get("payload")
    return payload if isinstance(payload, dict) else env


    # Decode signature material; b'' on any malformed input so verify FAILs, never crashes.
def _safe_b64(value: Any) -> bytes:
    if not isinstance(value, str):
        return b""
    try:
        return _vr._b64decode(value)
    except Exception:
        return b""


    # Asqav Compliance Receipt - ML-DSA-65 over canonical bytes (compliance or hash mode).
class AsqavNativeAdapter(FormatAdapter):

    name = "asqav-native"

    def detect(self, doc: dict) -> bool:
        if _is_hash_mode(doc):
            return True
        sig = doc.get("signature")
        # An ACTA receipt carries a lowercase-hex sig; decline it so the formats stay disjoint.
        if isinstance(sig, dict) and _is_lower_hex(sig.get("sig")):
            return False
        payload = doc.get("payload")
        if isinstance(payload, dict) and "previousReceiptHash" in payload:
            return True
        # A bare payload (the conformance-vector shape) is also Asqav-native.
        return "previousReceiptHash" in doc and "issuer_id" in doc

    def extract_signature(self, doc: dict) -> SignatureMaterial:
        if _is_hash_mode(doc):
            return SignatureMaterial(
                sig=_safe_b64(doc.get("signature_b64") or doc.get("signature", "")),
                alg=doc.get("algorithm", "ML-DSA-65"),
                kid=doc.get("key_id", ""),
            )
        env = _vr.normalise_envelope(doc)
        sig_obj = env.get("signature", {})
        if isinstance(sig_obj, str):
            sig_obj = {"alg": "ML-DSA-65", "kid": _payload(doc).get("issuer_id", ""), "sig": sig_obj}
        return SignatureMaterial(
            sig=_safe_b64(sig_obj.get("sig", "")),
            alg=sig_obj.get("alg", "ML-DSA-65"),
            kid=sig_obj.get("kid", ""),
        )

    def _signing_key_entry(self, doc: dict, jwks: dict) -> dict | None:
        """The one jwks entry this receipt's signature is checked against.

        Cloud receipts set kid to the issuer id; when that resolves nothing, the
        agent's own key answers, mirroring run(). agent_id is attacker-controlled,
        so the agent match trusts only a key whose published issuer_id equals the
        one the receipt claims inside its signed bytes.

        Every axis resolves through here. kid lives OUTSIDE the signed bytes, so a
        second independent lookup on it is attacker-steerable: a receipt could
        verify against a key found by the agent route while the revocation and
        issuer axes read a kid that resolves nothing and emit no axis at all. An
        axis that never exists cannot be blocked on, unlike one that reports
        SKIPPED, so the verdict would read PASS for a key the directory revoked.
        """
        payload = _payload(doc)
        return _vr.match_signing_key(
            jwks,
            self.extract_signature(doc).kid,
            payload.get("agent_id") or doc.get("agent_id"),
            payload.get("issuer_id"),
            payload.get("org_id") or doc.get("org_id"),
        )

    def resolve_key(self, doc: dict, key_provider: Any) -> tuple[bytes | None, str]:
        jwks = key_provider or {"keys": []}
        kid = self.extract_signature(doc).kid
        entry = self._signing_key_entry(doc, jwks)
        if entry is None:
            return None, f"kid {kid!r} not in jwks directory"
        pk = _vr._b64decode(entry["public_key"])
        status = entry.get("status")
        if kid and kid in (entry.get("issuer_id"), entry.get("kid")):
            return pk, f"resolved kid {kid} (status={status})"
        return pk, f"resolved agent key {entry.get('kid')} (status={status})"

    def signing_input(self, doc: dict) -> bytes:
        if _is_hash_mode(doc):
            return self._hash_mode_signing_input(doc)
        # Asqav signs the canonical bytes of the payload directly, no pre-hash.
        return asqav_jcs(_payload(doc))

    def _hash_mode_signing_input(self, doc: dict) -> bytes:
        """Rebuild the flat object the cloud's hash-mode path signs, then canonicalise.

        Mirrors ``agents.py::_build_signing_message`` hash-mode branch field-for-field;
        ``asqav_jcs`` sorts the keys, so insertion order is cosmetic but kept aligned.
        """
        flat = {
            "v": 1,
            "mode": "hash",
            "hash": doc.get("hash"),
            "hash_algo": doc.get("hash_algo") or "sha256",
            "metadata": doc.get("metadata") or {},
            "server_timestamp": doc.get("server_timestamp"),
            "action_id": doc.get("action_id"),
            "agent_id": doc.get("agent_id"),
            "org_id": doc.get("org_id"),
            "policy_digest": doc.get("policy_digest"),
            "policy_decision": doc.get("policy_decision"),
        }
        return asqav_jcs(flat)

    def chain_step(self, doc: dict) -> ChainStep:
        if _is_hash_mode(doc):
            # A hash-mode signature receipt carries no in-band chain link of its own.
            return ChainStep(prev_field=None, is_genesis=True, recompute=lambda _pred: "")
        prev = _payload(doc).get("previousReceiptHash")
        is_genesis = prev == _vr.FIRST_RECEIPT_SEED
        return ChainStep(
            prev_field=prev,
            is_genesis=is_genesis,
            recompute=lambda pred: sha256_hex(asqav_jcs(_payload(pred))),
        )

    def schema(self, doc: dict) -> tuple[str, str]:
        if _is_hash_mode(doc):
            missing = [f for f in _HASH_MODE_FIELDS if doc.get(f) is None and f != "policy_digest"]
            if missing:
                return "FAIL", f"hash-mode receipt missing fields: {','.join(missing)}"
            # A claim outside the signed field set is unauthenticated, whatever it
            # says, so refuse it rather than reporting on bytes nobody signed.
            unsigned = [f for f in _UNSIGNED_CLAIM_FIELDS if f in doc]
            if unsigned:
                return "FAIL", (
                    f"hash-mode receipt carries claim fields its signature does not "
                    f"cover: {','.join(unsigned)}"
                )
            return "PASS", "hash-mode signature receipt; required flat fields present"
        return _vr.check_structure(_payload(doc))

    def __init__(self) -> None:
        # Shared per instance, so a duplicate (issuer_id, nonce) pair is flagged (draft 5.7).
        self._seen_nonces: set[str] = set()

    def extra_axes(self, doc: dict, key_provider: Any) -> list[tuple[str, str, str]]:
        """Gate the verdict on expiry, the signing key's revocation status, and its issuer.

        A receipt signed by a revoked key, or by a key the directory publishes
        under a different issuer, must not PASS offline, matching the hosted
        /verify. No key resolved at all means the signature axis already reports no
        key, so those two axes only weigh in once a key is found. Both wire shapes name
        their issuer inside the signed bytes: issuer_id in compliance mode, org_id
        in hash mode.

        Resolution goes through the same entry the signature axis verifies
        against, so these axes weigh the key that actually signed rather than
        whatever the unsigned kid happens to name.
        """
        # Expiry reads only the signed bytes, so no key is needed. Hash mode signs no
        # expires_at, and reading the flat doc would gate on an uncovered field.
        hash_mode = _is_hash_mode(doc)
        signed = {} if hash_mode else _payload(doc)
        axes: list[tuple[str, str, str]] = [("expiry", *_vr.check_expiry(signed))]
        axes.append(("nonce", *_vr.check_nonce(signed, self._seen_nonces)))
        jwks = key_provider or {"keys": []}
        entry = self._signing_key_entry(doc, jwks)
        # Reported before the no-entry return, so a receipt binding no thumbprint
        # still says so rather than dropping the axis when nothing resolved.
        bound_alg, bound_pk = _resolved_key_material(entry)
        axes.append(("key_binding", *_vr.check_key_binding(signed, bound_alg, bound_pk)))
        # No database offline, so a claimed binding reports unresolved rather than
        # riding along as corroboration nobody checked
        axes.append(("counterparty", *_vr.check_counterparty_binding(signed)))
        axes.append(("payload_digest", *_vr.check_payload_digest(signed)))
        # Hash mode signs no issued_at, so skew reads the flat server_timestamp; without
        # this the oracle accepted a 2099 issue time the standalone verifier refuses.
        stamp = doc.get("server_timestamp", "") if hash_mode else signed.get("issued_at", "")
        axes.append(("skew", *_vr.check_skew(stamp)))
        if entry is None:
            return axes
        key_issuer = _vr.key_issuer_of(entry)
        # The resolved key is bound back to the issuer the signed bytes name.
        if _is_hash_mode(doc):
            issued_at = doc.get("server_timestamp", "")
            bind = _vr.check_org_binding(key_issuer, _vr.key_org_of(entry), doc.get("org_id"))
        else:
            payload = _payload(doc)
            issued_at = payload.get("issued_at", "")
            bind = _vr.check_issuer_binding(key_issuer, payload.get("issuer_id"))
        # Only a caller-verified anchor counts as trusted timing. This adapter pins no
        # TSA material, so it passes False: a forged anchor never rides a revoked key.
        res, note = _vr.check_key_status(
            entry.get("status"), issued_at, _vr.revoked_at_of(entry), False
        )
        return axes + [("key_status", res, note), ("issuer_bind", *bind)]

    def attestation(self, doc: dict) -> dict[str, Any]:
        """Surface the v:2 in-body ``signer``. None for v:1 and hash-mode.

        Read only from the signed payload, so a signer appended as loose
        metadata outside the canonical body is never surfaced (and would also
        break the signature).
        """
        if _is_hash_mode(doc):
            return {}
        signer = _payload(doc).get("signer")
        return {"signer": signer} if signer is not None else {}

    def keyed_digest(self, doc: dict) -> bool:
        """A hash-mode digest sealed with the org salt is keyed (criterion 438).

        hash_algo hmac-sha256 is internally consistent but not third-party
        re-derivable, so a fully-checked receipt reports verified_keyed, never
        plain verified. Read from the signed field set only.
        """
        return _is_hash_mode(doc) and doc.get("hash_algo") == "hmac-sha256"
