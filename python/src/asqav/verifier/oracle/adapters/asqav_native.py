"""Asqav-native adapter - the existing verify_receipt logic. Two wire shapes route here, mutually exclusive:
COMPLIANCE mode, the 3-key ``{payload, signature, anchors}`` envelope whose ML-DSA-65 signature covers
the canonical bytes of ``payload``, and HASH mode, a FLAT receipt (``mode:"hash"``, ``payload:null``,
``signature_b64``) over the flat 11-field object the cloud rebuilds (no pre-hash, no strip). Scope: signature, chain, structure
only - NOT anchor liveness or skew. Key resolution and the compliance-mode structure check delegate to
``verify_receipt``: both surfaces stay byte-for-byte identical; the oracle does not re-implement them."""
# Canonicalisation goes through ``asqav_jcs``, asserted byte-identical to
# ``verify_receipt.canonical_json`` by the cloud-parity test.
from __future__ import annotations

import inspect
from typing import Any

from asqav.verifier import verify_receipt as _vr

from .. import crypto
from ..adapter import (
    ChainStep,
    FormatAdapter,
    SignatureMaterial,
    VerificationContext,
    resolve_evaluation_now,
)
from ..canonical import asqav_jcs
from ..core import sha256_hex
from .acta import _is_lower_hex

#: Fields the cloud's hash-mode signer canonicalises, in signer order.
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
#: so a thumbprint or seq pasted onto one binds nothing.
_UNSIGNED_CLAIM_FIELDS = ("issuer_id", "previousReceiptHash", "key_thumbprint", "seq")


def _is_hash_mode(doc: dict) -> bool:
    """True for a flat hash-mode signature receipt (mode=hash, null payload, a sig).

    Routing reads the shape of the signed unit and nothing else: a dict ``payload`` is
    the compliance signed unit, any other shape routes here. No field an attacker can
    add or omit moves a doc between the paths, so an added field cannot select a
    weaker axis set. A doc displaying claims outside its signed unit FAILs, not rerouted."""
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


    # The flat object the cloud's hash-mode path signs, before canonicalisation.
def _flat_signed_fields(doc: dict) -> dict:
    return {
        # The displayed v binds the signature: a relabelled copy rebuilds
        # different bytes and fails crypto, it never inherits the old PASS.
        "v": doc.get("v"),
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


    # Decode signature material; b'' on bad input so verify FAILs, never crashes.
def _safe_b64(value: Any) -> bytes:
    if not isinstance(value, str):
        return b""
    try:
        return _vr._b64decode(value)
    except Exception:
        return b""


def _context_call_style(method: Any) -> str:
    """How an extra_axes override takes context: positional, keyword, or not at all."""
    try:
        params = list(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return "positional"
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(positional) >= 3 or any(p.kind is p.VAR_POSITIONAL for p in params):
        return "positional"
    if any(p.kind is p.VAR_KEYWORD or (p.kind is p.KEYWORD_ONLY and p.name == "context") for p in params):
        return "keyword"
    return "none"


# Asqav Compliance Receipt - ML-DSA-65 over the canonical bytes (both modes).
class AsqavNativeAdapter(FormatAdapter):

    name = "asqav-native"

    def detect(self, doc: dict) -> bool:
        if _is_hash_mode(doc):
            return True
        sig = doc.get("signature")
        # An ACTA receipt carries a lowercase-hex sig; decline it to stay disjoint.
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
        Cloud receipts set kid to the issuer id; when that resolves nothing the agent's own key
        answers, mirroring run(). agent_id is attacker-controlled, so the agent match trusts only a
        key whose published issuer_id equals the one the receipt claims in its signed bytes. Every
        axis resolves through here: kid lives OUTSIDE the signed bytes, so a second lookup on it is
        attacker-steerable: an axis that never exists cannot be blocked on, unlike SKIPPED."""
        payload = _payload(doc)
        return _vr.match_signing_key(
            jwks,
            self.extract_signature(doc).kid,
            payload.get("agent_id") or doc.get("agent_id"),
            payload.get("issuer_id"),
            payload.get("org_id") or doc.get("org_id"),
            payload.get("key_thumbprint"),
        )

    def _verifying_key(self, doc: dict, pk: bytes | None, msg: bytes) -> bytes | None:
        """The resolved key when the signature verifies under it, else None."""
        if pk is None:
            return None
        sm = self.extract_signature(doc)
        res, _ = crypto.verify_signature(sm.alg, pk, msg, sm.sig)
        return pk if res == crypto.PASS else None

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
        return asqav_jcs(_flat_signed_fields(doc))

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
            # Absence FAILs above (v is required); here its value must be a
            # recognised integer, exactly as the payload-mode structure gate.
            v = doc.get("v")
            if not _vr.is_recognised_wire_version(v):
                return "SKIPPED", (
                    f"unsupported wire version {v!r}: "
                    "unverifiable under this document, reported not failed"
                )
            return "PASS", "hash-mode signature receipt; required flat fields present"
        return _vr.check_structure(_payload(doc))

    def __init__(self) -> None:
        # Per instance, so a duplicate (issuer_id, nonce) pair is flagged (draft 5.7).
        self._seen_nonces: set[str] = set()

    def extra_axes_with_context(
        self, doc: dict, key_provider: Any, context: VerificationContext,
    ) -> list[tuple[str, str, str]]:
        # A subclass may still override extra_axes with the older two-argument form.
        style = _context_call_style(self.extra_axes)
        if style == "positional":
            axes = self.extra_axes(doc, key_provider, context)
        elif style == "keyword":
            axes = self.extra_axes(doc, key_provider, context=context)
        else:
            axes = self.extra_axes(doc, key_provider)
        if context.originating_envelope is None:
            return axes
        signed = {} if _is_hash_mode(doc) else _payload(doc)
        kid = _vr.counterparty_acknowledging_kid(doc)
        baseline = ("counterparty", *_vr.check_counterparty_binding(signed, acknowledging_kid=kid))
        entry = self._signing_key_entry(doc, key_provider or {"keys": []})
        if entry is not None:
            kid = _vr.bound_counterparty_kid(kid, entry.get("kid"), _vr.key_issuer_of(entry))
        outcome = _vr.check_counterparty_binding(signed, context.originating_envelope, acknowledging_kid=kid)
        return [("counterparty", *outcome) if axis == baseline else axis for axis in axes]

    def extra_axes(
        self, doc: dict, key_provider: Any, context: VerificationContext | None = None,
    ) -> list[tuple[str, str, str]]:
        """Gate the verdict on expiry, the signing key's revocation status, and its issuer.

        A receipt signed by a revoked key, or by a key the directory publishes under a different
        issuer, must not PASS offline, matching the hosted /verify. No key resolved at all means
        the signature axis already reports no key, so those axes weigh in only once a key is
        found. Both wire shapes name their issuer in the signed bytes (issuer_id / org_id)."""
        # Expiry reads only the signed bytes, so no key is needed. Hash mode signs no
        # expires_at, and reading the flat doc would gate on an uncovered field.
        hash_mode = _is_hash_mode(doc)
        signed = {} if hash_mode else _payload(doc)
        now = resolve_evaluation_now(getattr(context, "evaluation_time", None))
        axes: list[tuple[str, str, str]] = [("expiry", *_vr.check_expiry(signed, now=now))]
        axes.append(("nonce", *_vr.check_nonce(signed, self._seen_nonces)))
        jwks = key_provider or {"keys": []}
        entry = self._signing_key_entry(doc, jwks)
        # Reported before the no-entry return, so a receipt binding no thumbprint
        # still says so rather than dropping the axis when nothing resolved.
        bound_alg, bound_pk = _resolved_key_material(entry)
        axes.append(("key_binding", *_vr.check_key_binding(signed, bound_alg, bound_pk)))
        # No database offline, so a claimed binding reports unresolved rather than
        # riding along as corroboration nobody checked
        kid = _vr.counterparty_acknowledging_kid(doc)
        axes.append(("counterparty", *_vr.check_counterparty_binding(signed, acknowledging_kid=kid)))
        pd_res, pd_note, _pd_app = _vr.check_payload_digest(signed)
        # Oracle axes stay 3-tuples; applicability surfaces in run_structured
        axes.append(("payload_digest", pd_res, pd_note))
        # Hash mode signs no issued_at, so skew reads the flat server_timestamp; without
        # this the oracle accepted a 2099 issue time the standalone verifier refuses.
        stamp = doc.get("server_timestamp", "") if hash_mode else signed.get("issued_at", "")
        axes.append(("skew", *_vr.check_skew(stamp, now=now)))
        # Weigh the plain copies against the signed payload and the key that verified
        msg = self.signing_input(doc)
        verifying = self._verifying_key(doc, bound_pk, msg)
        kid = entry.get("kid") if verifying is not None and entry is not None else None
        axes.append(("outer_copies", *_vr.check_outer_copies(doc, msg, verifying, kid)))
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
            entry.get("status"), issued_at, _vr.revoked_at_of(entry), False,
            entry.get("revocation_reason"),
        )
        return axes + [("key_status", res, note), ("issuer_bind", *bind)]

    def seq_of(self, doc: dict) -> Any:
        # Hash mode signs the flat field set only, so a seq sitting there binds nothing.
        if _is_hash_mode(doc):
            return None
        return _payload(doc).get("seq")

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

    def profile_precheck(
        self, doc: dict, predecessor: Any = None, predecessor_fmt: str | None = None,
        counterparty: Any = None,
    ) -> str | None:
        """Refuse current-profile digest inputs outside the range, else None."""
        if _is_hash_mode(doc):
            if not _vr.is_current_profile_version(doc.get("v")):
                return None
            return _vr.profile_range_note(_flat_signed_fields(doc))
        # The standalone rule itself, so both Python verifiers refuse the same inputs.
        env = _vr.normalise_envelope(doc)
        same_fmt = isinstance(predecessor, dict) and predecessor_fmt == self.name
        return _vr._profile_range_refusal(
            _payload(doc),
            env.get("signature"),
            _payload(predecessor) if same_fmt else None,
            counterparty,
            env.get("anchors"),
        )
