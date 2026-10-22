"""Offline verification of a whole exported Audit Pack; every check
reuses verify_receipt. Nothing here reaches the network."""

from __future__ import annotations

import hashlib
from typing import Any

from asqav.verifier import verify_receipt as vr

VERIFIED = "verified"
NOT_OFFLINE = "not_verifiable_offline"
INVALID = "invalid"
#: Pack status when nothing failed but the keys came from the pack (no --jwks).
SELF_CONSISTENT = "self_consistent"
SELF_CONSISTENT_LABEL = "self-consistent, not anchored to a trusted key"

#: Exit code per pack status: 0 verified under supplied keys, 1 invalid, else 2.
EXIT_CODES = {VERIFIED: 0, NOT_OFFLINE: 2, SELF_CONSISTENT: 2, INVALID: 1}
_RANK = {VERIFIED: 0, NOT_OFFLINE: 1, INVALID: 2}
_BUNDLE_ALGS = frozenset({"ML-DSA-65", "ED25519", "ES256"})

#: Optional pack members the bundle signature covers whenever they are carried.
_PRESENCE_MEMBERS = (
    "content_disclosure",
    "regime_mapping_disclaimer",
    "regime_provenance",
    "jwks",
    "exit_manifest",
    "tsa_trust",
)

_FLAT_REASON = (
    "non-compliance-mode receipt: it has no signed receipt envelope "
    "(issuer, chain, expiry), so it cannot be verified offline"
)
_WITHHELD_REASON = (
    "signed bytes withheld at export; re-export with --include-signed-bytes to verify it"
)
_UNVOUCHED_REASON = "bundle signature not verified, so nothing vouches for this entry"


def signed_bundle_payload(pack: dict) -> dict:
    """The pack projection the bundle signature covers (export contract)."""
    receipts = pack.get("receipts") or []
    payload = {
        "organization_id": pack.get("organization_id"),
        "start": pack.get("start"),
        "end": pack.get("end"),
        "agent_id": pack.get("agent_id"),
        "only_compliance": pack.get("only_compliance"),
        "algorithm_registry_version": pack.get("algorithm_registry_version"),
        "receipt_count": pack.get("receipt_count", len(receipts)),
        "receipts": receipts,
        "revocation_manifest": pack.get("revocation_manifest") or [],
        "regime_mapping": pack.get("regime_mapping") or {},
    }
    for key in _PRESENCE_MEMBERS:
        if pack.get(key) is not None:
            payload[key] = pack[key]
    if pack.get("attested_framework_coverage"):
        payload["attested_framework_coverage"] = pack["attested_framework_coverage"]
    return payload


def pack_tsa_count(pack: dict) -> int:
    """TSA entries the pack carries; reported only, never used as trust roots.
    """
    raw = pack.get("tsa_trust")
    return len(raw) if isinstance(raw, list) else int(bool(raw))


def _key_bytes(entry) -> bytes | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("public_key") or entry.get("pub")
    try:
        return vr._b64decode(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _jwks_keys(jwks) -> list:
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    return [k for k in keys if isinstance(k, dict)] if isinstance(keys, list) else []


def _outcome(status: str, reason: str, thumbprint: str | None = None) -> dict:
    return {"status": status, "reason": reason, "key_thumbprint": thumbprint}


def _thumbprint(entry: dict, alg: str, key: bytes) -> str | None:
    try:
        return entry.get("key_thumbprint") or vr.thumbprint_for_key(alg=alg, public_key=key)
    except (ValueError, TypeError, KeyError):
        return None


def _bundle_material(pack: dict):
    """(message, key, signature, alg) or an invalid outcome; a missing
member is never benign."""
    digest, sig, key_b64 = (pack.get(k) for k in ("bundle_digest", "bundle_signature", "bundle_public_key"))
    if not all(isinstance(v, str) and v for v in (digest, sig, key_b64)):
        return _outcome(INVALID, "pack carries no bundle signature")
    alg = pack.get("bundle_signature_algorithm")
    if not isinstance(alg, str) or alg.upper() not in _BUNDLE_ALGS:
        return _outcome(INVALID, f"unsupported bundle signature algorithm {alg!r}")
    try:
        msg = vr.canonical_json(signed_bundle_payload(pack))
        key, sig_bytes = vr._b64decode(key_b64), vr._b64decode(sig)
    except (ValueError, TypeError, RecursionError) as exc:
        return _outcome(INVALID, f"bundle cannot be read: {exc}")
    if "sha256:" + hashlib.sha256(msg).hexdigest() != digest:
        return _outcome(INVALID, "bundle_digest does not match the pack contents")
    return msg, key, sig_bytes, alg


def _revocation_outcome(pack: dict, entry: dict, trusted_times: list) -> dict:
    """Not verifiable when a key was revoked after signing; invalid only
when signing after the revocation is proven."""
    label = str(entry.get("status") or "revoked").lower()
    when = entry.get("revoked_at") or entry.get("suspended_at")
    rev = vr._parse_stamp(when) if when else None
    if rev is not None:
        start = vr._parse_stamp(pack.get("start"))
        if start is not None and start >= rev:
            return _outcome(INVALID, f"bundle key {label} at {when}, before the signed window start")
        if any(t >= rev for t in trusted_times):
            return _outcome(INVALID, f"bundle key {label} at {when}; a receipt in the bundle has a "
                            "trusted timestamp after that")
    return _outcome(
        NOT_OFFLINE, f"bundle key {label} at {when or 'an unknown time'}; "
        "nothing proves the bundle was signed before it"
    )


def _bundle_key_problem(pack: dict, entry: dict | None, alg: str, trusted_times: list):
    if entry is None:
        return _outcome(INVALID, "bundle key is not in the key set")
    key_alg = entry.get("alg")
    if isinstance(key_alg, str) and key_alg.upper() != alg.upper():
        return _outcome(INVALID, f"bundle algorithm {alg} differs from its key's alg {key_alg}")
    org = pack.get("organization_id")
    if org and entry.get("org_id") and entry.get("org_id") != org:
        return _outcome(INVALID, "bundle signing key belongs to another organization")
    status = str(entry.get("status") or "").lower()
    if vr.key_status_kind(status, entry.get("revocation_reason")) == "compromised":
        return _outcome(INVALID, "bundle signing key compromised")
    if not status or status == "unknown":
        return _outcome(NOT_OFFLINE, "bundle signing key status unknown")
    if status in vr.REVOKED_KEY_STATUSES or entry.get("revoked_at") or entry.get("suspended_at"):
        return _revocation_outcome(pack, entry, trusted_times)
    return None


def mapping_issues(pack: dict[str, Any]) -> list[str]:
    """Pack-level reasons when mapping references resolve to no entry.

    Draft-09: every reference resolves to exactly one entry; count matches.
    """
    receipts = pack.get("receipts")
    if not isinstance(receipts, list):
        receipts = []
    known = {
        e.get("signature_id") for e in receipts
        if isinstance(e, dict) and isinstance(e.get("signature_id"), str)
    }
    issues = []
    if "receipt_count" in pack:
        count = pack["receipt_count"]
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count != len(receipts)
        ):
            issues.append("receipt_count does not match the number of receipts")
    mapping = pack.get("regime_mapping")
    if not isinstance(mapping, dict):
        return issues
    for regime, ids in mapping.items():
        if not isinstance(ids, list):
            continue
        for ref in ids:
            if not isinstance(ref, str) or ref not in known:
                issues.append(
                    f"regime_mapping {regime!r} references unknown signature_id {ref!r}"
                )
    return issues


def check_bundle(pack: dict, jwks: dict, trusted_times: list | None = None) -> dict:
    """Recompute the bundle digest, verify its signature, and tie the key
to the key set; pinned TSA anchor times place a revocation in time."""
    if vr._holds_unsafe_integer(pack, jwks):
        # Raw-text ingest refuses such a pack; so does a parsed dict.
        return _outcome(NOT_OFFLINE, f"pack: {vr._UNSAFE_INTEGER_INPUT_NOTE}")
    dangling = mapping_issues(pack)
    if dangling:
        return _outcome(INVALID, f"pack: {dangling[0]}")
    material = _bundle_material(pack)
    if isinstance(material, dict):
        return material
    msg, key, sig_bytes, alg = material
    entry = next((k for k in _jwks_keys(jwks) if _key_bytes(k) == key), None)
    problem = _bundle_key_problem(pack, entry, alg, list(trusted_times or []))
    if problem is not None:
        return problem
    result, note = vr.verify_signature(key, msg, sig_bytes, alg)
    if result != "PASS":
        return _outcome(INVALID if result == "FAIL" else NOT_OFFLINE, f"bundle signature: {note}")
    thumb = _thumbprint(entry, alg, key)
    return _outcome(VERIFIED, f"bundle signature valid under key {entry.get('kid')}", thumb)


def _seq(payload: dict) -> int | None:
    seq = payload.get("seq")
    return seq if isinstance(seq, int) and not isinstance(seq, bool) else None


def _agent(entry: dict) -> str:
    return str(entry["payload"].get("agent_id") or entry.get("agent_id") or "")


def _link_hash(entry: dict) -> str | None:
    if entry.get("signed_bytes_withheld"):
        value = entry.get("message_sha256")
        return value if isinstance(value, str) else None
    try:
        return hashlib.sha256(vr.canonical_json(entry["payload"])).hexdigest()
    except (ValueError, TypeError, RecursionError):
        return None


def _envelopes(receipts: list) -> dict:
    return {
        i: e for i, e in enumerate(receipts)
        if isinstance(e, dict) and isinstance(e.get("payload"), dict)
    }


def _link(i: int, entry: dict, by_seq: dict, by_hash: dict, envelopes: dict) -> tuple:
    """(predecessor payload, chain issue) for one envelope receipt."""
    payload, seq = entry["payload"], _seq(entry["payload"])
    prev, agent = payload.get("previousReceiptHash"), _agent(entry)
    if prev == vr.FIRST_RECEIPT_SEED:
        return None, None
    if seq is not None and len(by_seq.get((agent, seq - 1), [])) == 1:
        return envelopes[by_seq[(agent, seq - 1)][0]]["payload"], None
    j = by_hash.get((agent, prev))
    if j is not None:
        pred = envelopes[j]
        return (None if pred.get("signed_bytes_withheld") else pred["payload"]), None
    if seq is not None and any(a == agent and s < seq for a, s in by_seq):
        return None, f"chain: gap, the predecessor of seq {seq} is missing from the pack"
    return None, None


def _index(envelopes: dict) -> tuple:
    by_seq: dict[tuple, list[int]] = {}
    by_hash: dict[tuple, int] = {}
    for i, e in envelopes.items():
        seq = None if e.get("signed_bytes_withheld") else _seq(e["payload"])
        if seq is not None:
            by_seq.setdefault((_agent(e), seq), []).append(i)
        h = _link_hash(e)
        if h is not None:
            by_hash[(_agent(e), h)] = i
    return by_seq, by_hash


def _duplicates(pairs) -> dict:
    groups: dict = {}
    for i, key in pairs:
        if key is not None:
            groups.setdefault(key, []).append(i)
    return {i: key for key, idxs in groups.items() if len(idxs) > 1 for i in idxs}


def chain_issues(receipts: list, envelopes: dict, by_seq: dict) -> dict:
    """Index -> invalid reasons for forked chains and duplicate entries."""
    issues: dict[int, list[str]] = {}
    forks = {i: k for k, idxs in by_seq.items() if len(idxs) > 1 for i in idxs}
    prevs = _duplicates(
        (i, (_agent(e), e["payload"].get("previousReceiptHash"))) for i, e in envelopes.items()
    )
    ids = _duplicates(
        (i, e.get("signature_id")) for i, e in enumerate(receipts) if isinstance(e, dict)
    )
    for i, (agent, seq) in forks.items():
        issues.setdefault(i, []).append(f"chain: fork, seq {seq} appears twice for agent {agent}")
    for i, (agent, _) in prevs.items():
        issues.setdefault(i, []).append(f"chain: fork, two receipts of {agent} share a predecessor")
    for i, sid in ids.items():
        issues.setdefault(i, []).append(f"duplicate entry: signature_id {sid}")
    return issues


def chain_context(receipts: list) -> tuple:
    """(index -> predecessor payload, index -> invalid reasons) for its
receipts."""
    envelopes = _envelopes(receipts)
    by_seq, by_hash = _index(envelopes)
    issues = chain_issues(receipts, envelopes, by_seq)
    preds: dict[int, dict | None] = {}
    for i, e in envelopes.items():
        preds[i], issue = _link(i, e, by_seq, by_hash, envelopes)
        if issue:
            issues.setdefault(i, []).append(issue)
    return preds, issues


def _expiry_note(payload: dict, axes: list) -> str | None:
    axis = next((a for a in axes if a["name"] == "expiry"), None)
    if axis is None or axis["result"] != "FAIL" or "lapsed" not in axis["note"]:
        return None
    issued, expires = vr._parse_stamp(payload.get("issued_at")), vr._parse_stamp(payload.get("expires_at"))
    if issued is not None and expires is not None and issued <= expires:
        return f"expires_at {payload.get('expires_at')} has passed; signed before it, so not a failure"
    return f"expires_at {payload.get('expires_at')} is not after issued_at"


def _blocking_reasons(status: str, axes: list) -> list:
    if status == INVALID:
        rows = [a for a in axes if a["failure_class"] == INVALID and a["name"] != "expiry"]
    else:
        rows = [
            a for a in axes
            if a["result"] != "PASS" and a["name"] != "expiry"
            and not (a["name"] == "chain" and a["result"] == "SKIPPED")
        ]
    return [f"{a['name']}: {' '.join(a['note'].split())}" for a in rows]


def _envelope_result(entry: dict, jwks: dict, predecessor, tsa_keys, headers) -> dict:
    res = vr.run_structured(
        entry, jwks, predecessor, trusted_tsa_keys=tsa_keys, bitcoin_headers=headers
    )
    if res["verdict"] in (vr.VERDICT_VERIFIED, vr.VERDICT_VERIFIED_KEYED):
        status = VERIFIED
    else:
        status = INVALID if res["failure_class"] == vr.FAILURE_INVALID else NOT_OFFLINE
    axis = {a["name"]: a for a in res["axes"]}
    payload = entry.get("payload")
    expiry = _expiry_note(payload if isinstance(payload, dict) else {}, res["axes"])
    return {
        "status": status,
        "verdict": res["verdict"],
        "reasons": _blocking_reasons(status, res["axes"]),
        "signature": axis["signature"]["note"] if "signature" in axis else None,
        "chain": axis["chain"]["note"] if "chain" in axis else None,
        "notes": [expiry] if expiry else [],
        "axes": res["axes"],
    }


def _bare(status: str, reason: str) -> dict:
    return {"status": status, "reasons": [reason], "signature": None, "chain": None, "notes": []}


def _flat_signature_results(entry: dict, jwks: dict) -> list:
    agent = entry.get("agent_id")
    keys = [k for k in _jwks_keys(jwks) if agent and k.get("agent_id") == agent]
    msg, sig = entry.get("message_b64"), entry.get("signature_b64")
    if not (keys and isinstance(msg, str) and msg and isinstance(sig, str) and sig):
        return []
    try:
        msg_bytes, sig_bytes = vr._b64decode(msg), vr._b64decode(sig)
    except ValueError:
        return []
    alg = entry.get("algorithm")
    return [vr.verify_signature(_key_bytes(k) or b"", msg_bytes, sig_bytes, alg)[0] for k in keys]


def _flat_result(entry: dict, jwks: dict) -> dict:
    """A flat receipt is never verified offline; a signature its agent's
keys all reject is invalid."""
    out = _bare(NOT_OFFLINE, _FLAT_REASON)
    results = _flat_signature_results(entry, jwks)
    if "PASS" in results:
        out["signature"] = "signature valid under the agent's published key"
    elif results and all(r == "FAIL" for r in results):
        out["status"] = INVALID
        out["reasons"] = [f"signature does not verify under any key of agent {entry.get('agent_id')}"]
    return out


def verify_entry(entry, jwks: dict, predecessor=None, *, tsa_keys=None, headers=None) -> dict:
    """Status, reasons and notes for one pack entry."""
    if not isinstance(entry, dict):
        return _bare(INVALID, "entry is not a JSON object")
    head = {"signature_id": entry.get("signature_id"), "agent_id": entry.get("agent_id")}
    if entry.get("signed_bytes_withheld"):
        return {**head, **_bare(NOT_OFFLINE, _WITHHELD_REASON)}
    if isinstance(entry.get("payload"), dict):
        return {**head, **_envelope_result(entry, jwks, predecessor, tsa_keys, headers)}
    if entry.get("compliance_mode") is False:
        return {**head, **_flat_result(entry, jwks)}
    return {**head, **_envelope_result(entry, jwks, None, tsa_keys, headers)}


def trusted_anchor_times(receipts: list, tsa_keys: list, headers) -> list:
    """Anchor times the pinned TSA keys or supplied block data prove,
across the pack."""
    times: list = []
    for entry in receipts:
        if isinstance(entry, dict) and isinstance(entry.get("payload"), dict):
            env = vr.normalise_envelope(entry)
            ev = vr.evaluate_anchors(env, trusted_tsa_keys=tsa_keys, bitcoin_headers=headers)
            times.extend(ev.trusted_times)
    return times


def _worst(statuses) -> str:
    return max(statuses, key=_RANK.__getitem__, default=VERIFIED)


def _apply(result: dict, issues: list, bundle_ok: bool) -> dict:
    if issues:
        return {**result, "status": INVALID, "reasons": issues + result["reasons"]}
    if result["status"] == VERIFIED and not bundle_ok:
        return {**result, "status": NOT_OFFLINE, "reasons": [_UNVOUCHED_REASON]}
    return result


def verify_pack(pack: dict, *, jwks: dict | None = None, tsa_keys=None, bitcoin_headers=None) -> dict:
    """Per-receipt and pack report. Trust comes only from ``jwks`` and
``tsa_keys``; without ``jwks`` the pack snapshot is self_consistent."""
    anchored = isinstance(jwks, dict)
    key_set = jwks if anchored else pack.get("jwks")
    key_set = key_set if isinstance(key_set, dict) else {"keys": []}
    pinned = list(tsa_keys or [])
    receipts = pack.get("receipts") if isinstance(pack.get("receipts"), list) else []
    preds, issues = chain_context(receipts)
    bundle = check_bundle(pack, key_set, trusted_anchor_times(receipts, pinned, bitcoin_headers))
    results = [
        _apply(
            verify_entry(r, key_set, preds.get(i), tsa_keys=pinned, headers=bitcoin_headers),
            issues.get(i, []),
            bundle["status"] == VERIFIED,
        )
        for i, r in enumerate(receipts)
    ]
    counts = {s: sum(1 for r in results if r["status"] == s) for s in (VERIFIED, NOT_OFFLINE, INVALID)}
    status = _worst([bundle["status"]] + [r["status"] for r in results])
    if status != INVALID and not anchored:
        status = SELF_CONSISTENT
    return {
        "status": status,
        "exit_code": EXIT_CODES[status],
        "bundle": bundle,
        "receipts": results,
        "summary": {"total": len(results), **counts},
        "trust": {
            "anchored": anchored,
            "keys": "supplied key set" if anchored else "pack key snapshot (self-asserted)",
            "key_count": len(_jwks_keys(key_set)),
            "tsa_keys": len(pinned),
            "tsa_carried_not_trusted": pack_tsa_count(pack),
        },
        "pack": {
            "organization_id": pack.get("organization_id"),
            "start": pack.get("start"),
            "end": pack.get("end"),
            "bundle_digest": pack.get("bundle_digest"),
        },
    }
