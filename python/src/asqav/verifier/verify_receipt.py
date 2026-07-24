# Copyright 2026 Asqav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# This standalone verifier is deliberately licensed Apache-2.0 (a file-level
# exception to the Elastic-License-2.0 SDK) so it can ship in the exit artifact
# as a permanently free, dependency-light tool a customer runs offline forever.

"""Standalone Asqav receipt verifier - one dependency, mostly stdlib.

Verify an Asqav Compliance Receipt yourself, without the Asqav SDK or liboqs.

WHAT THIS CHECKS (independently, on your machine):
  - ML-DSA-65 (FIPS 204) signature over the receipt's canonical bytes
  - canonical-bytes integrity (JCS / RFC8785 reproduction, stdlib json)
  - hash-chain link to a predecessor receipt
  - anchor binding (which envelope each anchor commits to) plus anchor presence
  - issuer public-key resolution from the public /.well-known/jwks.json

WHAT THIS DOES NOT CHECK (needs server state or ASN.1; use the hosted /verify):
  - full RFC3161 certificate-chain walk
  - policy_digest artefact resolution

The only non-stdlib dependency is the signature check:
  pip install dilithium-py        (pure python; verify path uses stdlib SHAKE)
Without it, every other check still runs and the signature axis reports SKIPPED;
the overall verdict is then INCOMPLETE, never a PASS. This tool will not emit a
PASS unless the post-quantum signature was actually verified.

Security note: dilithium-py is a pure-python FIPS 204 implementation that is not
constant-time. For a verify-only tool this is acceptable: verification touches
only public data (public key, signature, message), so there is no secret to leak
through timing.

Run:
  python verify_receipt.py --id sig_abc123
  python verify_receipt.py --receipt receipt.json --jwks jwks.json --offline
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

API_BASE = "https://api.asqav.com/api/v1"
JWKS_URL = "https://api.asqav.com/.well-known/jwks.json"


def _user_agent() -> str:
    """Identify this tool; the api.asqav.com edge 403s the bare urllib default.

    Self-contained (no asqav import) so the file still runs as a standalone
    download, per the module docstring.
    """
    try:
        from importlib.metadata import version

        ver = version("asqav")
    except Exception:
        ver = "standalone"
    return f"asqav-python/{ver} (+https://www.asqav.com)"


USER_AGENT = _user_agent()
SKEW_BOUND_SECONDS = 300
FIRST_RECEIPT_SEED = "0" * 64  # mirrors core/integrity.py FIRST_RECEIPT_SEED
REQUIRED_FIELDS = (
    "type",
    "issued_at",
    "issuer_id",
    "action_ref",
    "payload_digest",
    "policy_digest",
    "previousReceiptHash",
    "decision",
)
ALLOWED_TYPES = {
    "protectmcp:decision",
    "protectmcp:restraint",
    "protectmcp:lifecycle",
    "protectmcp:lifecycle:configuration_change",
    # risk-acceptance / exception receipt; no-policy lifecycle opt-out.
    "protectmcp:lifecycle:risk_acceptance",
    # code-authorship receipt; producer-asserted record of who authored a code change.
    "protectmcp:lifecycle:code_authorship",
    "protectmcp:acknowledgment",
    "protectmcp:observation",
    "protectmcp:observation:result_bound",
}


#: Nesting depth above which a receipt is malformed input, not a crash; the
#: stdlib json encoder recurses per level and crashes past ~1000 on Python <= 3.12.
MAX_NESTING_DEPTH = 200


def canonical_json(obj) -> bytes:
    """JCS canonical bytes - byte-identical to the server's canonical_json.

    Sorted keys, no whitespace, UTF-8, NaN/Infinity rejected. This is the exact
    byte string the server signs and the verifier must reproduce.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _describe_value(value) -> str:
    """Short, safe description of a value's actual shape for an error message.

    Truncated so a huge string payload cannot blow up the error line itself.
    """
    if value is None:
        return "null"
    if isinstance(value, (bool, int, float, str)):
        text = f"{type(value).__name__} {value!r}"
        return text if len(text) <= 80 else text[:77] + "..."
    if isinstance(value, list):
        return f"list ({len(value)} item{'s' if len(value) != 1 else ''})"
    return type(value).__name__


def envelope_minus_anchors_jcs(env: dict) -> bytes:
    """JCS bytes of the envelope with ``anchors`` removed (the anchored bytes)."""
    e = dict(env)
    e.pop("anchors", None)
    return canonical_json(e)


def _scan_shape(obj, max_depth: int | None = None) -> str | None:
    """Iteratively walk ``obj``; returns "non_finite", "too_deep", or None (ok).

    Explicit stack, no recursion: depth alone cannot crash this walk. With
    ``max_depth`` set, nesting past it stops the walk and reports "too_deep"
    before the caller can hand the structure to the recursive stdlib json
    encoder (canonical_json), which crashes past ~1000 levels on Python's C
    encoder for versions <= 3.12.
    """
    stack = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if max_depth is not None and depth > max_depth:
            return "too_deep"
        if isinstance(cur, float):
            if not math.isfinite(cur):
                return "non_finite"
        elif isinstance(cur, dict):
            stack.extend((v, depth + 1) for v in cur.values())
        elif isinstance(cur, list):
            stack.extend((v, depth + 1) for v in cur)
    return None


def _contains_non_finite(obj) -> bool:
    """True if any float inside ``obj`` is NaN or Infinity.

    ``canonical_json`` rejects those (allow_nan=False) and the server never
    emits one, so a receipt carrying one is reported as a readable input error
    rather than leaking the json ValueError from a canonicalisation call.
    """
    return _scan_shape(obj) == "non_finite"


#: Shape-check outcome -> the readable message run()/run_structured() print.
_SHAPE_MESSAGES = {
    "non_finite": "receipt carries a non-finite number (NaN/Infinity); it cannot be canonicalised",
    "too_deep": f"receipt nesting exceeds the supported depth (> {MAX_NESTING_DEPTH} levels)",
}


class VerifierInputError(Exception):
    """A receipt or JWKS input was missing, empty, or not a JSON object.

    Raised so the CLI prints one readable line and exits nonzero rather than
    leaking a urllib/json traceback on bad input.
    """


def _parse_object(text: str, source: str) -> dict:
    """Parse ``text`` as a JSON object, or raise VerifierInputError.

    Rejects empty input and any non-object (array/string/number/null) so a
    later ``.get`` never lands on a non-dict.
    """
    if not text or not text.strip():
        raise VerifierInputError(f"{source}: empty input, expected a JSON object")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VerifierInputError(f"{source}: not valid JSON ({exc})") from exc
    except RecursionError as exc:
        # Some json decoder flavors (a pure-Python fallback) recurse per nesting
        # level; a too-deep input is an input error, never a raw crash.
        raise VerifierInputError(f"{source}: too deeply nested to parse ({exc})") from exc
    if not isinstance(value, dict):
        raise VerifierInputError(
            f"{source}: expected a JSON object, got {type(value).__name__}"
        )
    return value


def _get_json(url: str, *, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise VerifierInputError(
            f"{url}: server returned HTTP {exc.code} {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise VerifierInputError(
            f"{url}: could not reach the server ({exc.reason})"
        ) from exc
    return _parse_object(body, url)


def fetch_jwks(url: str = JWKS_URL, *, timeout: int = 30) -> dict:
    """Fetch and return the Asqav public JWKS directory as a dict.

    Snapshot this before going air-gapped; pass the result to
    ``verify_receipt_offline(receipt, jwks)`` or ``run(envelope, jwks, ...)``.
    The endpoint is public and unauthenticated.

    Args:
        url: JWKS URL (default: https://api.asqav.com/.well-known/jwks.json).
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JWKS dict with a ``"keys"`` list.
    """
    return _get_json(url, timeout=timeout)


def _b64decode(value: str) -> bytes:
    """Decode standard or url-safe base64, padding-tolerant."""
    s = value.replace("-", "+").replace("_", "/")
    s += "=" * ((-len(s)) % 4)
    return base64.b64decode(s, validate=False)


def _safe_b64(v: str) -> bool:
    try:
        _b64decode(v)
        return True
    except Exception:
        return False


def normalise_envelope(raw: dict) -> dict:
    """Remap a hosted /verify response into the canonical 3-key envelope.

    An Audit Pack already ships the canonical ``{payload, signature, anchors}``
    shape, so pass it through untouched. The hosted ``/verify/{id}`` JSON instead
    nests the signed dict under ``payload`` and exposes the signature object as
    ``signature_envelope`` (and a possibly flat-string ``signature``); rebuild
    the canonical envelope from those parts so ``run()`` sees one shape.
    """
    # Already canonical: a top-level signature object plus a payload dict.
    sig = raw.get("signature")
    if isinstance(sig, dict) and isinstance(raw.get("payload"), dict):
        return raw

    payload = raw.get("payload")
    if not isinstance(payload, dict):
        # Not a hosted response and not canonical; hand it back as-is so the
        # structure check reports the problem clearly.
        return raw

    sig_obj = raw.get("signature_envelope")
    if not isinstance(sig_obj, dict):
        if isinstance(sig, dict):
            sig_obj = sig
        else:
            sig_obj = {
                "alg": raw.get("algorithm", "ML-DSA-65"),
                "kid": payload.get("issuer_id", ""),
                "sig": sig if isinstance(sig, str) else "",
            }
    return {
        "payload": payload,
        "signature": sig_obj,
        # Preserve the raw anchors value (even a malformed {} / "" / 0) so
        # check_anchors sees it and can FAIL it, rather than laundering to [].
        "anchors": raw.get("anchors"),
    }


def resolve_key(jwks: dict, kid: str):
    """Return (public_key_bytes, status, alg) for the receipt's signature.kid.

    The public jwks directory lists each key under both its bare issuer id and
    its crypto key id; match either so the bare-kid wire form resolves. A
    non-list ``keys``, a non-dict entry, or a matched key with no usable
    ``public_key`` is a resolution failure (returns None), never a crash.
    """
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    if not isinstance(keys, list):
        return None, None, None
    for k in keys:
        if not isinstance(k, dict):
            continue
        if kid and kid in (k.get("issuer_id"), k.get("kid")):
            pub = k.get("public_key")
            if not isinstance(pub, str):
                continue
            return _b64decode(pub), k.get("status"), k.get("alg")
    return None, None, None


def resolve_revoked_at(jwks: dict, kid: str):
    """Return the JWKS revoked_at for kid if published, else None.

    Optional field: the public JWKS publishes status today but not the
    timestamp, so the key-status axis falls back to a bare status gate. Shares
    resolve_key's defensive shape so a malformed JWKS never crashes a direct call.
    """
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    if not isinstance(keys, list):
        return None
    for k in keys:
        if not isinstance(k, dict):
            continue
        if kid and kid in (k.get("issuer_id"), k.get("kid")):
            return k.get("revoked_at")
    return None


#: Key statuses that revoke trust in the signing key for attestation.
REVOKED_KEY_STATUSES = {"revoked", "suspended", "compromised"}


def check_key_status(status, issued_at: str, revoked_at=None, has_trusted_anchor: bool = False):
    """Gate the verdict on the signing key's published status.

    The public JWKS marks a key revoked once its agent is revoked. A receipt
    signed by such a key must not PASS offline, mirroring the hosted /verify.

    When the JWKS carries a precise revoked_at, prefer the at-or-before-issuance
    check so a receipt signed BEFORE revocation still PASSes (historical verify).
    Without a revoked_at the axis cannot place issuance, so any revoked key fails.

    has_trusted_anchor must be an anchor the caller CRYPTOGRAPHICALLY verified as
    pre-revocation timing; mere presence is not enough (anchors are unsigned and
    attacker-addable). Without one the self-attested issued_at is backdateable, so
    that case downgrades to SKIPPED (INCOMPLETE), never a hiding PASS. The offline
    verifier cannot do that walk and passes False; default False so a caller that
    ignores the parameter never hides behind a bare timestamp.
    """
    s = (status or "").lower()
    if s not in REVOKED_KEY_STATUSES:
        return "PASS", f"signing key status {status!r} is active"
    if revoked_at:
        try:
            rev = datetime.fromisoformat(str(revoked_at).replace("Z", "+00:00"))
            iss = datetime.fromisoformat(str(issued_at).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return "FAIL", f"signing key status {status!r}; unparseable revoked_at/issued_at"
        if rev.tzinfo is None:
            rev = rev.replace(tzinfo=timezone.utc)
        if iss.tzinfo is None:
            iss = iss.replace(tzinfo=timezone.utc)
        if rev <= iss:
            return "FAIL", f"signing key revoked at {revoked_at} on/before issuance {issued_at}"
        if not has_trusted_anchor:
            return (
                "SKIPPED",
                f"signing key revoked at {revoked_at}; issued_at {issued_at} is "
                f"self-attested, no anchor proves pre-revocation timing",
            )
        return "PASS", f"signing key revoked at {revoked_at}, after issuance {issued_at}"
    return "FAIL", f"signing key status {status!r}; receipt cannot be trusted"


def verify_signature(pk: bytes, msg: bytes, sig: bytes, alg: object):
    """ML-DSA-65 verify. Returns (result, note); result in PASS/FAIL/SKIPPED."""
    if (alg if isinstance(alg, str) else "").upper() != "ML-DSA-65":
        return "SKIPPED", f"unsupported alg {alg!r} (this tool checks ML-DSA-65)"
    try:
        from dilithium_py.ml_dsa import ML_DSA_65
    except ImportError:
        return "SKIPPED", "run 'pip install dilithium-py' for the post-quantum check"
    try:
        ok = ML_DSA_65.verify(pk, msg, sig)
        return ("PASS" if ok else "FAIL"), (
            "signature valid" if ok else "signature mismatch"
        )
    except Exception as exc:  # malformed key or signature bytes
        return "FAIL", f"verify error: {exc}"


def check_skew(issued_at: str):
    try:
        ts = datetime.fromisoformat(issued_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return "FAIL", f"unparseable issued_at {issued_at!r}"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    skew = (ts - datetime.now(timezone.utc)).total_seconds()
    if skew > SKEW_BOUND_SECONDS:
        return (
            "FAIL",
            f"issued_at {skew:.0f}s ahead of wall clock (> {SKEW_BOUND_SECONDS}s)",
        )
    return "PASS", f"skew {skew:.0f}s within bound"


def check_chain(payload: dict, predecessor_payload: dict | None):
    prev = payload.get("previousReceiptHash")
    if prev == FIRST_RECEIPT_SEED:
        return "PASS", "first receipt on chain (all-zero seed)"
    if predecessor_payload is None:
        return "SKIPPED", "no predecessor supplied (pass --predecessor to check)"
    try:
        actual = hashlib.sha256(canonical_json(predecessor_payload)).hexdigest()
    except RecursionError:
        # Defense in depth: the caller's depth-cap gate should already have
        # rejected this; this stops a bypassing call from crashing here too.
        return "FAIL", "predecessor payload too deeply nested to canonicalise"
    if actual == prev:
        return "PASS", "chain link rederives from predecessor"
    return "FAIL", f"chain break: expected {str(prev)[:16]}.. got {actual[:16]}.."


def check_anchors(envelope: dict):
    anchors = envelope.get("anchors")
    # Absent/null is a legitimate no-anchors receipt (SKIPPED); a present
    # non-list value ({} / "" / 0) is malformed and FAILs, never laundered to [].
    if anchors is None:
        return "SKIPPED", "no anchors on this receipt"
    if not isinstance(anchors, list):
        return "FAIL", f"anchors field is not a list (got {type(anchors).__name__})"
    if not anchors:
        return "SKIPPED", "no anchors on this receipt"
    try:
        bound = hashlib.sha256(envelope_minus_anchors_jcs(envelope)).hexdigest()
    except RecursionError:
        # Defense in depth, same rationale as check_chain's guard above.
        return "FAIL", "envelope too deeply nested to canonicalise for anchor binding"
    lines = [f"anchors bind envelope digest sha256:{bound[:16]}.."]
    all_ok = True
    for a in anchors:
        if not isinstance(a, dict):
            all_ok = False
            lines.append(f"    - malformed anchor entry (got {type(a).__name__}, expected an object)")
            continue
        atype = a.get("type", "?")
        val = a.get("value")
        ok = bool(val) and _safe_b64(val)
        all_ok = all_ok and ok
        state = "present, base64-ok" if ok else "MISSING or malformed"
        lines.append(f"    - {atype}: value {state}")
    return ("PASS" if all_ok else "FAIL"), "; ".join(lines)


def check_structure(payload: dict):
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return "FAIL", f"missing required fields: {','.join(missing)}"
    rt = payload.get("type")
    if rt not in ALLOWED_TYPES:
        return "FAIL", f"type {rt!r} outside the allowed namespace"
    return "PASS", f"required fields present; type {rt}"


def run(envelope: dict, jwks: dict, predecessor_payload: dict | None) -> int:
    if not isinstance(envelope, dict):
        print("Asqav receipt verification")
        print(
            f"  [FAIL] input       expected a JSON object receipt, got "
            f"{type(envelope).__name__}"
        )
        print("\n  => INCOMPLETE (no receipt object to verify; never a PASS)")
        return 2
    envelope = normalise_envelope(envelope)
    payload = envelope.get("payload", envelope)
    if not isinstance(payload, dict):
        # Hosted /verify can return ``payload: null`` (e.g. hash-only or
        # redacted receipts). Nothing to verify; say so instead of crashing.
        print("Asqav receipt verification")
        print(
            f"  [FAIL] payload      receipt payload not available from this "
            f"surface (got {_describe_value(payload)} instead of an object). "
            f"Verify with a saved receipt instead: --receipt FILE from your "
            f"Audit Pack or SDK capture."
        )
        print("\n  => INCOMPLETE (no payload to verify; never a PASS)")
        return 2
    shape = _scan_shape(envelope, max_depth=MAX_NESTING_DEPTH)
    if shape is None:
        shape = _scan_shape(predecessor_payload, max_depth=MAX_NESTING_DEPTH)
    if shape is not None:
        print("Asqav receipt verification")
        print(f"  [FAIL] input       {_SHAPE_MESSAGES[shape]}")
        print("\n  => INCOMPLETE (receipt not canonicalisable; never a PASS)")
        return 2
    sig_obj = envelope.get("signature", {})
    if isinstance(sig_obj, str):  # flat-string signature, derive the object
        sig_obj = {
            "alg": envelope.get("algorithm", "ML-DSA-65"),
            "kid": payload.get("issuer_id", ""),
            "sig": sig_obj,
        }
    elif not isinstance(sig_obj, dict):
        # A non-object signature (list, number, ...) carries no kid/sig; key
        # resolution then FAILs cleanly instead of .get on a non-dict.
        sig_obj = {}
    kid = sig_obj.get("kid", "")
    alg = sig_obj.get("alg", "ML-DSA-65")
    try:
        msg = canonical_json(payload)
    except RecursionError:
        # Defense in depth: the shape gate above should already have caught
        # this; this stops a bypassing caller from crashing here too.
        print("Asqav receipt verification")
        print(f"  [FAIL] input       {_SHAPE_MESSAGES['too_deep']}")
        print("\n  => INCOMPLETE (receipt not canonicalisable; never a PASS)")
        return 2

    results = []
    results.append(("structure", *check_structure(payload)))

    pk, status, jwks_alg = resolve_key(jwks, kid)
    if pk is None:
        results.append(("issuer_key", "FAIL", f"kid {kid!r} not in jwks directory"))
        results.append(("signature", "SKIPPED", "no issuer key to verify against"))
    else:
        results.append(("issuer_key", "PASS", f"resolved kid {kid} (status={status})"))
        revoked_at = resolve_revoked_at(jwks, kid)
        # Offline anchor presence is unverifiable (anchors are unsigned); pass
        # False so a forged anchor never rides a revoked key to PASS.
        results.append(
            ("key_status", *check_key_status(status, payload.get("issued_at", ""), revoked_at, False))
        )
        _raw_sig = sig_obj.get("sig", "")
        sig = _b64decode(_raw_sig) if isinstance(_raw_sig, str) else b""
        results.append(("signature", *verify_signature(pk, msg, sig, alg or jwks_alg)))

    results.append(("chain", *check_chain(payload, predecessor_payload)))
    results.append(("anchors", *check_anchors(envelope)))
    results.append(("skew", *check_skew(payload.get("issued_at", ""))))

    print("Asqav receipt verification")
    print(f"  canonical bytes: sha256:{hashlib.sha256(msg).hexdigest()}")
    print(f"  signature.kid:   {kid}")
    for name, res, note in results:
        mark = {"PASS": "ok", "FAIL": "FAIL", "SKIPPED": "skip"}[res]
        print(f"  [{mark:>4}] {name:<11} {note}")

    has_fail = any(r == "FAIL" for _, r, _ in results)
    # A skipped chain (no predecessor supplied) is expected and does not block a
    # PASS; any other skip - notably the signature - downgrades to INCOMPLETE.
    has_blocking_skip = any(r == "SKIPPED" and n != "chain" for n, r, _ in results)
    if has_fail:
        verdict, code = "FAIL", 1
    elif has_blocking_skip:
        verdict, code = "INCOMPLETE (some checks skipped; never a PASS)", 2
    else:
        verdict, code = "PASS", 0
    print(f"\n  => {verdict}")
    return code


def run_structured(
    envelope: dict,
    jwks: dict,
    predecessor_payload: dict | None = None,
) -> dict:
    """Verify a receipt offline and return a structured result dict.

    Same logic as ``run()`` but returns a dict instead of printing and exiting.
    Callable directly from the verifier module; the public SDK uses the oracle
    adapter path for multi-format support.

    Returns:
        dict with keys:
          - ``"verdict"``: "PASS" | "FAIL" | "INCOMPLETE"
          - ``"axes"``: list of ``{"name": str, "result": str, "note": str}``
          - ``"canonical_sha256"``: hex SHA-256 of the canonical payload bytes
          - ``"kid"``: the signature key id resolved
    """
    if not isinstance(envelope, dict):
        return {
            "verdict": "INCOMPLETE",
            "axes": [
                {
                    "name": "input",
                    "result": "FAIL",
                    "note": (
                        "expected a JSON object receipt, got "
                        f"{type(envelope).__name__}"
                    ),
                }
            ],
            "canonical_sha256": None,
            "kid": None,
        }
    envelope = normalise_envelope(envelope)
    payload = envelope.get("payload", envelope)
    if not isinstance(payload, dict):
        return {
            "verdict": "INCOMPLETE",
            "axes": [
                {
                    "name": "payload",
                    "result": "FAIL",
                    "note": (
                        "receipt payload not available from this surface "
                        f"(got {_describe_value(payload)} instead of an object). "
                        "Verify with a saved receipt instead."
                    ),
                }
            ],
            "canonical_sha256": None,
            "kid": None,
        }
    shape = _scan_shape(envelope, max_depth=MAX_NESTING_DEPTH)
    if shape is None:
        shape = _scan_shape(predecessor_payload, max_depth=MAX_NESTING_DEPTH)
    if shape is not None:
        return {
            "verdict": "INCOMPLETE",
            "axes": [{"name": "input", "result": "FAIL", "note": _SHAPE_MESSAGES[shape]}],
            "canonical_sha256": None,
            "kid": None,
        }

    sig_obj = envelope.get("signature", {})
    if isinstance(sig_obj, str):
        sig_obj = {
            "alg": envelope.get("algorithm", "ML-DSA-65"),
            "kid": payload.get("issuer_id", ""),
            "sig": sig_obj,
        }
    elif not isinstance(sig_obj, dict):
        sig_obj = {}  # non-object signature: no usable kid/sig, key resolution FAILs cleanly
    kid = sig_obj.get("kid", "")
    alg = sig_obj.get("alg", "ML-DSA-65")
    try:
        msg = canonical_json(payload)
    except RecursionError:
        # Defense in depth; the shape gate above should already have caught this.
        return {
            "verdict": "INCOMPLETE",
            "axes": [{"name": "input", "result": "FAIL", "note": _SHAPE_MESSAGES["too_deep"]}],
            "canonical_sha256": None,
            "kid": None,
        }

    axes: list[dict] = []
    axes.append(
        {
            "name": "structure",
            "result": check_structure(payload)[0],
            "note": check_structure(payload)[1],
        }
    )

    pk, status, jwks_alg = resolve_key(jwks, kid)
    if pk is None:
        axes.append(
            {
                "name": "issuer_key",
                "result": "FAIL",
                "note": f"kid {kid!r} not in jwks directory",
            }
        )
        axes.append(
            {
                "name": "signature",
                "result": "SKIPPED",
                "note": "no issuer key to verify against",
            }
        )
    else:
        axes.append(
            {
                "name": "issuer_key",
                "result": "PASS",
                "note": f"resolved kid {kid} (status={status})",
            }
        )
        revoked_at = resolve_revoked_at(jwks, kid)
        # Offline anchor presence is not trusted timing; pass False so a forged
        # anchor never upgrades a revoked key to PASS (hosted /verify does).
        ks_r, ks_n = check_key_status(
            status, payload.get("issued_at", ""), revoked_at, False
        )
        axes.append({"name": "key_status", "result": ks_r, "note": ks_n})
        _raw_sig = sig_obj.get("sig", "")
        sig_bytes = _b64decode(_raw_sig) if isinstance(_raw_sig, str) else b""
        sig_r, sig_n = verify_signature(pk, msg, sig_bytes, alg or jwks_alg)
        axes.append({"name": "signature", "result": sig_r, "note": sig_n})

    chain_r, chain_n = check_chain(payload, predecessor_payload)
    axes.append({"name": "chain", "result": chain_r, "note": chain_n})
    anch_r, anch_n = check_anchors(envelope)
    axes.append({"name": "anchors", "result": anch_r, "note": anch_n})
    skew_r, skew_n = check_skew(payload.get("issued_at", ""))
    axes.append({"name": "skew", "result": skew_r, "note": skew_n})

    has_fail = any(a["result"] == "FAIL" for a in axes)
    has_blocking_skip = any(
        a["result"] == "SKIPPED" and a["name"] != "chain" for a in axes
    )
    if has_fail:
        verdict = "FAIL"
    elif has_blocking_skip:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "axes": axes,
        "canonical_sha256": hashlib.sha256(msg).hexdigest(),
        "kid": kid,
    }


def _load(path: str) -> dict:
    if path == "-":
        return _parse_object(sys.stdin.read(), "stdin")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise VerifierInputError(f"{path}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise VerifierInputError(f"{path}: not valid UTF-8 text ({exc})") from exc
    return _parse_object(text, path)


def main() -> int:
    p = argparse.ArgumentParser(description="Standalone Asqav receipt verifier.")
    p.add_argument("--id", help="signature id to fetch from api.asqav.com")
    p.add_argument("--receipt", help="path to a saved receipt JSON, or - for stdin")
    p.add_argument("--jwks", help="path to a saved jwks.json, or - for stdin (offline)")
    p.add_argument(
        "--predecessor", help="path to predecessor receipt JSON for the chain check"
    )
    p.add_argument("--offline", action="store_true", help="never reach the network")
    args = p.parse_args()

    try:
        if args.receipt:
            envelope = _load(args.receipt)
        elif args.id and not args.offline:
            envelope = _get_json(f"{API_BASE}/verify/{args.id}")
        else:
            p.error("supply --receipt FILE (or -), or --id ID without --offline")
            return 2

        if args.jwks:
            jwks = _load(args.jwks)
        elif not args.offline:
            jwks = _get_json(JWKS_URL)
        else:
            p.error("offline mode needs --jwks FILE")
            return 2

        predecessor_payload = None
        if args.predecessor:
            pred = _load(args.predecessor)
            predecessor_payload = pred.get("payload", pred)
    except VerifierInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return run(envelope, jwks, predecessor_payload)


if __name__ == "__main__":
    sys.exit(main())
