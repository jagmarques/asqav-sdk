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

CHECKS (independently, on your machine):
  - ML-DSA-65 (FIPS 204) signature over the receipt's canonical bytes
  - canonical-bytes integrity (JCS / RFC8785 reproduction, stdlib json)
  - hash-chain link to a predecessor receipt
  - expiry (the expires_at the signer committed to inside the signed bytes)
  - issuer key resolution from the public /.well-known/jwks.json, and issuer
    binding (the verifying key is published under the claimed issuer_id)
  - anchor binding AND each anchor's own proof: an RFC 3161 token must commit
    sha256(JCS(envelope minus anchors)) in its messageImprint and verify against
    caller-pinned TSA keys (--tsa-key); an OpenTimestamps proof must commit the
    same digest and, given --bitcoin-headers, land its merkle path in the stated
    block. Presence alone never PASSes the axis.

DOES NOT CHECK (needs server state or extra trust material): a TSA
certificate-chain walk to a public root - offline trust comes only from the TSA
keys you pin, never from the unsigned envelope; OpenTimestamps block placement
without a caller-supplied header source; policy_digest artefact resolution.

Only the signature checks need non-stdlib code:
  pip install dilithium-py        (pure python; verify path uses stdlib SHAKE)
  pip install cryptography        (pinned RSA/ECDSA/Ed25519 TSA certificates)
Without either, the matching axis reports SKIPPED and the verdict is INCOMPLETE:
this tool never emits a PASS on an unverified post-quantum signature, nor on
anchor presence alone. dilithium-py is not constant-time, which is fine for a
verify-only tool: it touches only public data (public key, signature, message),
so there is no secret to leak through timing.

Run:
  python verify_receipt.py --id sig_abc123
  python verify_receipt.py --receipt receipt.json --jwks jwks.json --offline
  python verify_receipt.py --receipt receipt.json --jwks jwks.json --offline \
      --tsa-key asqav-tsa-chain.pem   # pinned TSA material: anchors can verify
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import re
import sys
import urllib.error
import urllib.request
from collections import namedtuple
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


#: Closed controls_evaluated key set; mirrors the client false-attestation guard.
ALLOWED_CONTROL_KEYS = frozenset(
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

#: Bare 64-hex (no prefix); the quorum attestation_hash form.
_BARE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Nesting depth above which a receipt is malformed input, not a crash; the
#: stdlib json encoder recurses per level and crashes past ~1000 on Python <= 3.12.
MAX_NESTING_DEPTH = 200


#: Instant from which the issuing platform emits RFC 8785 member order on the wire. Pinned to
#: the production deploy of the emitter change; receipts issued later never get the retry below.
JCS_UTF16_CUTOVER = "2026-09-02T18:05:09+00:00"


def _utf16_member_order(name: str) -> bytes:
    # Big-endian UTF-16 bytes compare in the same order as the code units they encode.
    return name.encode("utf-16-be")


def _member_name(key) -> str:
    # The member name json.dumps would emit for a non-string key.
    if key is True:
        return "true"
    if key is False:
        return "false"
    if key is None:
        return "null"
    if isinstance(key, float):
        return repr(key)
    return str(key)


def _utf16_ordered(obj):
    # Rebuild obj with every object's members in RFC 8785 section 3.2.3 order.
    if isinstance(obj, dict):
        named = [(k if isinstance(k, str) else _member_name(k), k) for k in obj]
        named.sort(key=lambda pair: _utf16_member_order(pair[0]))
        return {name: _utf16_ordered(obj[original]) for name, original in named}
    if isinstance(obj, list):
        return [_utf16_ordered(v) for v in obj]
    return obj


def canonical_json(obj, default=None) -> bytes:
    """JCS canonical bytes - byte-identical to the server's canonical_json.

    Member names in UTF-16 code-unit order (RFC 8785 section 3.2.3), no whitespace,
    UTF-8, NaN/Infinity rejected. json.dumps(sort_keys=True) orders by code point
    instead, which diverges above U+FFFF, so this sorts explicitly. This is the exact
    byte string the server signs and the verifier must reproduce.
    """
    return json.dumps(
        _utf16_ordered(obj),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=default,
    ).encode("utf-8")


def canonical_json_pre_cutover(obj) -> bytes:
    """The code-point member order the platform emitted before JCS_UTF16_CUTOVER.

    Diagnostic only: a signature that verifies solely under these bytes is reported
    as the pre-cutover dialect and never as verified.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def has_supplementary_member_name(obj) -> bool:
    """True when any object member name, at any depth, carries a character above U+FFFF."""
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and any(ord(ch) > 0xFFFF for ch in k):
                    return True
                stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    return False


def _issued_before_cutover(payload: dict) -> bool:
    # An unreadable issued_at earns no retry; the receipt is simply not verified.
    issued = _parse_stamp(payload.get("issued_at", ""))
    cutover = _parse_stamp(JCS_UTF16_CUTOVER)
    return issued is not None and cutover is not None and issued < cutover


def _pre_cutover_diagnostic(payload: dict, sig: bytes, candidates, sig_res):
    """Name the pre-cutover dialect when that is the only reason a signature fails.

    Runs only for a receipt issued before JCS_UTF16_CUTOVER whose member names reach
    above U+FFFF. The result stays FAIL: the dialect is reported, never accepted.
    """
    if not has_supplementary_member_name(payload) or not _issued_before_cutover(payload):
        return sig_res
    try:
        legacy_msg = canonical_json_pre_cutover(payload)
    except (TypeError, ValueError, RecursionError):
        return sig_res
    for pk, alg in candidates:
        if pk is not None and verify_signature(pk, legacy_msg, sig, alg)[0] == "PASS":
            return (
                "FAIL",
                "pre-cutover dialect: the signature verifies only under code-point member "
                "order, which is not RFC 8785; reported unverified, never verified",
            )
    return sig_res


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


# JCS bytes of the two-key {payload, signature} object, the bytes every anchor commits to.
def envelope_minus_anchors_jcs(env: dict) -> bytes:
    # An Audit Pack export carries more top-level members than the signer anchored;
    # only payload and signature were committed, so only those two are hashed.
    return canonical_json({k: env[k] for k in ("payload", "signature") if k in env})


# The same envelope with the signature string carried in the other base64 alphabet.
def _sig_alphabet_twin(env: dict) -> dict | None:
    # A local-key signer commits base64url while an export may re-encode the bytes in
    # standard base64; the twin lets the commitment be checked against both spellings.
    sig_obj = env.get("signature")
    if not isinstance(sig_obj, dict) or not isinstance(sig_obj.get("sig"), str):
        return None
    sig = sig_obj["sig"]
    try:
        if "-" in sig or "_" in sig:
            raw = base64.urlsafe_b64decode(sig + "=" * (-len(sig) % 4))
            other = base64.b64encode(raw).decode("ascii")
        else:
            raw = base64.b64decode(sig + "=" * (-len(sig) % 4))
            other = base64.urlsafe_b64encode(raw).decode("ascii")
    except (ValueError, binascii.Error):
        return None
    if sig.endswith("=") is False:
        other = other.rstrip("=")
    if other == sig:
        return None
    twin = dict(env)
    twin["signature"] = dict(sig_obj, sig=other)
    return twin


def _scan_shape(obj, max_depth: int | None = None) -> str | None:
    """Iteratively walk ``obj``; returns "non_finite", "too_deep", or None (ok).

    Explicit stack, no recursion: depth alone cannot crash this walk. With
    ``max_depth``, nesting past it stops and reports "too_deep" before the caller
    hands the structure to canonical_json's recursive stdlib encoder, which
    crashes past ~1000 levels on CPython <= 3.12.
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

#: Public verdict vocabulary (criteria 418/438). The per-axis PASS/FAIL/SKIPPED
#: tokens stay internal; the surface a caller reads speaks these three only.
VERDICT_VERIFIED = "verified"
#: Passing, but keyed under a holder salt so not third-party re-derivable.
#: Never collapsed into `verified` or `unverified`.
VERDICT_VERIFIED_KEYED = "verified_keyed"
VERDICT_UNVERIFIED = "unverified"

#: Unkeyed hash_algo values (absent defaults to sha256). Anything else counts
#: as keyed, so a near-miss spelling under-claims rather than over-claims.
_UNKEYED_HASH_ALGOS = frozenset({"sha256"})


def is_keyed_digest(payload: dict) -> bool:
    """True when the receipt's context digest is keyed (not re-derivable)."""
    if not isinstance(payload, dict):
        return False
    algo = payload.get("hash_algo")
    if algo is None:
        return False
    if not isinstance(algo, str):
        return True
    return algo.strip().lower() not in _UNKEYED_HASH_ALGOS

#: Failure classes carried by every unverified verdict (criterion 418); the two
#: are never collapsed - a proven binding failure is not an incomplete check.
FAILURE_INVALID = "invalid"
FAILURE_UNVERIFIABLE = "unverifiable"

#: Axes whose FAIL proves a cryptographic/policy binding failure (invalid).
_INVALID_FAIL_AXES = frozenset(
    {
        "signature",
        "anchors",
        "issuer_bind",
        "key_status",
        "nonce",
        "key_binding",
        "counterparty",
        "payload_digest",
    }
)


def _axis_failure_class(axis: str, result: str, note: str) -> str | None:
    """Map one axis outcome to its failure class (criterion 418).

    PASS carries none; SKIPPED means the recompute could not complete
    (unverifiable); a FAIL is invalid when a binding was proven broken and
    unverifiable when the receipt's own bytes stopped the recompute. Mirrors the
    oracle's ``core.axis_failure_class`` byte for byte, and a FAIL the table does
    not name reads unverifiable, never a proven binding failure.
    """
    if result == "PASS":
        return None
    if result == "SKIPPED":
        return FAILURE_UNVERIFIABLE
    if axis in _INVALID_FAIL_AXES:
        return FAILURE_INVALID
    if axis == "chain":
        if note.startswith("chain break:"):
            return FAILURE_INVALID
        return FAILURE_UNVERIFIABLE
    if axis == "skew":
        if note.startswith("unparseable issued_at"):
            return FAILURE_UNVERIFIABLE
        return FAILURE_INVALID
    if axis == "structure":
        return FAILURE_UNVERIFIABLE
    if axis == "expiry":
        if note.startswith("unreadable expires_at"):
            return FAILURE_UNVERIFIABLE
        return FAILURE_INVALID
    # issuer_key and anything unlisted: the recompute could not complete.
    return FAILURE_UNVERIFIABLE


    # Fold per-axis (name, result, note) rows into verdict + failure class.
def _fold_verdict(results, keyed: bool = False) -> tuple[str, str | None]:
    # Expiry reports on its own axis and never folds the verdict (criterion 426).
    failed = [(n, r, note) for n, r, note in results if r == "FAIL" and n != "expiry"]
    # A skipped chain (no predecessor supplied) is expected and does not block a
    # verified verdict; any other skip downgrades to unverified/unverifiable.
    blocking_skip = any(r == "SKIPPED" and n != "chain" for n, r, _ in results)
    if failed:
        classes = {_axis_failure_class(n, r, note) for n, r, note in failed}
        failure_class = FAILURE_INVALID if FAILURE_INVALID in classes else FAILURE_UNVERIFIABLE
        return VERDICT_UNVERIFIED, failure_class
    if blocking_skip:
        return VERDICT_UNVERIFIED, FAILURE_UNVERIFIABLE
    if keyed:
        return VERDICT_VERIFIED_KEYED, None
    return VERDICT_VERIFIED, None


class VerifierInputError(Exception):
    """A receipt or JWKS input was missing, empty, or not a JSON object.

    Raised so the CLI prints one readable line and exits nonzero rather than
    leaking a urllib/json traceback on bad input.
    """


class DuplicateMemberError(ValueError):
    """A receipt or JWKS repeated a JSON member name (criterion 419).

    The stdlib decoder is last-wins on duplicates, which would hash the bytes an
    attacker kept and drop the ones they replaced, so a duplicated name is a
    terminal parse failure raised before any hashing. Self-contained here (mirrors
    ``asqav/strict_json.py``) because this file ships standalone.
    """


#: Largest integer magnitude both SDKs canonicalise identically: 2**53 is exactly
#: representable and both emit the same digits for it, while 2**53 + 1 has no exact
#: double and JavaScript rounds it. One ABOVE JavaScript's Number.isSafeInteger bound,
#: deliberately: the upstream interop vector `number_2_to_53` pins 2**53 as canonical.
#: The bound also excludes every integer at or above 1e21, where JavaScript's toString
#: switches to exponential notation and Python's str does not.
MAX_CANONICAL_INTEGER = 2**53


class UnsafeIntegerError(ValueError):
    """An integer with no exact double; two readers would canonicalise it differently.

    A JavaScript reader rounds such a value while Python keeps it, so the same
    receipt produces two byte strings and one digest each. Refusing it at ingest is
    what keeps a verdict meaningful. Mirrors ``asqav/strict_json.py``.
    """


    # object_pairs_hook: reject a repeated member name at any depth (419).
def _reject_duplicate_members(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateMemberError(f"duplicate JSON member name: {key!r}")
        out[key] = value
    return out


    # parse_int: refuse an integer literal with no exact double.
def _reject_unsafe_integer(literal):
    value = int(literal)
    if not -MAX_CANONICAL_INTEGER <= value <= MAX_CANONICAL_INTEGER:
        raise UnsafeIntegerError(
            f"integer outside the canonical integer range +/-2**53: {literal}; serialise "
            "it as a JSON string or an integer-rational pair"
        )
    return value


def _parse_object(text: str, source: str) -> dict:
    """Parse ``text`` as a JSON object, or raise VerifierInputError.

    Rejects empty input, any non-object (array/string/number/null), and any
    duplicated member name at any depth (419), so a later ``.get`` never lands
    on a non-dict and last-wins bytes never reach the canonicaliser. Also rejects an
    integer outside +/-2**53, which no two readers canonicalise alike.
    """
    if not text or not text.strip():
        raise VerifierInputError(f"{source}: empty input, expected a JSON object")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_members,
            parse_int=_reject_unsafe_integer,
        )
    except (DuplicateMemberError, UnsafeIntegerError) as exc:
        raise VerifierInputError(f"{source}: {exc}") from exc
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

    Snapshot this before going air-gapped and pass the result to
    ``verify_receipt_offline(receipt, jwks)`` or ``run(envelope, jwks, ...)``.
    The endpoint is public and unauthenticated. Returns the parsed JWKS dict
    with a ``"keys"`` list.
    """
    return _get_json(url, timeout=timeout)


    # Decode standard or url-safe base64, padding-tolerant.
def _b64decode(value: str) -> bytes:
    s = value.replace("-", "+").replace("_", "/")
    s += "=" * ((-len(s)) % 4)
    return base64.b64decode(s, validate=False)


# RFC 7638 JWK Thumbprint over an ML-DSA (AKP) public key, mirroring the cloud's
# key_thumbprint. Inlined so this stays the single-file offline verifier.

#: draft-ietf-cose-dilithium key type for ML-DSA key pairs
AKP_KTY = "AKP"

#: The only well-formed key_thumbprint shape; anything else is refused, not compared
_THUMBPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

#: FIPS 204 public-key widths in bytes, fixed by the standard. A KMS-backed agent
#: publishes a PEM where a raw key would sit, so width is what separates the two.
_ML_DSA_PUBLIC_KEY_BYTES = {"ML-DSA-44": 1312, "ML-DSA-65": 1952, "ML-DSA-87": 2592}


    # base64url with padding stripped - the encoding RFC 7638 `pub` requires.
def _b64url_nopad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


    # True when these bytes are a raw ML-DSA public key of the width `alg` fixes.
def is_akp_public_key(*, alg: str, public_key: bytes | None) -> bool:
    if not public_key:
        return False
    expected = _ML_DSA_PUBLIC_KEY_BYTES.get(alg)
    return expected is not None and len(public_key) == expected


    # True for a `sha256:<64 lowercase hex>` string, the only comparable form. fullmatch,
    # not match: Python's `$` also matches before a trailing newline, where the JS test does not.
def is_well_formed(value) -> bool:
    return isinstance(value, str) and _THUMBPRINT_RE.fullmatch(value) is not None


def akp_jwk(*, alg: str, public_key: bytes) -> dict:
    """Build the required-members-only AKP JWK the thumbprint is taken over.

    `pub` is base64url WITHOUT padding, which is the one encoding trap here: the
    published directory carries the same bytes as standard base64 under
    `public_key`, and thumbprinting that alphabet yields a digest no third-party
    verifier reproduces.
    """
    if not alg:
        raise ValueError("alg is required to build an AKP JWK")
    if not public_key:
        raise ValueError("public_key is required to build an AKP JWK")
    return {"alg": alg, "kty": AKP_KTY, "pub": _b64url_nopad(public_key)}


def jwk_thumbprint(jwk: dict) -> str:
    """Return `sha256:<hex>` for a JWK already reduced to its required members.

    Sorting here is what RFC 7638 section 3 calls for, so the caller cannot change
    the digest by handing the members in a different order.
    """
    return f"sha256:{hashlib.sha256(canonical_json(jwk)).hexdigest()}"


    # The profile's key_thumbprint value for one ML-DSA public key.
def thumbprint_for_key(*, alg: str, public_key: bytes) -> str:
    return jwk_thumbprint(akp_jwk(alg=alg, public_key=public_key))


# Anchor values only, and strict on purpose: anchors sit outside the signed
# bytes, so a forged value is attacker steerable
_ANCHOR_B64_RE = re.compile(r"[A-Za-z0-9+/]+={0,2}")


    # Strict base64 -> decoded bytes, or None on any junk (None never raises).
def _anchor_bytes(v: object):
    if not isinstance(v, str) or not v:
        return None
    s = v.replace("-", "+").replace("_", "/")
    s += "=" * ((-len(s)) % 4)
    # The alphabet and padding decision lives here, not in b64decode, because
    # b64decode(validate=True) accepts excess padding on 3.11 and raises on 3.12
    if not _ANCHOR_B64_RE.fullmatch(s):
        return None
    # fullmatch leaves only alphabet characters and a length that is a multiple
    # of 4 with at most two pads, so a strict decode cannot raise here
    raw = base64.b64decode(s, validate=True)
    return raw if raw else None


def _safe_b64(v: object) -> bool:
    """True only when ``v`` is real base64 carrying at least one byte.

    Kept separate from ``_b64decode``, which stays lenient for keys and
    signatures. Anchors are unsigned, so this half refuses junk instead.
    """
    return _anchor_bytes(v) is not None


# --- Offline anchor verification: presence is not proof, each entry reports
# verified / invalid / unverifiable, and trust material is caller-pinned. ------


class _AnchorParseError(ValueError):
    """A token blob did not parse; the anchor reports unverifiable, never a crash."""


    # Read one DER TLV at ``off``; return (tag, content bytes, next offset).
def _der_read(buf: bytes, off: int) -> tuple[int, bytes, int]:
    if off + 2 > len(buf):
        raise _AnchorParseError("truncated DER header")
    tag = buf[off]
    if tag & 0x1F == 0x1F:
        raise _AnchorParseError("multi-byte DER tags unsupported")
    first = buf[off + 1]
    off += 2
    if first & 0x80:
        n = first & 0x7F
        if n == 0 or n > 4 or off + n > len(buf):
            raise _AnchorParseError("bad DER length")
        length = int.from_bytes(buf[off : off + n], "big")
        off += n
    else:
        length = first
    if length > len(buf) - off:
        raise _AnchorParseError("DER content overruns buffer")
    return tag, buf[off : off + length], off + length


    # DER length bytes for a content of ``n`` bytes (definite form).
def _der_len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    body = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(body)]) + body


    # Iterate the TLVs concatenated inside a constructed element's content.
def _der_children(content: bytes) -> list[tuple[int, bytes, int, int]]:
    out = []
    off = 0
    while off < len(content):
        tag, value, nxt = _der_read(content, off)
        out.append((tag, value, off, nxt))
        off = nxt
    return out


#: Real arcs are tiny. Unbounded, a relay-supplied arc builds an int whose
#: str() raises the interpreter's digit-limit ValueError, crashing the verifier.
_MAX_OID_ARC_BITS = 512


def _der_oid_str(content: bytes) -> str:
    if not content:
        raise _AnchorParseError("empty OID")
    first = content[0]
    arcs = [str(min(first // 40, 2)), str(first - 40 * min(first // 40, 2))]
    val = 0
    for byte in content[1:]:
        val = (val << 7) | (byte & 0x7F)
        if val.bit_length() > _MAX_OID_ARC_BITS:
            raise _AnchorParseError("OID arc exceeds the supported width")
        if not byte & 0x80:
            arcs.append(str(val))
            val = 0
    if content[-1] & 0x80 and len(content) > 1:
        raise _AnchorParseError("truncated OID arc")
    return ".".join(arcs)


    # Decode an AlgorithmIdentifier's OID out of its SEQUENCE content.
def _algid_oid(content: bytes) -> str:
    children = _der_children(content)
    if not children or children[0][0] != 0x06:
        raise _AnchorParseError("AlgorithmIdentifier without OID")
    return _der_oid_str(children[0][1])


_OID_SIGNED_DATA = "1.2.840.113549.1.7.2"
_OID_MESSAGE_DIGEST = "1.2.840.113549.1.9.4"
#: RFC 5652 s11.1 contentType and the eContentType it must equal. Binding both
#: blocks reuse of a TSA signature over content that parses as a TSTInfo.
_OID_CONTENT_TYPE = "1.2.840.113549.1.9.3"
_OID_CT_TST_INFO = "1.2.840.113549.1.9.16.1.4"
#: digestAlgorithm OIDs this verifier hashes with, OID -> hashlib name.
_DIGEST_OIDS = {
    "2.16.840.1.101.3.4.2.1": "sha256",
    "2.16.840.1.101.3.4.2.2": "sha384",
    "2.16.840.1.101.3.4.2.3": "sha512",
}
#: TSA signature algorithms verified via dilithium-py (raw pinned keys).
_ML_DSA_SIG_OIDS = {
    "2.16.840.1.101.3.4.3.17": "ML_DSA_44",
    "2.16.840.1.101.3.4.3.18": "ML_DSA_65",
    "2.16.840.1.101.3.4.3.19": "ML_DSA_87",
}
_OID_ED25519 = "1.3.101.112"
#: Bare rsaEncryption (RFC 5652): the SignerInfo digestAlgorithm names the hash.
_OID_RSA_ENCRYPTION = "1.2.840.113549.1.1.1"
#: RSA/ECDSA signature OIDs verified via cryptography (pinned X.509 certs).
_RSA_SIG_OIDS = {
    "1.2.840.113549.1.1.11": "sha256",
    "1.2.840.113549.1.1.12": "sha384",
    "1.2.840.113549.1.1.13": "sha512",
}
_ECDSA_SIG_OIDS = {
    "1.2.840.10045.4.3.2": "sha256",
    "1.2.840.10045.4.3.3": "sha384",
    "1.2.840.10045.4.3.4": "sha512",
}


def _parse_generalized_time(content: bytes):
    """GeneralizedTime -> aware datetime; RFC 3161 pins the Z (UTC) form."""
    m = re.fullmatch(rb"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?:\.\d+)?Z", content)
    if m is None:
        raise _AnchorParseError("unparseable genTime")
    try:
        return datetime(
            *(int(g) for g in m.groups()), tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise _AnchorParseError(f"genTime out of range: {exc}") from exc


    # Parse a retained TimeStampResp into the fields the checks below read.
def _parse_time_stamp_resp(der: bytes) -> dict:
    tag, content, nxt = _der_read(der, 0)
    if tag != 0x30 or nxt != len(der):
        raise _AnchorParseError("TimeStampResp is not one DER SEQUENCE")
    resp = _der_children(content)
    if not resp or resp[0][0] != 0x30:
        raise _AnchorParseError("missing PKIStatusInfo")
    status_fields = _der_children(resp[0][1])
    if not status_fields or status_fields[0][0] != 0x02:
        raise _AnchorParseError("PKIStatusInfo without status integer")
    status = int.from_bytes(status_fields[0][1], "big")
    if status not in (0, 1):  # granted / grantedWithMods; anything else is a refusal
        return {"status": status}
    if len(resp) < 2 or resp[1][0] != 0x30:
        raise _AnchorParseError("granted status but no timeStampToken")
    ci = _der_children(resp[1][1])
    if len(ci) != 2 or ci[0][0] != 0x06 or _der_oid_str(ci[0][1]) != _OID_SIGNED_DATA:
        raise _AnchorParseError("timeStampToken is not a CMS SignedData")
    if ci[1][0] != 0xA0:
        raise _AnchorParseError("SignedData content missing")
    sd_children = _der_children(ci[1][1])
    if not sd_children or sd_children[0][0] != 0x30:
        raise _AnchorParseError("SignedData body missing")
    sd = _der_children(sd_children[0][1])
    if len(sd) < 4:
        raise _AnchorParseError("SignedData too short")
    encap = _der_children(sd[2][1])
    if len(encap) != 2 or encap[1][0] != 0xA0:
        raise _AnchorParseError("encapContentInfo without eContent")
    if encap[0][0] != 0x06 or _der_oid_str(encap[0][1]) != _OID_CT_TST_INFO:
        # Not a time-stamp token: the signature covers some other content type.
        raise _AnchorParseError("eContentType is not id-ct-TSTInfo")
    etag, tst_bytes, _ = _der_read(encap[1][1], 0)
    if etag != 0x04:
        raise _AnchorParseError("eContent is not an OCTET STRING")
    certs = []
    signer_infos = None
    for stag, svalue, _sstart, _send in sd[3:]:
        if stag == 0xA0:
            certs = [svalue[cstart:send] for _t, _v, cstart, send in _der_children(svalue)]
        elif stag == 0x31:
            signer_infos = _der_children(svalue)
    if not signer_infos:
        raise _AnchorParseError("SignedData without signerInfos")
    si = _der_children(signer_infos[0][1])
    if len(si) < 5:
        raise _AnchorParseError("SignerInfo too short")
    # version, sid, digestAlgorithm, [signedAttrs], signatureAlgorithm, signature
    rest = si[3:]
    signed_attrs = None
    if rest[0][0] == 0xA0:
        signed_attrs = rest[0][1]
        rest = rest[1:]
    if len(rest) < 2 or rest[0][0] != 0x30 or rest[1][0] != 0x04:
        raise _AnchorParseError("SignerInfo without signatureAlgorithm/signature")
    return {
        "status": status,
        "tst": tst_bytes,
        "certs": certs,
        "digest_alg": _algid_oid(si[2][1]) if si[2][0] == 0x30 else None,
        "signed_attrs": signed_attrs,
        "sig_alg": _algid_oid(rest[0][1]),
        "signature": rest[1][1],
    }


    # Parse TSTInfo; return (imprint digest bytes, imprint hash OID, genTime).
def _parse_tst_info(tst: bytes):
    tag, content, nxt = _der_read(tst, 0)
    if tag != 0x30 or nxt != len(tst):
        raise _AnchorParseError("TSTInfo is not one DER SEQUENCE")
    fields = _der_children(content)
    if len(fields) < 5 or fields[2][0] != 0x30:
        raise _AnchorParseError("TSTInfo without messageImprint")
    imprint = _der_children(fields[2][1])
    if len(imprint) != 2 or imprint[0][0] != 0x30 or imprint[1][0] != 0x04:
        raise _AnchorParseError("malformed messageImprint")
    if fields[4][0] != 0x18:
        raise _AnchorParseError("TSTInfo without genTime")
    return imprint[1][1], _algid_oid(imprint[0][1]), _parse_generalized_time(fields[4][1])


    # The bytes a CMS signature covers: signedAttrs re-tagged as SET OF, else eContent.
def _cms_signed_bytes(info: dict) -> bytes:
    attrs = info["signed_attrs"]
    if attrs is None:
        return info["tst"]
    return b"\x31" + _der_len(len(attrs)) + attrs


    # A signedAttrs messageDigest attribute must commit the eContent bytes.
def _cms_message_digest_ok(info: dict) -> bool:
    attrs = info["signed_attrs"]
    if attrs is None:
        return True
    digest_name = _DIGEST_OIDS.get(info.get("digest_alg") or "")
    if digest_name is None:
        return False
    want = hashlib.new(digest_name, info["tst"]).digest()
    digest_ok = False
    content_type_ok = False
    for atag, avalue, _s, _e in _der_children(attrs):
        if atag != 0x30:
            continue
        attr = _der_children(avalue)
        if len(attr) != 2 or attr[0][0] != 0x06:
            continue
        attr_oid = _der_oid_str(attr[0][1])
        if attr_oid == _OID_MESSAGE_DIGEST:
            vals = _der_children(attr[1][1])
            digest_ok = len(vals) == 1 and vals[0][0] == 0x04 and vals[0][1] == want
        elif attr_oid == _OID_CONTENT_TYPE:
            vals = _der_children(attr[1][1])
            content_type_ok = (
                len(vals) == 1
                and vals[0][0] == 0x06
                and _der_oid_str(vals[0][1]) == _OID_CT_TST_INFO
            )
    # contentType is mandatory alongside signedAttrs (RFC 5652 s11.1).
    return digest_ok and content_type_ok


def _tsa_key_candidates(trusted_tsa_keys):
    """Split pinned TSA material into (raw key bytes, cryptography public keys).

    A PEM/DER X.509 certificate or public key needs ``cryptography`` to decode;
    without it those entries are unusable and the caller's axis reports
    unverifiable rather than trusting anything. Raw bytes pass through for the
    ML-DSA / Ed25519 paths.
    """
    raw, pkeys = [], []
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives.serialization import (
            load_der_public_key,
            load_pem_public_key,
        )
    except ImportError:
        x509 = None
    for entry in trusted_tsa_keys or []:
        if not isinstance(entry, (bytes, bytearray)) or not entry:
            continue
        blob = bytes(entry)
        if x509 is not None:
            try:
                if blob.startswith(b"-----BEGIN"):
                    try:
                        pkeys.append(x509.load_pem_x509_certificate(blob).public_key())
                    except ValueError:
                        pkeys.append(load_pem_public_key(blob))
                    continue
                pkeys.append(x509.load_der_x509_certificate(blob).public_key())
                continue
            except ValueError:
                try:
                    pkeys.append(load_der_public_key(blob))
                    continue
                except ValueError:
                    pass
        raw.append(blob)
    return raw, pkeys


def _verify_tsa_signature(
    sig_alg: str,
    signed: bytes,
    signature: bytes,
    trusted_tsa_keys,
    digest_alg: str | None = None,
) -> str:
    """Verify the TSA signature against pinned key material only.

    Returns "verified", "invalid" (a usable key was present and no signature
    verified), or "unverifiable" (no usable key material or optional dep).
    ``digest_alg`` is consulted only under the bare rsaEncryption OID, which
    names no hash of its own; an unknown digest there reads unverifiable.
    """
    raw_keys, pkeys = _tsa_key_candidates(trusted_tsa_keys)
    usable = False
    if sig_alg in _ML_DSA_SIG_OIDS:
        try:
            import dilithium_py.ml_dsa as _ml
        except ImportError:
            return "unverifiable"
        mod = getattr(_ml, _ML_DSA_SIG_OIDS[sig_alg])
        for pk in raw_keys:
            usable = True
            try:
                if mod.verify(pk, signed, signature):
                    return "verified"
            except Exception:  # wrong-size or malformed candidate key; try the next
                continue
    elif sig_alg == _OID_ED25519:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError:
            # cryptography is optional; without it this branch reports unverifiable,
            # as the RSA/ECDSA one does, rather than raising out of the anchors axis.
            return "unverifiable"

        for pk in raw_keys:
            if len(pk) != 32:
                continue
            usable = True
            try:
                Ed25519PublicKey.from_public_bytes(pk).verify(signature, signed)
                return "verified"
            except Exception:
                continue
        for pk in pkeys:
            usable = True
            try:
                pk.verify(signature, signed)
                return "verified"
            except Exception:
                continue
    elif (
        sig_alg in _RSA_SIG_OIDS
        or sig_alg in _ECDSA_SIG_OIDS
        or sig_alg == _OID_RSA_ENCRYPTION
    ):
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import ec, padding
        except ImportError:
            return "unverifiable"
        if sig_alg == _OID_RSA_ENCRYPTION:
            # Bare rsaEncryption: the SignerInfo digestAlgorithm names the hash.
            hash_name = _DIGEST_OIDS.get(digest_alg or "")
            if hash_name is None:
                return "unverifiable"
        else:
            table = _RSA_SIG_OIDS if sig_alg in _RSA_SIG_OIDS else _ECDSA_SIG_OIDS
            hash_name = table[sig_alg]
        hash_alg = getattr(hashes, hash_name.upper())()
        for pk in pkeys:
            usable = True
            try:
                if sig_alg in _ECDSA_SIG_OIDS:
                    pk.verify(signature, signed, ec.ECDSA(hash_alg))
                else:
                    pk.verify(signature, signed, padding.PKCS1v15(), hash_alg)
                return "verified"
            except Exception:
                continue
    else:
        # An algorithm family this verifier cannot evaluate at all.
        return "unverifiable"
    return "invalid" if usable else "unverifiable"


def _check_rfc3161_anchor(
    blob: bytes, bound: bytes, trusted_tsa_keys, env_jcs: bytes | None = None
) -> tuple[str, str, object]:
    """One RFC 3161 anchor -> (outcome, detail, genTime when verified).

    ``env_jcs`` is the JCS of the envelope minus anchors, needed only to re-derive
    the committed digest under a non-sha256 messageImprint algorithm; ``bound`` is
    its sha256 and stays the default. The imprint comparison is stdlib-only; the
    TSA signature needs pinned key material (dilithium-py for ML-DSA, cryptography
    for RSA/ECDSA/Ed25519). A full PKI chain walk is out of scope offline: trust
    comes from the caller's allowlist, never from the unsigned envelope.
    """
    # Every DER walk stays inside this guard: an escaping exception exits 1
    # (invalid), claiming a binding failure for input never evaluated.
    try:
        info = _parse_time_stamp_resp(blob)
        if info["status"] not in (0, 1):
            return "invalid", f"TimeStampResp carries TSA status {info['status']}, not granted", None
        imprint, imprint_alg, gen_time = _parse_tst_info(info["tst"])
        digest_name = _DIGEST_OIDS.get(imprint_alg)
        if digest_name is None:
            return (
                "unverifiable",
                f"offline RFC3161 check did not complete; unknown messageImprint OID {imprint_alg}",
                None,
            )
        # A sha384/sha512 imprint we never computed is not a proven mismatch.
        if digest_name != "sha256":
            if env_jcs is None:
                return (
                    "unverifiable",
                    f"offline RFC3161 check did not complete; {digest_name} imprint not recomputable",
                    None,
                )
            bound = hashlib.new(digest_name, env_jcs).digest()
        if imprint != bound:
            return "invalid", "RFC3161 imprint commits a different digest than this envelope", None
        if not _cms_message_digest_ok(info):
            return "invalid", "signedAttrs messageDigest does not commit the TSTInfo bytes", None
        sig_state = _verify_tsa_signature(
            info["sig_alg"],
            _cms_signed_bytes(info),
            info["signature"],
            trusted_tsa_keys,
            digest_alg=info.get("digest_alg"),
        )
    except _AnchorParseError:
        # Shared note with the TypeScript shim, which runs no DER parse; the
        # parity corpora pin this exact wording for shape-valid junk values.
        return "unverifiable", "offline RFC3161 check did not complete", None
    except (ValueError, IndexError, TypeError, MemoryError, RecursionError):
        # Anything else the DER walk raises: still unverifiable, never a pass.
        return "unverifiable", "offline RFC3161 check did not complete", None
    if sig_state == "verified":
        return (
            "verified",
            f"RFC3161 imprint commits this envelope; TSA signature verifies at {gen_time.isoformat()}",
            gen_time,
        )
    if sig_state == "invalid":
        return "invalid", "TSA signature does not verify against any trusted TSA key", None
    return (
        "unverifiable",
        "offline RFC3161 check did not complete; imprint matches but no trusted "
        "TSA key material verified the token signature",
        None,
    )


#: Detached .ots header (python-opentimestamps DetachedTimestampFile.HEADER_MAGIC).
_OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
#: BitcoinBlockHeaderAttestation.TAG (opentimestamps.core.notary).
_OTS_BITCOIN_TAG = bytes.fromhex("0588960d73d71901")
#: CryptOp tags this evaluator can hash with (RFC4880 numbering, per ots op.py).
_OTS_HASH_OPS = {0x02: "sha1", 0x03: "ripemd160", 0x08: "sha256"}
_OTS_MAX_BLOB = 1 << 20
_OTS_MAX_MSG = 4096
_OTS_MAX_DEPTH = 128


class _OtsState:
    def __init__(self) -> None:
        self.attestations: list[tuple[int, bytes]] = []
        self.unknown = False


#: Block heights fit in five groups. Unbounded, ~1 MiB of continuation bytes
#: builds a multi-megabit int one shift at a time: 326s of CPU per anchor.
_OTS_MAX_VARUINT_GROUPS = 10


def _ots_varuint(buf: bytes, off: int) -> tuple[int, int]:
    val = 0
    groups = 0
    while True:
        if off >= len(buf):
            raise _AnchorParseError("truncated ots varuint")
        groups += 1
        if groups > _OTS_MAX_VARUINT_GROUPS:
            raise _AnchorParseError("ots varuint exceeds the supported width")
        byte = buf[off]
        off += 1
        val = (val << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return val, off


def _ots_varbytes(buf: bytes, off: int) -> tuple[bytes, int]:
    n, off = _ots_varuint(buf, off)
    if n > len(buf) - off:
        raise _AnchorParseError("truncated ots varbytes")
    return buf[off : off + n], off + n


def _ots_item(tag: int, buf: bytes, off: int, msg: bytes, depth: int, state: _OtsState) -> int:
    if tag == 0x00:
        atag = buf[off : off + 8]
        if len(atag) != 8:
            raise _AnchorParseError("truncated ots attestation tag")
        off += 8
        if atag == _OTS_BITCOIN_TAG:
            height, off = _ots_varuint(buf, off)
            state.attestations.append((height, msg))
        else:
            _payload, off = _ots_varbytes(buf, off)
            state.unknown = True
        return off
    if tag in (0xF0, 0xF1):
        arg, off = _ots_varbytes(buf, off)
        new_msg = msg + arg if tag == 0xF0 else arg + msg
        if len(new_msg) > _OTS_MAX_MSG:
            raise _AnchorParseError("ots message exceeds the 4096-byte op limit")
        return _ots_node(buf, off, new_msg, depth + 1, state)
    if tag in _OTS_HASH_OPS:
        try:
            new_msg = hashlib.new(_OTS_HASH_OPS[tag], msg).digest()
        except (ValueError, TypeError) as exc:
            raise _AnchorParseError(f"ots hash op unavailable here: {exc}") from exc
        return _ots_node(buf, off, new_msg, depth + 1, state)
    raise _AnchorParseError(f"unknown ots op tag 0x{tag:02x}")


def _ots_node(buf: bytes, off: int, msg: bytes, depth: int, state: _OtsState) -> int:
    if depth > _OTS_MAX_DEPTH:
        raise _AnchorParseError("ots proof nesting exceeds the supported depth")
    if off >= len(buf):
        raise _AnchorParseError("truncated ots timestamp")
    tag = buf[off]
    off += 1
    while tag == 0xFF:
        if off >= len(buf):
            raise _AnchorParseError("truncated ots fork")
        off = _ots_item(buf[off], buf, off + 1, msg, depth, state)
        if off >= len(buf):
            raise _AnchorParseError("truncated ots fork")
        tag = buf[off]
        off += 1
    return _ots_item(tag, buf, off, msg, depth, state)


def _check_ots_anchor(blob: bytes, bound: bytes, bitcoin_headers) -> tuple[str, str, object]:
    """One OpenTimestamps anchor -> (outcome, detail, block time when verified).

    Offline-checkable portion: the proof's initial commitment must equal this
    envelope's digest, and the op chain evaluates to a merkle root. Placing that
    root in a real block needs a caller-supplied header source; without one the
    block portion reports unverifiable, never a PASS.
    """
    try:
        if len(blob) > _OTS_MAX_BLOB or not blob.startswith(_OTS_MAGIC):
            raise _AnchorParseError("not a detached .ots proof")
        off = len(_OTS_MAGIC)
        if blob[off] != 1:
            raise _AnchorParseError("unsupported .ots major version")
        off += 1
        hash_op = blob[off]
        off += 1
        if hash_op != 0x08:  # the envelope digest is sha256; other ops cannot commit it
            raise _AnchorParseError("proof does not commit a sha256 digest")
        committed, blob_rest = blob[off : off + 32], off + 32
        if len(committed) != 32:
            raise _AnchorParseError("truncated commitment digest")
        if committed != bound:
            return "invalid", "OpenTimestamps proof commits a different digest than this envelope", None
        state = _OtsState()
        if _ots_node(blob, blob_rest, committed, 0, state) != len(blob):
            raise _AnchorParseError("trailing bytes after the ots timestamp")
    except (_AnchorParseError, IndexError):
        return "unverifiable", "offline OpenTimestamps check did not complete", None
    if not state.attestations:
        return "unverifiable", "offline OpenTimestamps check did not complete; no bitcoin attestation in proof", None
    headers = {}
    if isinstance(bitcoin_headers, dict):
        for k, v in bitcoin_headers.items():
            try:
                headers[int(k)] = v
            except (TypeError, ValueError):
                continue
    saw_mismatch = None
    for height, root in state.attestations:
        header = headers.get(height)
        if not isinstance(header, dict) or not isinstance(header.get("merkle_root"), str):
            continue
        try:
            # The attestation message is the block's merkle root in internal
            # (little-endian) order; the header source carries the display hex.
            want = bytes.fromhex(header["merkle_root"])[::-1]
        except ValueError:
            continue
        if len(want) != 32:
            continue
        if root != want:
            saw_mismatch = height
            continue
        when = _parse_stamp(header.get("time")) if header.get("time") else None
        return "verified", f"OpenTimestamps proof lands in bitcoin block {height}", when
    if saw_mismatch is not None:
        return "invalid", f"merkle path does not land in bitcoin block {saw_mismatch}", None
    return (
        "unverifiable",
        "offline OpenTimestamps check did not complete; commitment matches but no "
        "supplied bitcoin header confirms the block",
        None,
    )


#: (result, note, trusted_times) for the anchors axis. ``trusted_times`` holds times of
#: anchors that CRYPTOGRAPHICALLY verified, the only ones weighable against revoked_at.
AnchorEvaluation = namedtuple("AnchorEvaluation", ("result", "note", "trusted_times"))


def evaluate_anchors(envelope: dict, *, trusted_tsa_keys=None, bitcoin_headers=None) -> AnchorEvaluation:
    """Cryptographically evaluate every anchor against the envelope digest.

    Shape rules are unchanged (malformed entries FAIL, absent/empty SKIPs), but a
    shape-valid entry no longer PASSes on presence: the axis PASSes only when every
    entry cryptographically verifies, FAILs when any entry's check runs and fails,
    and reports SKIPPED (unverifiable) when a check cannot complete offline. Anchors
    declared ``status: pending``/``failed`` never count as trusted anchors.
    """
    anchors = envelope.get("anchors")
    # Absent/null is a legitimate no-anchors receipt (SKIPPED); a present
    # non-list value ({} / "" / 0) is malformed and FAILs, never laundered to [].
    if anchors is None:
        return AnchorEvaluation("SKIPPED", "no anchors on this receipt", [])
    if not isinstance(anchors, list):
        return AnchorEvaluation("FAIL", f"anchors field is not a list (got {type(anchors).__name__})", [])
    if not anchors:
        return AnchorEvaluation("SKIPPED", "no anchors on this receipt", [])
    try:
        _env_jcs = envelope_minus_anchors_jcs(envelope)
        bound = hashlib.sha256(_env_jcs).digest()
        _twin = _sig_alphabet_twin(envelope)
        _twin_jcs = envelope_minus_anchors_jcs(_twin) if _twin is not None else None
    except RecursionError:
        # Defense in depth, same rationale as check_chain's guard above.
        return AnchorEvaluation(
            "FAIL", "envelope too deeply nested to canonicalise for anchor binding", []
        )
    lines = [f"anchors bind envelope digest sha256:{bound.hex()[:16]}.."]
    _twin_bound = hashlib.sha256(_twin_jcs).digest() if _twin_jcs is not None else None
    _twin_used = False
    saw_invalid = False
    saw_unverifiable = False
    trusted_times = []
    for a in anchors:
        if not isinstance(a, dict):
            saw_invalid = True
            lines.append(f"    - malformed anchor entry (got {type(a).__name__}, expected an object)")
            continue
        atype = a.get("type", "?")
        blob = _anchor_bytes(a.get("value"))
        if blob is None:
            saw_invalid = True
            lines.append(f"    - {atype}: value MISSING or malformed")
            continue
        status = a.get("status")
        if status in ("pending", "failed"):
            saw_unverifiable = True
            lines.append(
                f"    - {atype}: value present, base64-ok; status {status}, not an anchored proof"
            )
            continue
        if atype == "rfc3161":
            outcome, detail, when = _check_rfc3161_anchor(
                blob, bound, trusted_tsa_keys, _env_jcs
            )
            if outcome == "invalid" and "different digest" in detail and _twin_bound is not None:
                outcome, detail, when = _check_rfc3161_anchor(
                    blob, _twin_bound, trusted_tsa_keys, _twin_jcs
                )
                _twin_used = _twin_used or outcome != "invalid"
        elif atype == "opentimestamps":
            outcome, detail, when = _check_ots_anchor(blob, bound, bitcoin_headers)
            if outcome == "invalid" and "different digest" in detail and _twin_bound is not None:
                outcome, detail, when = _check_ots_anchor(blob, _twin_bound, bitcoin_headers)
                _twin_used = _twin_used or outcome != "invalid"
        else:
            outcome, detail, when = (
                "unverifiable",
                "no offline verifier for this anchor type",
                None,
            )
        if outcome == "verified":
            lines.append(f"    - {atype}: verified ({detail})")
            if when is not None:
                trusted_times.append(when)
        elif outcome == "invalid":
            saw_invalid = True
            lines.append(f"    - {atype}: invalid ({detail})")
        else:
            saw_unverifiable = True
            lines.append(f"    - {atype}: value present, base64-ok; unverifiable ({detail})")
    if _twin_used:
        lines.append(
            "    - signature string re-encoded to the alphabet the signer committed "
            "(the export carries the other base64 alphabet)"
        )
    if saw_invalid:
        return AnchorEvaluation("FAIL", "; ".join(lines), trusted_times)
    if saw_unverifiable:
        return AnchorEvaluation("SKIPPED", "; ".join(lines), trusted_times)
    return AnchorEvaluation("PASS", "; ".join(lines), trusted_times)


def check_anchors(envelope: dict, *, trusted_tsa_keys=None, bitcoin_headers=None):
    """(result, note) for the anchors axis; see evaluate_anchors."""
    ev = evaluate_anchors(
        envelope, trusted_tsa_keys=trusted_tsa_keys, bitcoin_headers=bitcoin_headers
    )
    return ev.result, ev.note


def normalise_envelope(raw: dict) -> dict:
    """Project any receipt shape onto the canonical 3-key envelope.

    An Audit Pack export carries ``{payload, signature, anchors}`` plus export-side
    members (ids, base64 copies of the stored bytes); only the three canonical members
    are kept, so no export-side member reaches an axis. The hosted ``/verify/{id}`` JSON
    nests the signed dict under ``payload`` and exposes the signature object as
    ``signature_envelope`` (plus a possibly flat-string ``signature``); rebuild from
    those so ``run()`` sees one shape.
    """
    # A top-level signature object plus a payload dict: keep exactly the three members.
    sig = raw.get("signature")
    if isinstance(sig, dict) and isinstance(raw.get("payload"), dict):
        return {"payload": raw["payload"], "signature": sig, "anchors": raw.get("anchors")}

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


def _match_key(jwks: dict, kid: str):
    """Return the first usable jwks entry whose issuer id or key id is kid.

    One matcher, so key material and the key's published fields always come from
    the same entry. A malformed jwks is a miss (returns None), never a crash.
    """
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    if not isinstance(keys, list):
        return None
    for k in keys:
        if not isinstance(k, dict):
            continue
        if kid and kid in (k.get("issuer_id"), k.get("kid")):
            if not isinstance(k.get("public_key"), str):
                continue
            return k
    return None


def _match_key_by_id(jwks: dict, kid: str):
    """Return the first usable jwks entry whose crypto key id is exactly kid.

    A key id names one key, so it decides on its own. The looser issuer-id match in
    ``_match_key`` answers for every key an org publishes, which is a set, not a key.
    """
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    if not isinstance(keys, list):
        return None
    for k in keys:
        if not isinstance(k, dict):
            continue
        if kid and kid == k.get("kid"):
            if not isinstance(k.get("public_key"), str):
                continue
            return k
    return None


def _key_bound_to_claimant(k: dict, issuer_id, org_id=None):
    """Return True when a key's published issuer or org equals the claimed one.

    agent_id is attacker-controlled, so this claim bind, not the agent match alone,
    decides which sibling a receipt verifies against. A hash-mode receipt signs the
    raw org_id rather than issuer_id, so the org binding answers for that shape and
    names exactly one signer.
    """
    issuer_bound = issuer_id and issuer_id == k.get("issuer_id")
    org_bound = org_id and org_id == k.get("org_id")
    return bool(issuer_bound or org_bound)


#: Agent-bound keys a failing signature is tried against, so a rotation stays bounded work.
_AGENT_BIND_CANDIDATE_CAP = 8


def _match_keys_by_agent(jwks: dict, agent_id, issuer_id, org_id=None) -> list:
    """Return every usable jwks entry for an agent key bound to a claimed issuer.

    The single agent-side matcher, so the entry a signature verifies against is the same
    entry every published field is read from. A rotation leaves an agent with more than
    one published key, so this answers with the whole candidate set, not a guess.
    """
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    if not isinstance(keys, list):
        return []
    out = []
    for k in keys:
        if not isinstance(k, dict):
            continue
        if not (agent_id and agent_id == k.get("agent_id")):
            continue
        if not _key_bound_to_claimant(k, issuer_id, org_id):
            continue
        if not isinstance(k.get("public_key"), str):
            continue
        out.append(k)
    return out


def _match_key_by_agent(jwks: dict, agent_id, issuer_id, org_id=None):
    # The first agent-bound key, for callers that resolve before they have a signature.
    candidates = _match_keys_by_agent(jwks, agent_id, issuer_id, org_id)
    return candidates[0] if candidates else None


def _select_agent_bound_key(jwks: dict, agent_id, issuer_id, org_id, msg, sig, alg_hint):
    """Return (entry, signature result, exhausted) for the agent-bound key that signed msg.

    Every candidate sharing the agent bind is tried, bounded by the candidate cap, and the
    one whose signature verifies is kept; when none does, the first is returned so the
    report can say the whole set was tried. Revocation is never weighed here.
    """
    candidates = _match_keys_by_agent(jwks, agent_id, issuer_id, org_id)
    if not candidates:
        return None, None, False
    last = None
    for k in candidates[:_AGENT_BIND_CANDIDATE_CAP]:
        res = verify_signature(_b64decode(k["public_key"]), msg, sig, k.get("alg") or alg_hint)
        if res[0] == "PASS":
            return k, res, False
        last = res
    return candidates[0], last, True


def _match_key_by_thumbprint(jwks: dict, thumbprint, agent_id=None, kid=None):
    """Return the one usable jwks entry publishing exactly this key_thumbprint.

    The thumbprint sits inside the signed bytes and names one key in one step, so it
    outranks every unsigned or set-valued identifier. Sibling rows can publish one key
    under different ids and statuses, so a tie is broken by the signed agent_id and then
    by the envelope kid; list position decides nothing on its own.
    """
    if not isinstance(thumbprint, str) or not is_well_formed(thumbprint):
        return None
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    if not isinstance(keys, list):
        return None
    matches = [
        k
        for k in keys
        if isinstance(k, dict)
        and k.get("key_thumbprint") == thumbprint
        and isinstance(k.get("public_key"), str)
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    narrowed = [k for k in matches if agent_id and k.get("agent_id") == agent_id] or matches
    for k in narrowed:
        if kid and kid == k.get("kid"):
            return k
    return narrowed[0]


def match_signing_key(
    jwks: dict, kid: str, agent_id=None, issuer_id=None, org_id=None, key_thumbprint=None
):
    """Return the one jwks entry a receipt's signature is checked against.

    Signed key_thumbprint, then exact key id, then the agent bind, then the bare-kid
    issuer match. Every caller resolves through here so the key that verifies a
    signature and the key whose status and issuer the axes weigh are one entry;
    resolving them separately lets a receipt verify against a key found one way
    while revocation and issuer read a key found the other.

    Order carries the security. A cloud receipt puts the org id in kid and the
    directory publishes issuer_id on every key that org owns, so an org-shaped kid
    matches each sibling alike and list position picks one. Position binds nothing -
    key_thumbprint, agent_id and issuer_id all sit inside the signed bytes - while
    the sibling it lands on holds other key bytes, another revocation status and
    another agent. The thumbprint names the exact key even after a rotation leaves
    an agent with two published keys; the agent bind carries the org_id a hash-mode
    receipt signs; both resolve before the loose match. A thumbprint that names a
    key the signature was not made with is caught downstream: the signature fails
    against it, the agent bind finds the real signer, and key_binding reports the
    substitution.
    """
    entry = _match_key_by_thumbprint(jwks, key_thumbprint, agent_id, kid)
    if entry is not None:
        return entry
    entry = _match_key_by_id(jwks, kid)
    if entry is not None:
        return entry
    entry = _match_key_by_agent(jwks, agent_id, issuer_id, org_id)
    if entry is not None:
        return entry
    return _match_key(jwks, kid)


def _entry_material(entry):
    # (public_key_bytes, status, alg) of a matched entry, or three Nones for a miss.
    if entry is None:
        return None, None, None
    return _b64decode(entry["public_key"]), entry.get("status"), entry.get("alg")


def _signing_key_entry(jwks: dict, kid: str, payload: dict, envelope: dict):
    # The standalone paths resolve exactly as the oracle adapter does, one entry for every axis.
    return match_signing_key(
        jwks,
        kid,
        payload.get("agent_id") or envelope.get("agent_id"),
        payload.get("issuer_id"),
        payload.get("org_id") or envelope.get("org_id"),
        payload.get("key_thumbprint"),
    )


    # Return the issuer id a jwks entry publishes, or None when it names no string.
def key_issuer_of(entry):
    issuer = entry.get("issuer_id") if entry else None
    return issuer if isinstance(issuer, str) else None


    # Return the org id a jwks entry publishes, or None when the value is not one.
def key_org_of(entry):
    org = entry.get("org_id") if entry else None
    return org if _is_org_id(org) else None


    # Return the revoked_at a jwks entry publishes, if any.
def revoked_at_of(entry):
    return entry.get("revoked_at") if entry else None


def resolve_key(jwks: dict, kid: str):
    """Return (public_key_bytes, status, alg) for the receipt's signature.kid.

    The public jwks directory lists each key under both its bare issuer id and
    its crypto key id; match either so the bare-kid wire form resolves.
    """
    k = _match_key(jwks, kid)
    if k is None:
        return None, None, None
    return _b64decode(k["public_key"]), k.get("status"), k.get("alg")


def resolve_key_issuer(jwks: dict, kid: str):
    """Return the issuer id the jwks publishes for the key kid resolves to.

    Reads the same entry resolve_key returns, so the bind weighs the key that
    actually verified. None when kid resolves nothing or publishes no string issuer.
    """
    return key_issuer_of(_match_key(jwks, kid))


def resolve_key_by_agent_id(jwks: dict, agent_id: str, issuer_id: str, org_id: str | None = None):
    """Return (public_key_bytes, status, alg, kid, key_issuer_id) for an agent key.

    Cloud receipts set signature.kid to the issuer (org) id but sign with the
    agent's own key, and the JWKS publishes agent_id per key, so the signing key
    resolves by agent_id. agent_id is attacker-controlled, so a match also requires
    the key's published issuer_id (or org_id) to equal the claimed issuer: without
    that bind a valid key from any org would verify a receipt claiming another.
    """
    k = _match_key_by_agent(jwks, agent_id, issuer_id, org_id)
    if k is None:
        return None, None, None, None, None
    return (
        _b64decode(k["public_key"]),
        k.get("status"),
        k.get("alg"),
        k.get("kid"),
        k.get("issuer_id"),
    )


def resolve_revoked_at(jwks: dict, kid: str):
    """Return the JWKS revoked_at for kid if published, else None.

    Optional field: the public JWKS publishes status today but not the
    timestamp, so the key-status axis falls back to a bare status gate. Reads the
    same entry resolve_key returns, so every axis weighs one key, not two.
    """
    return revoked_at_of(_match_key(jwks, kid))


def check_issuer_binding(key_issuer_id, claimed_issuer_id):
    """Gate the verdict on the verifying key belonging to the claimed issuer.

    One jwks serves every org, so a valid signature proves only that the signer
    holds some published key. The receipt is the claimed issuer's only when the
    directory publishes that key under the receipt's server-assigned issuer_id.
    """
    if isinstance(claimed_issuer_id, str) and claimed_issuer_id and key_issuer_id == claimed_issuer_id:
        return "PASS", f"signing key is published under the claimed issuer {claimed_issuer_id}"
    return "FAIL", (
        f"signing key is published under issuer {key_issuer_id!r}, not the "
        f"claimed issuer {claimed_issuer_id!r}"
    )


def resolve_key_org(jwks: dict, kid: str):
    """Return the org id the jwks publishes for the key kid resolves to, if any.

    A value that is not an org id cannot serve as one, so it reads as unpublished
    and the issuer branch decides, rather than counting as a mismatch.
    """
    return key_org_of(_match_key(jwks, kid))


#: Same body text as the TypeScript ORG_ID_RE. Anchored by fullmatch, not by $,
#: since python's $ also matches before a trailing newline and ECMAScript's does not.
_ORG_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _is_org_id(value) -> bool:
    """True for a canonical dashed UUID, the only form an org id takes on the wire.

    Strict on purpose. A permissive parse would read urn:uuid: or undashed text as
    an org id and then FAIL the bind against the canonical org_id beside it, which
    false-FAILs an honest receipt. Anything else is a label: SKIP, never PASS.
    """
    return isinstance(value, str) and _ORG_ID_RE.fullmatch(value) is not None


def _org_key(value):
    """Case-folded form of an org id, other values unchanged.

    Hex case carries no meaning in a UUID, so two spellings name one org and
    comparing them case-sensitively false-FAILs an honest receipt. Folding is
    limited to values that are org ids, which are ASCII, so the two languages
    fold identically. A label keeps its case, since a label is not hex.
    """
    return value.lower() if _is_org_id(value) else value


def check_org_binding(key_issuer_id, key_org_id, claimed_org_id):
    """Bind a hash-mode receipt's org_id to the org the directory names for its key.

    issuer_id is the org's legal entity or its id, so it equals org_id only while
    no org sets a legal entity; prefer the published per-key org_id, the field the
    server can guarantee equal. With neither, the claim is not comparable offline:
    SKIP rather than FAIL an honest receipt.
    """
    if not isinstance(claimed_org_id, str) or not claimed_org_id:
        return "FAIL", f"receipt org_id is {claimed_org_id!r}, so there is no org to bind"
    if _org_key(claimed_org_id) in (_org_key(key_org_id), _org_key(key_issuer_id)):
        return "PASS", f"signing key is published under the claimed org {claimed_org_id}"
    if key_org_id is None and not _is_org_id(key_issuer_id):
        return "SKIPPED", (
            f"jwks names issuer {key_issuer_id!r} for this key, a label rather than an "
            f"org id, so org {claimed_org_id} cannot be confirmed offline; publish "
            f"org_id per key to close this"
        )
    return "FAIL", (
        f"signing key is published under org {key_org_id or key_issuer_id!r}, not the "
        f"claimed {claimed_org_id!r}"
    )


#: Key statuses that revoke trust in the signing key for attestation.
REVOKED_KEY_STATUSES = {"revoked", "suspended", "compromised"}


def _has_trusted_pre_revocation_anchor(trusted_times, revoked_at) -> bool:
    """True when a cryptographically verified anchor proves the envelope existed
    at or before the key's revoked_at.

    Only then is the self-attested issued_at corroborated for the historical verify
    path: an anchor proves the envelope (payload + signature) existed at the anchor's
    own proven time, so a time at/before revocation rules out a signature made with
    the revoked key. An anchor with no proven time (an unplaced OTS proof) is False.
    """
    if not trusted_times or not revoked_at:
        return False
    rev = _parse_stamp(revoked_at)
    if rev is None:
        return False
    return any(t <= rev for t in trusted_times)


def check_key_status(status, issued_at: str, revoked_at=None, has_trusted_anchor: bool = False):
    """Gate the verdict on the signing key's published status.

    The public JWKS marks a key revoked once its agent is revoked; a receipt signed
    by such a key must not PASS offline, mirroring the hosted /verify. With a
    precise revoked_at, prefer the at-or-before-issuance check so a receipt signed
    BEFORE revocation still PASSes; without one the axis cannot place issuance, so
    any revoked key fails.

    has_trusted_anchor must be an anchor the caller CRYPTOGRAPHICALLY verified as
    pre-revocation; mere presence is not enough (anchors are unsigned and
    attacker-addable). Without one the self-attested issued_at is backdateable, so
    that case downgrades to SKIPPED (INCOMPLETE), never a hiding PASS. Defaults
    False so a caller that ignores the parameter never hides behind a bare timestamp.
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


    # ML-DSA-65 verify. Returns (result, note); result in PASS/FAIL/SKIPPED.
def verify_signature(pk: bytes, msg: bytes, sig: bytes, alg: object):
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


#: Basic-format UTC offset (+0200, +02); fromisoformat reads it only from CPython 3.11 on.
#: Matched after the 10-char date so a bare "2020-01-01" is never mistaken for an offset.
_BASIC_OFFSET_RE = re.compile(r"([+-])(\d{2}):?(\d{2})?$")

#: Colon-separated clock time, the spelling every supported version reads alike.
_CLOCK_RE = re.compile(r"^[T ](\d{2}):(\d{2})(?::(\d{2}))?")


def _check_clock_range(tail: str) -> None:
    """Raise for an out-of-range clock field, so one stamp draws one verdict.

    CPython reads hour 24 as the next midnight from 3.14 on and refuses it on
    every version this package claims, so the range is checked here instead.
    """
    m = _CLOCK_RE.match(tail)
    if m is None:
        return
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if hh > 23 or mm > 59 or ss > 59:
        raise ValueError(f"clock field out of range: {m.group(0)[1:]!r}")


def _extended_offset(issued_at: str) -> str:
    """Rewrite a basic-format UTC offset to the extended form, else pass through.

    One receipt has to draw one verdict on every supported Python: "+0200" reads
    FAIL on 3.10 and PASS on 3.11 without this. An out-of-range field raises,
    since CPython 3.11+ reads "+0260" as a 3-hour offset and shifts the instant.
    """
    date, tail = issued_at[:10], issued_at[10:]
    _check_clock_range(tail)
    m = _BASIC_OFFSET_RE.search(tail)
    if m is None:
        return issued_at
    hours, minutes = int(m.group(2)), int(m.group(3) or 0)
    if hours > 23 or minutes > 59:
        raise ValueError(f"utc offset out of range: {m.group(0)!r}")
    return f"{date}{tail[: m.start()]}{m.group(1)}{hours:02d}:{minutes:02d}"


#: Extended ISO-8601 form pinned so the basic form draws one verdict on every parser.
_EXTENDED_STAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    r"(?:Z|[+-]\d{2}(?::?\d{2})?)?$"
)


def _parse_stamp(value: object):
    """Parse an ISO-8601 stamp, or None when it is unreadable.

    One parser for every time axis, so a stamp the skew axis refuses is not
    quietly accepted by the expiry axis. A stamp with no zone reads as UTC.
    The grammar is matched explicitly before fromisoformat, whose acceptance of
    the basic form differs by version; one stamp must draw one verdict on all.
    """
    text = str(value)
    if _EXTENDED_STAMP_RE.match(text) is None:
        return None
    try:
        ts = datetime.fromisoformat(_extended_offset(text.replace("Z", "+00:00")))
    except (ValueError, AttributeError, TypeError):
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def check_skew(issued_at: str):
    ts = _parse_stamp(issued_at)
    if ts is None:
        return "FAIL", f"unparseable issued_at {issued_at!r}"
    skew = (ts - datetime.now(timezone.utc)).total_seconds()
    if skew > SKEW_BOUND_SECONDS:
        return (
            "FAIL",
            f"issued_at {skew:.0f}s ahead of wall clock (> {SKEW_BOUND_SECONDS}s)",
        )
    return "PASS", f"skew {skew:.0f}s within bound"


#: What this verifier does NOT check, declared in the output rather than in a document.
#:
#: A verifier that reports only what it checked lets a reader mistake silence for
#: coverage. Every result carries this list, passing results included, so the
#: reader sees the boundary of the claim beside the claim itself instead of having
#: to find it in a README that ships separately from the tool.
#:
#: `condition` is None when the check is never performed at any invocation, and
#: names the input that would enable it when the gap is one this caller can close.
NOT_CHECKED = (
    {
        "check": "tsa_certificate_path",
        "requirement": "anchor trust",
        "reason": (
            "no X.509 chain walk from the RFC 3161 signing certificate to a public "
            "root; offline trust comes only from the TSA keys the caller pins"
        ),
        "condition": "--tsa-key pins the key this tool will trust; it does not build a path to it",
    },
    {
        "check": "tsa_certificate_revocation",
        "requirement": "anchor trust",
        "reason": "no CRL or OCSP lookup, so a revoked TSA certificate is not detected",
        "condition": None,
    },
    {
        "check": "policy_digest_resolution",
        "requirement": "verifier check on policy_digest",
        "reason": (
            "the referenced policy artefact is not fetched or rehashed; the digest is "
            "checked for shape only, and resolving it needs the Audit Pack manifest"
        ),
        "condition": None,
    },
    {
        "check": "aggregate_anchor_inclusion",
        "requirement": "aggregate anchoring",
        "reason": (
            "where an anchor covers a batch rather than this receipt, the inclusion "
            "proof linking this receipt to that aggregate is not walked"
        ),
        "condition": None,
    },
    {
        "check": "framework_mapping_claims",
        "requirement": "taxonomy extension fields",
        "reason": (
            "the caller-supplied taxonomy lists are carried under the signature but "
            "their content is not evaluated against any framework; a receipt can "
            "name a control it never satisfied and this tool will not say so"
        ),
        "condition": None,
    },
    {
        "check": "receipt_set_completeness",
        "requirement": "selective omission",
        "reason": (
            "this tool verifies the receipt it is handed; it cannot tell that a "
            "receipt was withheld from the set it belongs to"
        ),
        "condition": "the seq axis detects a gap only when the predecessor is supplied",
    },
    {
        "check": "key_revocation_freshness",
        "requirement": "key resolution",
        "reason": (
            "revocation state is read from the key directory as supplied; the tool "
            "does not re-fetch it, so a key revoked after that snapshot reads current"
        ),
        "condition": "--jwks is trusted as given",
    },
    {
        "check": "opentimestamps_block_placement",
        "requirement": "anchor cryptographic re-verification",
        "reason": "the merkle path is not landed in a bitcoin block without a header source",
        "condition": "--bitcoin-headers",
    },
    {
        "check": "counterparty_receipt_resolution",
        "requirement": "counterparty binding",
        "reason": (
            "the bound peer envelope is not resolved from any store; without it the "
            "axis reports the claim unverifiable rather than checking the digest"
        ),
        "condition": "the counterparty envelope passed by the caller",
    },
    {
        "check": "chain_predecessor_retrieval",
        "requirement": "hash-chain linkage",
        "reason": "the predecessor receipt is not fetched; the link is checked only against one supplied",
        "condition": "the predecessor payload passed by the caller",
    },
)


def not_checked_declaration() -> list:
    """Return the non-coverage declaration as a fresh list of fresh dicts.

    Copied on every call so a caller mutating a result cannot edit the module
    constant and quietly narrow what every later result declares.
    """
    return [dict(entry) for entry in NOT_CHECKED]


def check_payload_digest(payload: dict):
    """Recompute payload_digest from the context carried in the same receipt.

    A receipt that carries BOTH the context and a digest over it is checkable with
    no external data at all, and the two disagreeing means one of them is a lie:
    the digest is what downstream systems bind the action content to, so a receipt
    whose own context does not hash to its own digest is internally inconsistent.
    Nothing checked this before, so an issuer could sign a benign context beside a
    digest committing to something else entirely and every verifier passed it.

    Absence PASSes on both sides. Hash mode carries no context, and a payload-mode
    receipt may legitimately omit it (redaction, or an org that does not store the
    full payload), so a missing context is the ordinary case and never a failure.
    """
    if not isinstance(payload, dict):
        return "PASS", "no signed payload; no digest to recompute"
    digest = payload.get("payload_digest")
    if digest is None:
        return "PASS", "receipt binds no payload_digest; nothing to recompute"
    if not isinstance(digest, dict):
        return "FAIL", f"payload_digest is {type(digest).__name__}, not an object"
    claimed = digest.get("hash")
    if not isinstance(claimed, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed):
        return "FAIL", f"payload_digest.hash {claimed!r} is not 64 lowercase hex"
    claimed_size = digest.get("size")
    if claimed_size is not None and (
        not isinstance(claimed_size, int) or isinstance(claimed_size, bool) or claimed_size < 0
    ):
        return "FAIL", f"payload_digest.size {claimed_size!r} is not a non-negative integer"
    if payload.get("context") is None:
        # An explicit null reads as absent: the hosted verifier returns payload: null
        # for redacted and hash-only receipts, and null is not a context to hash
        return "PASS", "no context carried; payload_digest not recomputable here"

    try:
        encoded = canonical_json(payload["context"])
    except (TypeError, ValueError, RecursionError):
        return "SKIPPED", "context is not canonicalisable, so payload_digest cannot be recomputed"
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != claimed:
        return "FAIL", (
            f"payload_digest_mismatch: receipt binds {claimed[:16]}.., "
            f"its own context hashes to {actual[:16]}.."
        )
    if claimed_size is not None and claimed_size != len(encoded):
        return "FAIL", (
            f"payload_digest_mismatch: size claims {claimed_size}, "
            f"canonical context is {len(encoded)} bytes"
        )
    return "PASS", f"payload_digest rederives from the carried context ({len(encoded)} bytes)"


def _envelope_hash(envelope: dict) -> str:
    """Base64 SHA-256 over the originating envelope's full canonical bytes.

    Scope includes the originator's signature bytes, which is the rule that stops a
    re-signing intermediary from escaping detection. Mirrors the cloud's
    core/envelope.py compute_envelope_hash.
    """
    return base64.b64encode(hashlib.sha256(canonical_json(envelope)).digest()).decode()


def check_counterparty_binding(payload: dict, originator: dict | None = None):
    """Weigh a claimed cross-agent binding instead of letting it ride unchecked.

    counterparty_binding is caller-supplied: an issuer can attach one asserting
    "the counterparty acknowledged this" over a receipt nobody else ever saw. The
    hosted verifier resolves receipt_ref against its own database, but an offline
    third party has no database, so before this axis existed a fabricated binding
    reached a plain verified verdict with the corroboration claim unexamined.

    Absence PASSes: a receipt that claims no corroboration is the ordinary case and
    must not be penalised, and a skip would block every one of them. A claim the
    verifier cannot resolve reports SKIPPED, which blocks, so an unchecked binding
    can never read as corroborated - the same rule anchors follow, where presence
    alone never passes the axis. A malformed or mismatched binding is a proven
    break and FAILs, because the receipt asserts a binding it cannot satisfy.
    """
    if not isinstance(payload, dict):
        return "PASS", "no signed payload; no counterparty binding to check"
    cpb = payload.get("counterparty_binding")
    if cpb is None:
        return "PASS", "no counterparty binding; content is unilaterally asserted"
    if not isinstance(cpb, dict):
        return "FAIL", f"counterparty_binding is {type(cpb).__name__}, not an object"

    receipt_ref = cpb.get("receipt_ref")
    envelope_hash = cpb.get("envelope_hash")
    if not isinstance(receipt_ref, str) or not receipt_ref:
        return "FAIL", "counterparty_binding.receipt_ref missing or not a string"
    if not isinstance(envelope_hash, str) or not envelope_hash:
        return "FAIL", "counterparty_binding.envelope_hash missing or not a string"
    try:
        raw = base64.b64decode(envelope_hash, validate=True)
    except Exception:
        return "FAIL", f"counterparty_binding.envelope_hash {envelope_hash!r} is not base64"
    if len(raw) != 32:
        return "FAIL", (
            f"counterparty_binding.envelope_hash decodes to {len(raw)} bytes, not a sha-256"
        )

    expect_ack_from = cpb.get("expect_ack_from")
    if expect_ack_from is not None and not isinstance(expect_ack_from, str):
        return "FAIL", "counterparty_binding.expect_ack_from is not a string"

    if not isinstance(originator, dict):
        # Structurally sound but nothing here corroborates it; blocking on purpose
        return "SKIPPED", (
            f"counterparty binding claims {receipt_ref}; no originating receipt supplied, "
            "so the corroboration is unchecked"
        )

    actual = _envelope_hash(originator)
    if actual != envelope_hash:
        return "FAIL", (
            f"counterparty_mismatch: binding commits {envelope_hash[:16]}.., "
            f"supplied originator hashes to {actual[:16]}.."
        )
    if expect_ack_from is not None and payload.get("issuer_id") != expect_ack_from:
        return "FAIL", (
            f"counterparty_mismatch: binding expects an acknowledgment from "
            f"{expect_ack_from}, this receipt is issued by {payload.get('issuer_id')!r}"
        )
    return "PASS", f"counterparty binding rederives from the supplied {receipt_ref}"


def check_key_binding(payload: dict, alg, public_key):
    """Recompute the signed key_thumbprint from the resolved key and compare.

    The receipt names its own signing key INSIDE the signed bytes, so a key swapped
    under the same kid stops rederiving the digest the issuer committed to. A
    mismatch is a proven binding break, never a warning: `key_binding` sits in
    `_INVALID_FAIL_AXES`, so the verdict reads unverified with failure_class invalid.

    Absence is the profile's legacy case and stays conformant, so it PASSes with a
    note rather than skipping; a skip would block every pre-binding receipt ever
    issued. The two cases that cannot be recomputed - no key resolved, and a
    resolved key that is not raw ML-DSA of the width its own alg fixes - report
    SKIPPED, which blocks, so an unrecomputable binding never reads as verified.

    ``public_key`` is the key the signature axis actually verified against, not
    whatever the unsigned kid names, matching how issuer_bind and key_status resolve.
    """
    if not isinstance(payload, dict):
        return "PASS", "no signed payload; binding not checked"
    claimed = payload.get("key_thumbprint")
    if claimed is None:
        return "PASS", "receipt binds no key_thumbprint; binding not checked"
    if not is_well_formed(claimed):
        return "FAIL", f"key_thumbprint {claimed!r} is not sha256:<64 lowercase hex>"
    if not public_key:
        return "SKIPPED", "no signing key resolved, so the bound key_thumbprint cannot be recomputed"
    if not isinstance(alg, str) or not is_akp_public_key(alg=alg, public_key=public_key):
        # A KMS-backed row publishes a PEM in this column, and a PEM is not the key
        # material an AKP `pub` carries, so the digest is not reconstructible here
        return "SKIPPED", (
            "resolved key is not raw ML-DSA material, so the bound key_thumbprint "
            "cannot be recomputed"
        )
    actual = thumbprint_for_key(alg=alg, public_key=public_key)
    if actual == claimed:
        return "PASS", f"key_thumbprint rederives from the resolved {alg} key"
    return "FAIL", f"key_substituted: receipt binds {claimed}, resolved key computes {actual}"


    # Axis-only expiry flag, mirrors the hosted signature_expired label; never folds the verdict (426).
def check_expiry(payload: dict):
    if not isinstance(payload, dict) or "expires_at" not in payload:
        return "PASS", "receipt declares no expiry"
    raw = payload["expires_at"]
    ts = _parse_stamp(raw)
    if ts is None:
        return "FAIL", f"unreadable expires_at {raw!r}; refused rather than read as no expiry"
    delta = (ts - datetime.now(timezone.utc)).total_seconds()
    if delta < 0:
        return "FAIL", f"expires_at {raw} lapsed {-delta:.0f}s ago"
    return "PASS", f"expires_at {raw} is {delta:.0f}s ahead"


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


    # Closed-key-set and false-attestation rules for controls_evaluated.
def check_controls_evaluated(payload: dict):
    ce = payload.get("controls_evaluated")
    if ce is None:
        return "PASS", "receipt declares no controls_evaluated block"
    if not isinstance(ce, dict):
        return "FAIL", "controls_evaluated must be an object (false_control_attestation_guard)"
    unknown = sorted(k for k in ce if k not in ALLOWED_CONTROL_KEYS)
    if unknown:
        return "FAIL", (
            f"controls_evaluated keys outside the closed set: {','.join(unknown)} "
            f"(false_control_attestation_guard)"
        )
    quorum = ce.get("quorum")
    if quorum is not None:
        ah = quorum.get("attestation_hash") if isinstance(quorum, dict) else None
        if (
            not isinstance(quorum, dict)
            or quorum.get("fired") is not True
            or not isinstance(ah, str)
            or not _BARE_SHA256_RE.match(ah)
        ):
            return "FAIL", (
                "quorum member requires fired=true and a bare 64-hex "
                "attestation_hash (false_control_attestation_guard)"
            )
    policy = ce.get("policy")
    if policy is not None:
        mc = policy.get("matched_count") if isinstance(policy, dict) else None
        if (
            not isinstance(policy, dict)
            or policy.get("evaluated") is not True
            or isinstance(mc, bool)
            or not isinstance(mc, int)
            or mc < 1
        ):
            return "FAIL", (
                "policy member requires evaluated=true and a real matched_count "
                ">= 1 (false_control_attestation_guard)"
            )
    return "PASS", "controls_evaluated block conforms to the closed key set"


def check_nonce(payload: dict, seen_nonces: set | None = None):
    """Replay-candidate axis over the draft 5.7 nonce; a DIFFERENT receipt reusing an
    (issuer_id, nonce) pair fails, re-verifying the identical receipt does not."""
    nonce = payload.get("nonce")
    if nonce is None or nonce == "":
        return "PASS", "receipt declares no nonce; nothing to flag"
    issuer = payload.get("issuer_id", "")
    if seen_nonces is None:
        return "PASS", (
            "nonce present; this surface holds no seen-nonce index, so "
            "duplicate_emission_candidate stays false (cloud passthrough axis, "
            "draft 10.4)"
        )
    try:
        identity = hashlib.sha256(canonical_json(payload)).hexdigest()
    except (ValueError, TypeError, RecursionError):
        # Internal identity for the seen-nonce index only; never externally recomputed.
        identity = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    pair = f"{issuer}\x00{nonce}"
    entry = f"{pair}\x00{identity}"
    if entry in seen_nonces:
        return "PASS", "identical receipt re-verified; not a duplicate emission"
    if pair in seen_nonces:
        return "FAIL", (
            f"duplicate nonce {nonce!r} under issuer_id {issuer!r}: replay "
            f"candidate (draft 5.7)"
        )
    seen_nonces.add(pair)
    seen_nonces.add(entry)
    return "PASS", "nonce recorded in the seen-nonce index; no duplicate observed"


def check_structure(payload: dict):
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return "FAIL", f"missing required fields: {','.join(missing)}"
    rt = payload.get("type")
    if rt not in ALLOWED_TYPES:
        return "FAIL", f"type {rt!r} outside the allowed namespace"
    ce_res, ce_note = check_controls_evaluated(payload)
    if ce_res == "FAIL":
        return "FAIL", ce_note
    return "PASS", f"required fields present; type {rt}"


def run(
    envelope: dict,
    jwks: dict,
    predecessor_payload: dict | None,
    seen_nonces: set | None = None,
    *,
    trusted_tsa_keys=None,
    bitcoin_headers=None,
    counterparty: dict | None = None,
) -> int:
    if not isinstance(envelope, dict):
        print("Asqav receipt verification")
        print(
            f"  [FAIL] input       expected a JSON object receipt, got "
            f"{type(envelope).__name__}"
        )
        print("\n  => unverified (failure_class: unverifiable; no receipt object to verify)")
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
        print("\n  => unverified (failure_class: unverifiable; no payload to verify)")
        return 2
    shape = _scan_shape(envelope, max_depth=MAX_NESTING_DEPTH)
    if shape is None:
        shape = _scan_shape(predecessor_payload, max_depth=MAX_NESTING_DEPTH)
    if shape is not None:
        print("Asqav receipt verification")
        print(f"  [FAIL] input       {_SHAPE_MESSAGES[shape]}")
        print("\n  => unverified (failure_class: unverifiable; receipt not canonicalisable)")
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
        print("\n  => unverified (failure_class: unverifiable; receipt not canonicalisable)")
        return 2

    results = []
    results.append(("structure", *check_structure(payload)))
    results.append(("nonce", *check_nonce(payload, seen_nonces)))
    # Evaluated once, up front: the anchors axis reports it, and the key_status
    # axis weighs its cryptographically-proven timing against any revoked_at.
    anchor_eval = evaluate_anchors(
        envelope, trusted_tsa_keys=trusted_tsa_keys, bitcoin_headers=bitcoin_headers
    )

    entry = _signing_key_entry(jwks, kid, payload, envelope)
    pk, status, jwks_alg = _entry_material(entry)
    # The key the signature axis actually verified against, which is what the
    # key_binding axis has to rederive the bound thumbprint from.
    eff_pk, eff_alg = None, None
    if pk is None:
        results.append(("issuer_key", "FAIL", f"kid {kid!r} not in jwks directory"))
        results.append(("signature", "SKIPPED", "no issuer key to verify against"))
    else:
        _raw_sig = sig_obj.get("sig", "")
        sig = _b64decode(_raw_sig) if isinstance(_raw_sig, str) else b""
        sig_res = verify_signature(pk, msg, sig, alg or jwks_alg)
        eff_status = status
        eff_kid = kid if kid and kid in (entry.get("issuer_id"), entry.get("kid")) else entry.get("kid")
        eff_issuer = key_issuer_of(entry)
        eff_revoked_at = revoked_at_of(entry)
        eff_pk, eff_alg = pk, jwks_alg
        # Cloud receipts sign with the agent key though kid is the issuer id; fall back.
        # agent_id is attacker-controlled, so trust only a key whose issuer_id matches.
        agent_note = ""
        if sig_res[0] != "PASS":
            agent_id = payload.get("agent_id") or envelope.get("agent_id")
            org_bind = payload.get("org_id") or envelope.get("org_id")
            entry_a, sig_res_a, exhausted = _select_agent_bound_key(
                jwks, agent_id, payload.get("issuer_id"), org_bind, msg, sig,
                alg or jwks_alg,
            )
            pk_a, alg_a = None, None
            if entry_a is not None:
                pk_a, alg_a = _b64decode(entry_a["public_key"]), entry_a.get("alg")
                if sig_res_a is not None and sig_res_a[0] == "PASS":
                    sig_res = sig_res_a
                    eff_status, eff_kid = entry_a.get("status"), entry_a.get("kid")
                    eff_issuer = key_issuer_of(entry_a)
                    eff_revoked_at = revoked_at_of(entry_a)
                    eff_pk, eff_alg = pk_a, alg_a
                elif exhausted:
                    agent_note = "; no key published for this agent verified"
            if sig_res[0] != "PASS":
                cands = [(pk, alg or jwks_alg)]
                if pk_a is not None:
                    cands.append((pk_a, alg_a or alg or jwks_alg))
                sig_res = _pre_cutover_diagnostic(payload, sig, cands, sig_res)
        results.append(
            ("issuer_key", "PASS", f"resolved signing key {eff_kid} (status={eff_status}){agent_note}")
        )
        # kid picks the key and the attacker picks the kid, so bind the key that
        # actually verified back to the issuer the receipt claims.
        results.append(
            ("issuer_bind", *check_issuer_binding(eff_issuer, payload.get("issuer_id")))
        )
        # Only a cryptographically verified anchor at or before revoked_at proves
        # pre-revocation timing; presence alone never upgrades a revoked key.
        trusted_anchor = _has_trusted_pre_revocation_anchor(
            anchor_eval.trusted_times, eff_revoked_at
        )
        results.append(
            ("key_status", *check_key_status(eff_status, payload.get("issued_at", ""), eff_revoked_at, trusted_anchor))
        )
        results.append(("signature", *sig_res))

    # Outside the else on purpose: a receipt binding no thumbprint still reports the
    # axis, so the report says the binding was not checked rather than staying silent.
    results.append(("key_binding", *check_key_binding(payload, eff_alg, eff_pk)))
    results.append(("counterparty", *check_counterparty_binding(payload, counterparty)))
    results.append(("payload_digest", *check_payload_digest(payload)))
    results.append(("chain", *check_chain(payload, predecessor_payload)))
    results.append(("anchors", anchor_eval.result, anchor_eval.note))
    results.append(("skew", *check_skew(payload.get("issued_at", ""))))
    results.append(("expiry", *check_expiry(payload)))

    print("Asqav receipt verification")
    print(f"  canonical bytes: sha256:{hashlib.sha256(msg).hexdigest()}")
    print(f"  signature.kid:   {kid}")
    # The algorithm is per-receipt, verbatim from the signed envelope: ML-DSA-65
    # for cloud-issued receipts, Ed25519/ES256 for locally signed ones.
    print(f"  signature.alg:   {alg}")
    for name, res, note in results:
        mark = {"PASS": "ok", "FAIL": "FAIL", "SKIPPED": "skip"}[res]
        print(f"  [{mark:>4}] {name:<11} {note}")

    # Expiry reports on its own axis and never folds the verdict (criterion 426).
    verdict, failure_class = _fold_verdict(results, keyed=is_keyed_digest(payload))
    if verdict in (VERDICT_VERIFIED, VERDICT_VERIFIED_KEYED):
        code = 0
        print(f"\n  => {verdict}")
    elif failure_class == FAILURE_INVALID:
        code = 1
        print(f"\n  => {verdict} (failure_class: invalid)")
    else:
        code = 2
        print(f"\n  => {verdict} (failure_class: unverifiable; never reported verified)")
    return code


    # One structured-axis row carrying its per-axis failure token (418/438).
def _struct_axis(name: str, result: str, note: str) -> dict:
    return {
        "name": name,
        "result": result,
        "note": note,
        "failure_class": _axis_failure_class(name, result, note),
    }


def run_structured(
    envelope: dict,
    jwks: dict,
    predecessor_payload: dict | None = None,
    *,
    trusted_tsa_keys=None,
    bitcoin_headers=None,
    counterparty: dict | None = None,
    seen_nonces: set | None = None,
) -> dict:
    """Verify a receipt offline and return a structured result dict.

    Same logic as ``run()`` but returns a dict instead of printing and exiting; the
    public SDK uses the oracle adapter path for multi-format support.

    ``seen_nonces`` is the caller's replay index, exactly as ``run()`` takes it: pass
    one to have the nonce axis flag a different receipt reusing an (issuer_id, nonce)
    pair. Without it the axis still reports, and says it holds no index.

    Keys: ``verdict`` ("verified" | "unverified", criteria 418/438);
    ``failure_class`` ("invalid" | "unverifiable" when unverified, else None - the
    two are never collapsed, criterion 418); ``axes``, a list of
    ``{"name", "result", "note", "failure_class"}``; ``canonical_sha256``, hex
    SHA-256 of the canonical payload bytes; ``kid``; and ``alg``, verbatim from the
    wire (ML-DSA-65 cloud-issued, Ed25519/ES256 local).
    """
    if not isinstance(envelope, dict):
        return {
            "not_checked": not_checked_declaration(),
            "verdict": VERDICT_UNVERIFIED,
            "failure_class": FAILURE_UNVERIFIABLE,
            "axes": [
                _struct_axis(
                    "input",
                    "FAIL",
                    f"expected a JSON object receipt, got {type(envelope).__name__}",
                )
            ],
            "canonical_sha256": None,
            "kid": None,
            "alg": None,
        }
    envelope = normalise_envelope(envelope)
    payload = envelope.get("payload", envelope)
    if not isinstance(payload, dict):
        return {
            "not_checked": not_checked_declaration(),
            "verdict": VERDICT_UNVERIFIED,
            "failure_class": FAILURE_UNVERIFIABLE,
            "axes": [
                _struct_axis(
                    "payload",
                    "FAIL",
                    "receipt payload not available from this surface "
                    f"(got {_describe_value(payload)} instead of an object). "
                    "Verify with a saved receipt instead.",
                )
            ],
            "canonical_sha256": None,
            "kid": None,
            "alg": None,
        }
    shape = _scan_shape(envelope, max_depth=MAX_NESTING_DEPTH)
    if shape is None:
        shape = _scan_shape(predecessor_payload, max_depth=MAX_NESTING_DEPTH)
    if shape is not None:
        return {
            "not_checked": not_checked_declaration(),
            "verdict": VERDICT_UNVERIFIED,
            "failure_class": FAILURE_UNVERIFIABLE,
            "axes": [_struct_axis("input", "FAIL", _SHAPE_MESSAGES[shape])],
            "canonical_sha256": None,
            "kid": None,
            "alg": None,
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
            "not_checked": not_checked_declaration(),
            "verdict": VERDICT_UNVERIFIED,
            "failure_class": FAILURE_UNVERIFIABLE,
            "axes": [_struct_axis("input", "FAIL", _SHAPE_MESSAGES["too_deep"])],
            "canonical_sha256": None,
            "kid": None,
            "alg": None,
        }

    axes: list[dict] = []
    axes.append(_struct_axis("structure", *check_structure(payload)))
    axes.append(_struct_axis("nonce", *check_nonce(payload, seen_nonces)))
    # Evaluated once, up front: the anchors axis reports it, and the key_status
    # axis weighs its cryptographically-proven timing against any revoked_at.
    anchor_eval = evaluate_anchors(
        envelope, trusted_tsa_keys=trusted_tsa_keys, bitcoin_headers=bitcoin_headers
    )

    entry = _signing_key_entry(jwks, kid, payload, envelope)
    pk, status, jwks_alg = _entry_material(entry)
    # The key the signature axis actually verified against, which is what the
    # key_binding axis has to rederive the bound thumbprint from.
    eff_pk, eff_alg = None, None
    if pk is None:
        axes.append(_struct_axis("issuer_key", "FAIL", f"kid {kid!r} not in jwks directory"))
        axes.append(_struct_axis("signature", "SKIPPED", "no issuer key to verify against"))
    else:
        _raw_sig = sig_obj.get("sig", "")
        sig_bytes = _b64decode(_raw_sig) if isinstance(_raw_sig, str) else b""
        sig_res = verify_signature(pk, msg, sig_bytes, alg or jwks_alg)
        eff_status = status
        eff_kid = kid if kid and kid in (entry.get("issuer_id"), entry.get("kid")) else entry.get("kid")
        eff_issuer = key_issuer_of(entry)
        eff_revoked_at = revoked_at_of(entry)
        eff_pk, eff_alg = pk, jwks_alg
        # Cloud receipts sign with the agent key though kid is the issuer id; fall back,
        # mirroring run(). agent_id is attacker-controlled, so match on issuer_id.
        agent_note = ""
        if sig_res[0] != "PASS":
            agent_id = payload.get("agent_id") or envelope.get("agent_id")
            org_bind = payload.get("org_id") or envelope.get("org_id")
            entry_a, sig_res_a, exhausted = _select_agent_bound_key(
                jwks, agent_id, payload.get("issuer_id"), org_bind, msg, sig_bytes,
                alg or jwks_alg,
            )
            pk_a, alg_a = None, None
            if entry_a is not None:
                pk_a, alg_a = _b64decode(entry_a["public_key"]), entry_a.get("alg")
                if sig_res_a is not None and sig_res_a[0] == "PASS":
                    sig_res = sig_res_a
                    eff_status, eff_kid = entry_a.get("status"), entry_a.get("kid")
                    eff_issuer = key_issuer_of(entry_a)
                    eff_revoked_at = revoked_at_of(entry_a)
                    eff_pk, eff_alg = pk_a, alg_a
                elif exhausted:
                    agent_note = "; no key published for this agent verified"
            if sig_res[0] != "PASS":
                cands = [(pk, alg or jwks_alg)]
                if pk_a is not None:
                    cands.append((pk_a, alg_a or alg or jwks_alg))
                sig_res = _pre_cutover_diagnostic(payload, sig_bytes, cands, sig_res)
        axes.append(
            _struct_axis(
                "issuer_key",
                "PASS",
                f"resolved signing key {eff_kid} (status={eff_status}){agent_note}",
            )
        )
        # kid picks the key and the attacker picks the kid, so bind the key that
        # actually verified back to the issuer the receipt claims.
        axes.append(_struct_axis("issuer_bind", *check_issuer_binding(eff_issuer, payload.get("issuer_id"))))
        # Only a cryptographically verified anchor at or before revoked_at counts as
        # trusted timing; presence alone never upgrades a revoked key.
        trusted_anchor = _has_trusted_pre_revocation_anchor(
            anchor_eval.trusted_times, eff_revoked_at
        )
        axes.append(
            _struct_axis(
                "key_status",
                *check_key_status(eff_status, payload.get("issued_at", ""), eff_revoked_at, trusted_anchor),
            )
        )
        axes.append(_struct_axis("signature", sig_res[0], sig_res[1]))

    # Outside the else on purpose: a receipt binding no thumbprint still reports the
    # axis, so the report says the binding was not checked rather than staying silent.
    axes.append(_struct_axis("key_binding", *check_key_binding(payload, eff_alg, eff_pk)))
    axes.append(
        _struct_axis("counterparty", *check_counterparty_binding(payload, counterparty))
    )
    axes.append(_struct_axis("payload_digest", *check_payload_digest(payload)))
    axes.append(_struct_axis("chain", *check_chain(payload, predecessor_payload)))
    axes.append(_struct_axis("anchors", anchor_eval.result, anchor_eval.note))
    axes.append(_struct_axis("skew", *check_skew(payload.get("issued_at", ""))))
    axes.append(_struct_axis("expiry", *check_expiry(payload)))

    # Expiry reports on its own axis and never folds the verdict (criterion 426).
    verdict, failure_class = _fold_verdict(
        [(a["name"], a["result"], a["note"]) for a in axes],
        keyed=is_keyed_digest(payload),
    )

    return {
        "not_checked": not_checked_declaration(),
        "verdict": verdict,
        "failure_class": failure_class,
        "axes": axes,
        "canonical_sha256": hashlib.sha256(msg).hexdigest(),
        "kid": kid,
        "alg": alg,
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


def _load_tsa_keys(paths) -> list:
    """Read pinned TSA key material: PEM/DER X.509 certs, or base64/raw keys.

    Anchors sit outside the signed bytes, so trust material must come from the
    caller, never from the envelope; each file is one TSA public key or
    certificate the RFC 3161 check may trust.
    """
    keys = []
    for path in paths or []:
        try:
            with open(path, "rb") as fh:
                blob = fh.read()
        except OSError as exc:
            raise VerifierInputError(f"{path}: {exc.strerror or exc}") from exc
        text = blob.strip()
        if text.startswith(b"-----BEGIN"):
            keys.append(text + b"\n")
            continue
        try:
            keys.append(base64.b64decode(text, validate=True))
            continue
        except Exception:
            pass
        keys.append(bytes(text))
    return keys


def main() -> int:
    p = argparse.ArgumentParser(description="Standalone Asqav receipt verifier.")
    p.add_argument("--id", help="signature id to fetch from api.asqav.com")
    p.add_argument("--receipt", help="path to a saved receipt JSON, or - for stdin")
    p.add_argument("--jwks", help="path to a saved jwks.json, or - for stdin (offline)")
    p.add_argument(
        "--predecessor", help="path to predecessor receipt JSON for the chain check"
    )
    p.add_argument(
        "--tsa-key",
        action="append",
        default=[],
        metavar="FILE",
        help="pinned TSA public key or X.509 certificate (PEM, DER, or base64); "
        "repeatable. Without one, an RFC 3161 anchor's TSA signature cannot be "
        "trusted offline and the anchors axis reports unverifiable",
    )
    p.add_argument(
        "--bitcoin-headers",
        metavar="FILE",
        help="JSON map of bitcoin block height to {\"merkle_root\": <display hex>, "
        "\"time\": <ISO-8601>} for the OpenTimestamps block check; without it the "
        "block portion reports unverifiable",
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

        trusted_tsa_keys = _load_tsa_keys(args.tsa_key)
        bitcoin_headers = _load(args.bitcoin_headers) if args.bitcoin_headers else None
    except VerifierInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return run(
        envelope,
        jwks,
        predecessor_payload,
        trusted_tsa_keys=trusted_tsa_keys,
        bitcoin_headers=bitcoin_headers,
    )


if __name__ == "__main__":
    sys.exit(main())
