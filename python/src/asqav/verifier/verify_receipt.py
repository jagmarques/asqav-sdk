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

# Licensed Apache-2.0 (a file-level exception to the Elastic-2.0 SDK) so it
# ships in the exit artifact as a free, dependency-light offline tool.

"""Standalone Asqav receipt verifier: stdlib, optional crypto deps. Offline:
signature, JCS, chain, expiry, issuer, anchors. A skipped check never PASSes."""

from __future__ import annotations

import argparse
import base64
import binascii
import calendar
import hashlib
import json
import math
import re
import struct
import sys
import urllib.error
import urllib.parse
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
    # code-authorship receipt; producer-asserted record of who wrote a change.
    "protectmcp:lifecycle:code_authorship",
    "protectmcp:acknowledgment",
    "protectmcp:observation",
    "protectmcp:observation:result_bound",
}

#: Edition this verifier resolves type names against, vendored from
#: verifier/registry/type-namespaces.json (criterion 524)
REGISTRY_EDITION = "2026-09-15.v1"

#: Upstream registry version the vendored set reconciles (registry_source
#: in the same file; criterion 625): v1.1.0 nine plus oversight_ruling.
REGISTRY_VERSION = "1.1.0"

#: Draft-registered namespaces (anchor iana-type-namespaces), all ten
#: The engine implements ALLOWED_TYPES; oversight_ruling names the gap
REGISTRY_NAMESPACES = frozenset(
    {
        "protectmcp:decision",
        "protectmcp:restraint",
        "protectmcp:lifecycle",
        "protectmcp:lifecycle:configuration_change",
        "protectmcp:lifecycle:risk_acceptance",
        "protectmcp:lifecycle:oversight_ruling",
        "protectmcp:lifecycle:code_authorship",
        "protectmcp:acknowledgment",
        "protectmcp:observation",
        "protectmcp:observation:result_bound",
    }
)

#: Wire versions a verifier accepts (draft 5.3): v=2 is recognised for
#: verification, not released for emission. Widening is a registry decision.
RECOGNISED_WIRE_VERSIONS = frozenset({1, 2})
RECOGNISED_MODES = frozenset({"hash", "payload"})


#: Closed controls_evaluated key set; mirrors the client false-attestation rule.
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
#: stdlib json encoder recurses per level, crashing past ~1000 (Python <= 3.12).
MAX_NESTING_DEPTH = 200


#: Instant from which the platform emits RFC 8785 member order on the wire,
#: pinned to the emitter's production deploy; later receipts never retry.
JCS_UTF16_CUTOVER = "2026-09-02T18:05:09+00:00"


def _utf16_member_order(name: str) -> bytes:
    # Big-endian UTF-16 bytes compare in the order of their code units.
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
    """JCS canonical bytes, byte-identical to the server's canonical_json:
    UTF-16 member order, no whitespace, UTF-8, NaN/Infinity rejected."""
    return json.dumps(
        _utf16_ordered(obj),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=default,
    ).encode("utf-8")


def canonical_json_pre_cutover(obj) -> bytes:
    """The code-point member order the platform
    emitted before JCS_UTF16_CUTOVER."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def has_supplementary_member_name(obj) -> bool:
    """True when any object member name, at any
    depth, carries a character above U+FFFF."""
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
    # An unreadable issued_at earns no retry; the receipt is not verified.
    issued = _parse_stamp(payload.get("issued_at", ""))
    cutover = _parse_stamp(JCS_UTF16_CUTOVER)
    return issued is not None and cutover is not None and issued < cutover


def _pre_cutover_diagnostic(payload: dict, sig: bytes, candidates, sig_res):
    """Name the legacy dialect only when that explains a signature failure:
    a pre-cutover receipt whose member names reach above U+FFFF."""
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
                "legacy dialect: the signature verifies only under code-point member "
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


# JCS bytes of the two-key {payload, signature} object every anchor commits to.
def envelope_minus_anchors_jcs(env: dict) -> bytes:
    # An Audit Pack export has more top-level members than the signer anchored;
    # only payload and signature were committed, so only those two are hashed.
    return canonical_json({k: env[k] for k in ("payload", "signature") if k in env})


# The same envelope with the signature string in the other base64 alphabet.
def _sig_alphabet_twin(env: dict) -> dict | None:
    # A local-key signer commits base64url; an export may re-encode the bytes
    # in standard base64; the twin checks the commitment against both spellings.
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
    Explicit stack, no recursion: depth alone cannot crash this walk."""
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
        "outer_copies",
        "required_extensions",
        "policy_digest",
    }
)

#: Caller cutoff pin -> the axis judging it; run_structured never fetches one.
CUTOFF_AXES = {"policy_digest": "policy_digest", "at_time": "skew", "status_cutoff": "key_status"}


def _axis_failure_class(axis: str, result: str, note: str) -> str | None:
    """Map one axis outcome to its failure class. PASS carries none; SKIPPED
    did not recompute; FAIL is invalid on a proven break, else unverifiable."""
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
        if note.startswith("unregistered type ") or note.startswith(
            "malformed type "
        ):
            return FAILURE_INVALID
        return FAILURE_UNVERIFIABLE
    if axis == "expiry":
        if note.startswith("unreadable expires_at"):
            return FAILURE_UNVERIFIABLE
        return FAILURE_INVALID
    # issuer_key and anything unlisted: the recompute could not complete.
    return FAILURE_UNVERIFIABLE


    # Fold per-axis (name, result, note) rows into verdict + failure class.
def _fold_verdict(results, keyed: bool = False) -> tuple[str, str | None]:
    # Expiry is its own axis and never folds the verdict.
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
    """A receipt or JWKS repeated a JSON member name. The stdlib decoder is
    last-wins, so a duplicate is a terminal parse failure before hashing."""


#: Largest integer both SDKs canonicalise alike (2**53 + 1 has no exact double).
#: One above Number.isSafeInteger: the vector `number_2_to_53` pins 2**53.
MAX_CANONICAL_INTEGER = 2**53

#: Asqav profile bound: draft Section 4 ends the safe interval at 2**53 - 1.
#: The profile precheck refuses exactly representable 2**53; ingest keeps it.
MAX_PROFILE_INTEGER = 2**53 - 1


class UnsafeIntegerError(ValueError):
    """An integer with no exact double; two readers canonicalise it
    differently: JS rounds it, Python keeps it, so two byte strings result."""


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
    # int() itself raises past the interpreter digit cap, so judge the
    # literal first: leading zeros never count toward the value's size.
    digits = literal[1:] if literal[:1] == "-" else literal
    significant = digits.lstrip("0") or "0"
    if _decimal_digits_exceed_str_limit(len(significant)):
        raise UnsafeIntegerError(
            f"integer outside the canonical integer range +/-2**53: "
            f"<{len(significant)}-digit integer>; serialise it as a JSON "
            "string or an integer-rational pair"
        )
    signed = literal[:1] + significant if literal[:1] == "-" else significant
    value = int(signed)
    if not -MAX_CANONICAL_INTEGER <= value <= MAX_CANONICAL_INTEGER:
        raise UnsafeIntegerError(
            f"integer outside the canonical integer range +/-2**53: {literal}; serialise "
            "it as a JSON string or an integer-rational pair"
        )
    return value


def _decimal_digits_exceed_str_limit(count: int) -> bool:
    """True when a decimal literal of ``count`` digits defeats int()."""
    get_limit = getattr(sys, "get_int_max_str_digits", None)
    if get_limit is None:
        return False
    limit = get_limit()
    return bool(limit) and count > limit


def _int_exceeds_str_limit(node: int) -> bool:
    """True when str(node) would breach the interpreter digit cap."""
    get_limit = getattr(sys, "get_int_max_str_digits", None)
    if get_limit is None:
        return False
    limit = get_limit()
    return bool(limit) and abs(node) >= 10**limit


def _int_digit_count(node: int) -> int:
    """Exact decimal digit count via bit_length, never via str()."""
    magnitude = abs(node)
    digits = (magnitude.bit_length() * 30103) // 100000 + 1
    while 10**digits <= magnitude:
        digits += 1
    while digits > 1 and 10 ** (digits - 1) > magnitude:
        digits -= 1
    return digits


    # A float literal preserved with its source lexeme, mirroring
    # ``asqav.strict_json.RawFloat`` (this file ships standalone).
class _RawFloat(float):
    def __new__(cls, value, lexeme):
        obj = super().__new__(cls, value)
        obj.lexeme = lexeme
        return obj


def _raw_float(literal):
    return _RawFloat(float(literal), literal)


def _parse_object(text: str, source: str) -> dict:
    """Parse ``text`` as a JSON object, or raise VerifierInputError. Duplicate
    names and ints past 2**53 are refused: last-wins bytes are never hashed."""
    if not text or not text.strip():
        raise VerifierInputError(f"{source}: empty input, expected a JSON object")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_members,
            parse_int=_reject_unsafe_integer,
            parse_float=_raw_float,
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


class _SameHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse cross-host redirects; a hop elsewhere is not the issuer."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).hostname != urllib.parse.urlsplit(
            req.full_url
        ).hostname:
            raise VerifierInputError(f"{req.full_url}: refusing cross-host redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _get_json(url: str, *, timeout: int = 30) -> dict:
    if urllib.parse.urlsplit(url).scheme != "https":
        raise VerifierInputError(f"{url}: refusing non-https fetch")
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    opener = urllib.request.build_opener(_SameHostRedirectHandler)
    try:
        with opener.open(req, timeout=timeout) as resp:
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
    """Fetch the public Asqav JWKS directory as a dict. Snapshot it before
    going air-gapped; pass it to verify_receipt_offline or run()."""
    return _get_json(url, timeout=timeout)


    # Decode standard or url-safe base64, padding-tolerant.
def _b64decode(value: str) -> bytes:
    s = value.replace("-", "+").replace("_", "/")
    s += "=" * ((-len(s)) % 4)
    return base64.b64decode(s, validate=False)


# RFC 7638 JWK thumbprint over an ML-DSA key; inlined, mirrors key_thumbprint.

#: draft-ietf-cose-dilithium key type for ML-DSA key pairs
AKP_KTY = "AKP"

#: The only well-formed key_thumbprint shape; others are refused, not compared
_THUMBPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

#: FIPS 204 public-key widths in bytes, fixed by the standard. A KMS-backed
#: agent publishes a PEM where a raw key would sit, so width separates the two.
_ML_DSA_PUBLIC_KEY_BYTES = {"ML-DSA-44": 1312, "ML-DSA-65": 1952, "ML-DSA-87": 2592}


    # base64url with padding stripped - the encoding RFC 7638 `pub` requires.
def _b64url_nopad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


    # True when the bytes are a raw ML-DSA public key of the width alg fixes.
def is_akp_public_key(*, alg: str, public_key: bytes | None) -> bool:
    if not public_key:
        return False
    expected = _ML_DSA_PUBLIC_KEY_BYTES.get(alg)
    return expected is not None and len(public_key) == expected


    # True for `sha256:<64 lowercase hex>`; fullmatch, not match, because
    # Python's `$` also matches before a trailing newline; the JS test does not.
def is_well_formed(value) -> bool:
    return isinstance(value, str) and _THUMBPRINT_RE.fullmatch(value) is not None


def akp_jwk(*, alg: str, public_key: bytes) -> dict:
    """Build the required-members-only AKP JWK the thumbprint covers. `pub`
    is base64url without padding; the directory carries base64 instead."""
    if not alg:
        raise ValueError("alg is required to build an AKP JWK")
    if not public_key:
        raise ValueError("public_key is required to build an AKP JWK")
    return {"alg": alg, "kty": AKP_KTY, "pub": _b64url_nopad(public_key)}


def jwk_thumbprint(jwk: dict) -> str:
    """Return `sha256:<hex>` for a JWK already reduced to required members.
    Sorting here is what RFC 7638 calls for, so input order cannot move it."""
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
    """A token blob did not parse; the anchor
    reports unverifiable, never a crash."""


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
    if status not in (0, 1):  # granted or grantedWithMods; else a refusal
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
    # SignerInfo: version, sid, digestAlgorithm, [signedAttrs],
    # signatureAlgorithm, signature
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


    # The bytes a CMS signature covers: signedAttrs as SET OF, else eContent.
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


#: One PEM block, header line through footer line. A trust file holds a chain.
_PEM_BLOCK_RE = re.compile(rb"-----BEGIN [^-]+-----.*?-----END [^-]+-----\s*", re.S)


def _tsa_key_candidates(trusted_tsa_keys):
    """Split pinned TSA material into (raw key bytes, cryptography public keys).
    Without cryptography, PEM/X.509 entries stay raw bytes, never trusted."""
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
            if blob.startswith(b"-----BEGIN"):
                blocks = _PEM_BLOCK_RE.findall(blob)
                if blocks:
                    for block in blocks:
                        try:
                            pkeys.append(x509.load_pem_x509_certificate(block).public_key())
                        except ValueError:
                            try:
                                pkeys.append(load_pem_public_key(block))
                            except ValueError:
                                raw.append(block)
                    continue
            else:
                try:
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
            # cryptography is optional; without it this branch is unverifiable,
            # like the RSA/ECDSA one, not raising out of the anchors axis.
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
    """One RFC 3161 anchor -> (outcome, detail, genTime when verified). Trust is
    the pinned TSA keys, never the unsigned envelope. ``env_jcs``: imprints."""
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


#: Detached .ots header (opentimestamps DetachedTimestampFile.HEADER_MAGIC).
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
        # OTS varuints are LEB128: low 7 bits first, the high bit continues.
        val |= (byte & 0x7F) << (7 * (groups - 1))
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
            # Payload is varbytes: length prefix, then the height as a varuint;
            # height must consume it exactly (TimeAttestation reads it first).
            payload, off = _ots_varbytes(buf, off)
            height, hend = _ots_varuint(payload, 0)
            if hend != len(payload):
                raise _AnchorParseError("ots bitcoin attestation payload has trailing bytes")
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


def _derive_block_id(header: dict, merkle_internal: bytes):
    """Rebuild the 80-byte header and check it against the claimed id.

    Returns (block_id, meets_target), or None when the six derive fields
    are absent or unparseable. Byte order: version/time/bits/nonce
    little-endian, both hashes byte-reversed from display hex.
    """
    try:
        version = int(header["version"])
        prev = bytes.fromhex(header["previousblockhash"])[::-1]
        stamp = _parse_stamp(header.get("time"))
        bits = int(header["bits"])
        nonce = int(header["nonce"])
        if stamp is None or len(prev) != 32 or not 0 <= version < 2**32:
            return None
        if not 0 <= bits < 2**32 or not 0 <= nonce < 2**32:
            return None
        exp = (bits >> 24) - 3
        if exp < 0:
            return None
        raw = struct.pack("<I", version) + prev + merkle_internal
        raw += struct.pack("<III", int(stamp.timestamp()), bits, nonce)
        if len(raw) != 80:
            return None
        block_id = hashlib.sha256(hashlib.sha256(raw).digest()).digest()[::-1].hex()
        target = (bits & 0xFFFFFF) * 256**exp
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return block_id, int(block_id, 16) < target


def _check_ots_anchor(blob: bytes, bound: bytes, bitcoin_headers) -> tuple[str, str, object]:
    """One OpenTimestamps anchor -> (outcome, detail, block time). Commitment
    must equal the envelope digest; with no header source, never a PASS."""
    try:
        if len(blob) > _OTS_MAX_BLOB or not blob.startswith(_OTS_MAGIC):
            raise _AnchorParseError("not a detached .ots proof")
        off = len(_OTS_MAGIC)
        if blob[off] != 1:
            raise _AnchorParseError("unsupported .ots major version")
        off += 1
        hash_op = blob[off]
        off += 1
        if hash_op != 0x08:  # digest is sha256; other ops cannot commit it
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
    saw_bad_header = None
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
        derived = _derive_block_id(header, want)
        if derived is not None:
            block_id, meets_target = derived
            claimed = header.get("hash")
            if not isinstance(claimed, str) or block_id != claimed.lower():
                saw_bad_header = height
                continue
            if not meets_target:
                saw_bad_header = height
                continue
        when = _parse_stamp(header.get("time")) if header.get("time") else None
        return "verified", f"OpenTimestamps proof lands in bitcoin block {height}", when
    if saw_mismatch is not None:
        return "invalid", f"merkle path does not land in bitcoin block {saw_mismatch}", None
    if saw_bad_header is not None:
        return "invalid", f"bitcoin block {saw_bad_header} header fields do not reproduce its id", None
    return (
        "unverifiable",
        "offline OpenTimestamps check did not complete; commitment matches but no "
        "supplied bitcoin header confirms the block",
        None,
    )


#: (result, note, trusted_times) for the anchors axis; ``trusted_times`` holds
#: times of CRYPTOGRAPHICALLY verified anchors, weighable only vs revoked_at.
AnchorEvaluation = namedtuple("AnchorEvaluation", ("result", "note", "trusted_times"))

#: Anchor-entry members declared by draft -09 "anchoring"; members outside this
#: set are ignored by the checks and preserved in the envelope bytes.
ANCHOR_ENTRY_MEMBERS = frozenset({"type", "value", "status", "anchor_block_hash", "tsa_url", "operator_id"})


def evaluate_anchors(envelope: dict, *, trusted_tsa_keys=None, bitcoin_headers=None, trust_bundle=None) -> AnchorEvaluation:
    """Cryptographically evaluate every anchor against the envelope digest.
    Presence never PASSes; a trust bundle with a wrong digest SKIPs the axis."""
    anchors = envelope.get("anchors")
    # Absent is SKIPPED; a present non-list ({} / "" / 0) FAILs. run_structured
    # refuses a wire null at the structure gate; here None is absent or null.
    if anchors is None:
        return AnchorEvaluation("SKIPPED", "no anchors on this receipt", [])
    if not isinstance(anchors, list):
        return AnchorEvaluation("FAIL", f"anchors field is not a list (got {type(anchors).__name__})", [])
    if not anchors:
        return AnchorEvaluation("SKIPPED", "no anchors on this receipt", [])
    if trust_bundle is not None:
        _tb_ok, _tb_digest = verify_trust_bundle(trust_bundle)
        if not _tb_ok:
            return AnchorEvaluation(
                "SKIPPED",
                f"trust bundle digest mismatch (recorded {_tb_digest or 'absent'}); "
                "pinned keys not evaluated",
                [],
            )
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
        # Members outside ANCHOR_ENTRY_MEMBERS never reach the checks, and the
        # envelope bytes (hence the digest scope) are untouched by this read.
        known = {k: a[k] for k in a if k in ANCHOR_ENTRY_MEMBERS}
        atype = known.get("type", "?")
        blob = _anchor_bytes(known.get("value"))
        if blob is None:
            saw_invalid = True
            lines.append(f"    - {atype}: value MISSING or malformed")
            continue
        status = known.get("status")
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


def check_anchors(envelope: dict, *, trusted_tsa_keys=None, bitcoin_headers=None, trust_bundle=None):
    """(result, note) for the anchors axis; see evaluate_anchors."""
    ev = evaluate_anchors(
        envelope, trusted_tsa_keys=trusted_tsa_keys, bitcoin_headers=bitcoin_headers,
        trust_bundle=trust_bundle,
    )
    return ev.result, ev.note


def normalise_envelope(raw: dict) -> dict:
    """Project any shape, including an Audit Pack export, onto the 3-key
    envelope; export-side ids and copies are dropped before any axis."""
    # A signature object plus a payload dict: keep exactly the three members.
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
    """Return the first usable jwks entry whose crypto key id is
    exactly kid. A key id names one key, so it decides on its own."""
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
    """True when a key's published issuer or org equals the claim. agent_id is
    attacker-controlled, so this bind picks the key; hash mode signs org_id."""
    issuer_bound = issuer_id and issuer_id == k.get("issuer_id")
    org_bound = org_id and org_id == k.get("org_id")
    return bool(issuer_bound or org_bound)


#: Agent-bound keys a failing signature is tried against; rotation is bounded.
_AGENT_BIND_CANDIDATE_CAP = 8


def _match_keys_by_agent(jwks: dict, agent_id, issuer_id, org_id=None) -> list:
    """Return every usable jwks entry for an
    agent key bound to a claimed issuer."""
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
    # The first agent-bound key, for callers resolving before a signature.
    candidates = _match_keys_by_agent(jwks, agent_id, issuer_id, org_id)
    return candidates[0] if candidates else None


def _select_agent_bound_key(jwks: dict, agent_id, issuer_id, org_id, msg, sig, alg_hint):
    """Return (entry, signature result, exhausted)
    for the agent-bound key that signed msg."""
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
    """Return the one usable jwks entry publishing this key_thumbprint.
    It sits inside the signed bytes and outranks unsigned identifiers."""
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
    Order carries the security: directory position binds nothing."""
    # The signed key_thumbprint names one exact key, even after a rotation. If
    # it names a non-signer, the agent bind finds the signer; key_binding FAILs.
    entry = _match_key_by_thumbprint(jwks, key_thumbprint, agent_id, kid)
    if entry is not None:
        return entry
    entry = _match_key_by_id(jwks, kid)
    if entry is not None:
        return entry
    # The agent bind carries the org_id a hash-mode receipt signs, and runs
    # before the loose match.
    entry = _match_key_by_agent(jwks, agent_id, issuer_id, org_id)
    if entry is not None:
        return entry
    # Loosest: an org-shaped kid matches every sibling key alike, so list
    # position would pick one; the exact matches above run first.
    return _match_key(jwks, kid)


def _entry_material(entry):
    # (public_key_bytes, status, alg) of a matched entry; three Nones on a miss.
    if entry is None:
        return None, None, None
    return _b64decode(entry["public_key"]), entry.get("status"), entry.get("alg")


def _signing_key_entry(jwks: dict, kid: str, payload: dict, envelope: dict):
    # The standalone paths resolve as the oracle adapter, one entry per axis.
    return match_signing_key(
        jwks,
        kid,
        payload.get("agent_id") or envelope.get("agent_id"),
        payload.get("issuer_id"),
        payload.get("org_id") or envelope.get("org_id"),
        payload.get("key_thumbprint"),
    )


    # Return the issuer id a jwks entry publishes, or None if not a string.
def key_issuer_of(entry):
    issuer = entry.get("issuer_id") if entry else None
    return issuer if isinstance(issuer, str) else None


    # Return the org id a jwks entry publishes, or None if the value is not one.
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
    Reads the entry that actually verified; None when it publishes none."""
    return key_issuer_of(_match_key(jwks, kid))


def resolve_key_by_agent_id(jwks: dict, agent_id: str, issuer_id: str, org_id: str | None = None):
    """Return (public_key_bytes, status, alg,
    kid, key_issuer_id) for an agent key."""
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
    The public JWKS publishes status today, so the axis falls back to it."""
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
    """Return the org id the jwks publishes for the key kid resolves to.
    A non-org value reads as unpublished, so the issuer branch decides."""
    return key_org_of(_match_key(jwks, kid))


#: Same body text as the TypeScript ORG_ID_RE. Anchored by fullmatch, not by $,
#: since python's $ matches before a trailing newline; ECMAScript's does not.
_ORG_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _is_org_id(value) -> bool:
    """True for a canonical dashed UUID, the only org id form on the wire.
    Strict: urn:uuid: or undashed text would false-FAIL an honest receipt."""
    return isinstance(value, str) and _ORG_ID_RE.fullmatch(value) is not None


def _org_key(value):
    """Case-fold an org id, leaving other values unchanged. A UUID's hex
    case is meaningless; labels keep case and folding stays ASCII-only."""
    return value.lower() if _is_org_id(value) else value


def check_org_binding(key_issuer_id, key_org_id, claimed_org_id):
    """Bind a hash-mode receipt's org_id to
    the org the directory names for its key."""
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


def key_status_kind(status: object, revocation_reason: object = None) -> str:
    """The effective key state: compromise
    wins over every other published label."""
    s = (status or "").lower() if isinstance(status, str) else ""
    reason = revocation_reason.lower() if isinstance(revocation_reason, str) else ""
    return "compromised" if s == "compromised" or reason == "compromised" else s


def _has_trusted_pre_revocation_anchor(trusted_times, revoked_at) -> bool:
    """True when a verified anchor proves the
    envelope existed at or before revoked_at."""
    if not trusted_times or not revoked_at:
        return False
    rev = _parse_stamp(revoked_at)
    if rev is None:
        return False
    return any(t <= rev for t in trusted_times)


def check_key_status_at(cutoff, status, issued_at: str, revoked_at=None, has_trusted_anchor: bool = False, revocation_reason=None):
    """check_key_status at the caller's status cutoff: a revocation after the
    cutoff is ignored, and the cutoff never upgrades a FAIL."""
    row = check_key_status(status, issued_at, revoked_at, has_trusted_anchor, revocation_reason)
    if cutoff is None:
        return row
    cut = _parse_stamp(cutoff)
    if cut is None:
        return "SKIPPED", f"unparseable status_cutoff {cutoff!r}; key status not judged"
    if row[0] == "FAIL" or not revoked_at:
        return row
    rev = _parse_stamp(revoked_at)
    if rev is None:
        return "SKIPPED", f"unparseable revoked_at {revoked_at!r}; not placed against status_cutoff {cutoff}"
    if rev <= cut or key_status_kind(status, revocation_reason) not in REVOKED_KEY_STATUSES:
        return row
    return "PASS", f"signing key revoked_at {revoked_at} after status_cutoff {cutoff}; unrevoked at the cutoff"


def check_key_status(status, issued_at: str, revoked_at=None, has_trusted_anchor: bool = False, revocation_reason=None):
    """Gate the verdict on the signing key's published status. A revoked
    key never PASSes on timestamps alone; pre-issuance revocation FAILs."""
    s = (status or "").lower()
    kind = key_status_kind(status, revocation_reason)
    if kind == "compromised":
        return "FAIL", (
            f"signing key compromised at {revoked_at or 'an unpublished time'}; "
            "a compromised key never verifies"
        )
    if s == "decommissioned":
        return "SKIPPED", (
            f"signing key decommissioned; issuance at {issued_at} cannot be "
            "placed against retirement"
        )
    if not s or s == "unknown":
        return "SKIPPED", (
            "signing key status unknown; cannot confirm the key was active"
        )
    if s not in REVOKED_KEY_STATUSES:
        return "PASS", f"signing key status {status!r} is active"
    if revoked_at:
        # Pinned grammar first: bare fromisoformat reads the basic form on 3.11+
        # but raises on 3.10, so one stamp drew different verdicts per version
        rev = _parse_stamp(revoked_at)
        iss = _parse_stamp(issued_at)
        if rev is None or iss is None:
            return "FAIL", f"signing key status {status!r}; unparseable revoked_at/issued_at"
        if rev <= iss:
            return "FAIL", f"signing key {kind} at {revoked_at} on/before issuance {issued_at}"
        if not has_trusted_anchor:
            return (
                "SKIPPED",
                f"signing key {kind} at {revoked_at}; issued_at {issued_at} is "
                f"self-attested, no anchor proves pre-revocation timing",
            )
        return "PASS", f"signing key {kind} at {revoked_at}, after issuance {issued_at}"
    return (
        "SKIPPED",
        f"signing key status {status!r}; no revoked_at published, cannot place "
        "issuance relative to revocation",
    )


    # ML-DSA-65 / Ed25519 / ES256 verify; returns (result, note).
def verify_signature(pk: bytes, msg: bytes, sig: bytes, alg: object):
    token = (alg if isinstance(alg, str) else "").upper()
    if token in ("ED25519", "ES256"):
        display = "Ed25519" if token == "ED25519" else "ES256"
        # Lazy import: cryptography is an optional dep and must never load at
        # module level (the standalone surface pins that).
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric.ec import (
                ECDSA,
                SECP256R1,
                EllipticCurvePublicKey,
            )
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            from cryptography.hazmat.primitives.asymmetric.utils import (
                encode_dss_signature,
            )
        except ImportError:
            return "SKIPPED", f"run 'pip install cryptography' for the {display} check"
        # Same key and signature forms the oracle verifies: raw 32-byte Ed25519
        # key; 65-byte P-256 point, 64-byte raw r||s (DER for the library).
        if token == "ED25519":
            try:
                key = Ed25519PublicKey.from_public_bytes(pk)
            except Exception as exc:  # malformed key bytes
                return "FAIL", f"bad Ed25519 public key: {exc}"
            try:
                key.verify(sig, msg)
                return "PASS", "signature valid"
            except InvalidSignature:
                return "FAIL", "signature mismatch"
            except Exception as exc:  # malformed signature bytes
                return "FAIL", f"verify error: {exc}"
        try:
            key = EllipticCurvePublicKey.from_encoded_point(SECP256R1(), pk)
        except Exception as exc:  # malformed point bytes
            return "FAIL", f"bad P-256 public key: {exc}"
        if len(sig) != 64:
            return "FAIL", f"ES256 signature must be 64-byte raw r||s, got {len(sig)}"
        der = encode_dss_signature(
            int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big")
        )
        try:
            key.verify(der, msg, ECDSA(hashes.SHA256()))
            return "PASS", "signature valid"
        except InvalidSignature:
            return "FAIL", "signature mismatch"
        except Exception as exc:  # malformed signature bytes
            return "FAIL", f"verify error: {exc}"
    if token != "ML-DSA-65":
        return "SKIPPED", f"unsupported alg {alg!r} (this tool checks ML-DSA-65, Ed25519, ES256)"
    # dilithium-py is not constant-time, fine for a verify-only tool: it touches
    # only public data (key, signature, message), so no secret leaks by timing.
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


#: Basic-format UTC offset (+0200, +02); only 3.11+ fromisoformat reads it.
#: Matched after a 10-char date so a bare date is not read as an offset.
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
    """Rewrite a basic-format UTC offset to the extended form, else pass it.
    Without this, '+0200' FAILs on 3.10 and PASSes on 3.11."""
    date, tail = issued_at[:10], issued_at[10:]
    _check_clock_range(tail)
    m = _BASIC_OFFSET_RE.search(tail)
    if m is None:
        return issued_at
    hours, minutes = int(m.group(2)), int(m.group(3) or 0)
    if hours > 23 or minutes > 59:
        raise ValueError(f"utc offset out of range: {m.group(0)!r}")
    return f"{date}{tail[: m.start()]}{m.group(1)}{hours:02d}:{minutes:02d}"


#: Extended ISO-8601 form pinned so the basic form draws one verdict everywhere.
_EXTENDED_STAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    r"(?:Z|[+-]\d{2}(?::?\d{2})?)?$"
)


def _parse_stamp(value: object):
    """Parse an ISO-8601 stamp, or None when it is unreadable. One parser
    for every time axis; a stamp with no zone reads as UTC."""
    text = str(value)
    if _EXTENDED_STAMP_RE.match(text) is None:
        return None
    try:
        ts = datetime.fromisoformat(_extended_offset(text.replace("Z", "+00:00")))
    except (ValueError, AttributeError, TypeError):
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def check_at_time(payload: dict, at_time=None) -> list:
    """skew and expiry rows judged at the caller's at_time, else the wall
    clock; an unparseable at_time skips both."""
    if at_time is None:
        return [("skew", *check_skew(payload.get("issued_at", ""))), ("expiry", *check_expiry(payload))]
    now = _parse_stamp(at_time)
    if now is None:
        note = f"unparseable at_time {at_time!r}; not judged at the wall clock"
        return [("skew", "SKIPPED", note), ("expiry", "SKIPPED", note)]
    rows = [("skew", *check_skew(payload.get("issued_at", ""), now=now)), ("expiry", *check_expiry(payload, now=now))]
    return [(name, res, f"{note} (at_time {now.isoformat()})") for name, res, note in rows]


def check_policy_digest(payload: dict, pinned) -> tuple[str, str]:
    """Compare the signed policy_digest with the caller's pin; a pin with
    nothing to compare SKIPs."""
    held = payload.get("policy_digest")
    if not isinstance(pinned, str) or not pinned:
        return "SKIPPED", f"pinned policy_digest {pinned!r} is not a digest string"
    if not isinstance(held, str) or not held:
        return "SKIPPED", "policy_digest pinned but the receipt carries none to compare"
    if held != pinned:
        return "FAIL", f"policy_digest {held} differs from the pinned {pinned}"
    return "PASS", f"policy_digest matches the pinned {pinned}"


def check_skew(issued_at: str, *, now: datetime | None = None):
    ts = _parse_stamp(issued_at)
    if ts is None:
        return "FAIL", f"unparseable issued_at {issued_at!r}"
    skew = (ts - (now or datetime.now(timezone.utc))).total_seconds()
    if skew > SKEW_BOUND_SECONDS:
        return (
            "FAIL",
            f"issued_at {skew:.0f}s ahead of wall clock (> {SKEW_BOUND_SECONDS}s)",
        )
    return "PASS", f"skew {skew:.0f}s within bound"


#: Text members an Audit Pack entry restates beside the signed payload, each
#: paired with the signed member it must equal. No signature covers a copy.
_OUTER_TEXT_COPIES = (
    ("agent_id", "agent_id"),
    ("action_id", "action_id"),
    ("action_type", "action_type"),
    ("receipt_type", "type"),
    ("supersedes", "supersedes"),
)

#: Members with no signed counterpart: reported, never trusted. The withheld
#: flag is one: it is a claim about the entry that no signature covers.
_OUTER_UNATTESTED = (
    "signature_id", "capture_topology", "stale_pending", "corrupted", "corruption_message",
    "signed_bytes_withheld",
)

#: Members an entry carries by design, each weighed below or listed above. Any
#: other top-level name is a stray: held to its signed twin by name, or listed.
_OUTER_MEMBERS = frozenset((
    "payload", "signature", "signature_envelope", "anchors", "signature_id", "agent_id",
    "action_id", "action_type", "algorithm", "signed_at", "compliance_mode",
    "signature_b64", "public_key_b64", "message_b64", "message_sha256",
    "signed_bytes_withheld", "stale_pending", "capture_topology", "corrupted",
    "corruption_message", "receipt_type", "supersedes",
))

#: Members of the signature object itself; any other is an unsigned extra.
_SIGNATURE_MEMBERS = ("alg", "kid", "sig")

#: What the hosted /verify response puts under its own top-level type member.
_HOSTED_RESPONSE_TYPE = "signature"

#: Stand-in the digest-only export writes over a signed value it withholds.
_WITHHELD_MARKER = "withheld: re-export with include_signed_bytes=true for the signed bytes"

#: What each outer member must agree with, for the mismatch note.
_OUTER_SIGNED_LABEL = {
    "receipt_type": "type",
    "signed_at": "issued_at",
    "algorithm": "signature.alg",
    "signature": "signature.sig",
    "signature_envelope": "signature",
    "signature_b64": "signature.sig",
    "message_b64": "the canonical signed bytes",
    "message_sha256": "the canonical signed bytes",
    "public_key_b64": "the verifying key",
    "compliance_mode": "the payload",
}

#: Export timestamps: date, optional clock, optional zone (none is UTC).
_INSTANT_RE = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})"
    r"(?:[T ]([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:\.([0-9]+))?)?)?"
    r"(Z|[+-][0-9]{2}(?::?[0-9]{2})?)?"
)
_STRICT_B64_RE = re.compile(r"[A-Za-z0-9+/]*={0,2}")


def _zone_shift(zone: str | None) -> int | None:
    """Seconds east of UTC for a zone suffix, None when out of range."""
    if not zone or zone == "Z":
        return 0
    digits = zone[1:].replace(":", "")
    hours, minutes = int(digits[:2]), int(digits[2:4] or 0)
    if hours > 23 or minutes > 59:
        return None
    return (hours * 3600 + minutes * 60) * (1 if zone[0] == "+" else -1)


def _instant(text: object):
    """(epoch seconds, microsecond digits) of an export timestamp, else None."""
    m = _INSTANT_RE.fullmatch(text) if isinstance(text, str) else None
    if m is None:
        return None
    shift = _zone_shift(m.group(8))
    year, month, day, hour, minute, second = (int(g or 0) for g in m.groups()[:6])
    try:
        moment = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None
    if shift is None:
        return None
    return calendar.timegm(moment.utctimetuple()) - shift, (m.group(7) or "")[:6].rstrip("0")


def _same_instant(outer: object, signed: object) -> bool:
    if isinstance(outer, str) and outer == signed:
        return True
    left = _instant(outer)
    return left is not None and left == _instant(signed)


def _strict_b64(value: object):
    """Bytes of a standard or url-safe base64 string, None on anything else."""
    if not isinstance(value, str):
        return None
    text = value.replace("-", "+").replace("_", "/")
    text += "=" * ((-len(text)) % 4)
    if _STRICT_B64_RE.fullmatch(text) is None:
        return None
    try:
        return base64.b64decode(text, validate=True)
    except ValueError:
        return None


#: Transport named when history verifies under a non-contract encoding.
LEGACY_ENCODING_POLICY = "pre-2026-09-02 profile transport"


def _sig_carrier(value: object):
    """Name a signature string's wire form: contract, legacy, or None.

    Contract is unpadded base64url. Decodable history (padded urlsafe,
    standard) is legacy, still accepted with the policy named. None is junk.
    """
    if not isinstance(value, str) or not value:
        return None
    if _strict_b64(value) is None:
        return None
    if "+" in value or "/" in value or value.endswith("="):
        return "legacy"
    return "contract"


def _same_text(outer: object, signed: object) -> bool:
    return isinstance(outer, str) and outer == signed


def _same_bytes(outer: object, expected: bytes) -> bool:
    decoded = _strict_b64(outer)
    return decoded is not None and decoded == expected


def _same_json(outer: object, signed: object) -> bool:
    """Equal once canonicalised; a value that
    cannot be canonicalised is unequal."""
    try:
        return canonical_json(outer) == canonical_json(signed)
    except Exception:
        return False


def _signed_twin(payload: dict, name: str):
    """A signed member's value, None when it
    is absent or withheld from the export."""
    twin = payload.get(name)
    return None if twin == _WITHHELD_MARKER else twin


def _text_copy_outcomes(raw: dict, payload: dict) -> list:
    """(member, agrees) per carried text copy;
    agrees is None with no signed twin."""
    out = []
    for outer, signed in _OUTER_TEXT_COPIES:
        if raw.get(outer) is None:
            continue
        twin = _signed_twin(payload, signed)
        out.append((outer, None if twin is None else _same_text(raw[outer], twin)))
    return out


def _time_copy_outcomes(raw: dict, payload: dict) -> list:
    """signed_at must equal issued_at; an observation receipt may also carry the
    caller-observed context.ts there, because that is what its row stores. When
    that time is withheld or in a form this check cannot read, the copy is
    unattested rather than contradicted."""
    if raw.get("signed_at") is None:
        return []
    observed = payload.get("type") == "protectmcp:observation"
    context = payload.get("context")
    seen = context.get("ts") if observed and isinstance(context, dict) else None
    signed = [t for t in (_signed_twin(payload, "issued_at"), seen) if t is not None]
    if any(_same_instant(raw["signed_at"], t) for t in signed):
        return [("signed_at", True)]
    unreadable = observed and (seen is None or _instant(seen) is None)
    return [("signed_at", None if not signed or unreadable else False)]


def _envelope_copy_outcomes(raw: dict, sig_obj: dict) -> list:
    """algorithm and the signature strings
    against the normalised signature object."""
    out = []
    twin_sig = _strict_b64(sig_obj.get("sig"))
    if raw.get("algorithm") is not None:
        twin = sig_obj.get("alg")
        out.append(("algorithm", None if twin is None else _same_text(raw["algorithm"], twin)))
    if raw.get("signature_b64") is not None:
        out.append(("signature_b64", None if twin_sig is None else _same_bytes(raw["signature_b64"], twin_sig)))
    flat = raw.get("signature")
    if flat is not None and not isinstance(flat, dict):
        out.append(("signature", None if twin_sig is None else _same_bytes(flat, twin_sig)))
    if raw.get("signature_envelope") is not None:
        out.append(("signature_envelope", _same_json(raw["signature_envelope"], sig_obj)))
    return out


def _byte_copy_outcomes(raw: dict, msg: bytes, key_bytes) -> list:
    """The signed-bytes copies and the carried key against what verified.

    key_bytes is the key the signature verified under, None when it did not
    verify. The digest and the key are compared only then: before that, msg
    need not be the signed bytes, so a difference proves nothing.
    """
    out = []
    if raw.get("message_b64") is not None:
        out.append(("message_b64", _same_bytes(raw["message_b64"], msg)))
    if raw.get("message_sha256") is not None:
        digest = hashlib.sha256(msg).hexdigest()
        out.append(("message_sha256", None if key_bytes is None else raw["message_sha256"] == digest))
    if raw.get("public_key_b64") is not None:
        out.append(("public_key_b64", None if key_bytes is None else _same_bytes(raw["public_key_b64"], key_bytes)))
    return out


def _stray_outcome(value: object, payload: dict, name: str):
    """A member under a signed member's name must equal it; any other name
    has no twin. The hosted response's type is not a receipt type."""
    twin = _signed_twin(payload, name)
    if twin is None or (name == "type" and value == _HOSTED_RESPONSE_TYPE):
        return None
    return _same_json(value, twin)


def _unit_order(name: str) -> bytes:
    # Big-endian UTF-16 bytes sort as the code units do, in every engine.
    return name.encode("utf-16-be", "surrogatepass")


def _stray_outcomes(raw: dict, payload: dict, sig_obj: dict) -> list:
    """Top-level members outside the export shape,
    then extras inside the signature object."""
    top = sorted((k for k in raw if k not in _OUTER_MEMBERS and raw[k] is not None), key=_unit_order)
    out = [(k, _stray_outcome(raw[k], payload, k)) for k in top]
    extra = sorted((k for k in sig_obj if k not in _SIGNATURE_MEMBERS and sig_obj[k] is not None), key=_unit_order)
    return out + [(f"signature.{k}", _stray_outcome(sig_obj[k], payload, k)) for k in extra]


#: Fields a flat hash-mode receipt signs; every other member of one is unsigned.
_FLAT_SIGNED = (
    "v", "mode", "hash", "hash_algo", "metadata", "server_timestamp", "action_id",
    "agent_id", "org_id", "policy_digest", "policy_decision", "previous_receipt_hash",
)

#: A pre-#1700 receipt signs the same fields minus the previous link.
_FLAT_SIGNED_LEGACY = _FLAT_SIGNED[:-1]


def flat_signed_names(raw: dict) -> tuple:
    """The fields a flat receipt signs. A present link key is signed even
    null; without it the receipt rebuilds the 11 fields it signed."""
    if "previous_receipt_hash" in raw:
        return _FLAT_SIGNED
    return _FLAT_SIGNED_LEGACY

#: Framing a flat receipt's checks read: algorithm and anchors.
_FLAT_FRAMING = frozenset(("anchors", "algorithm"))


def _is_flat_hash_receipt(raw: dict) -> bool:
    """A flat hash-mode receipt: mode=hash, no payload
    object, a signature (the oracle's rule)."""
    if raw.get("mode") != "hash" or isinstance(raw.get("payload"), dict):
        return False
    return bool(raw.get("signature_b64") or raw.get("signature"))


def _flat_signature_used(raw: dict, name: str) -> bool:
    """The signature string that verified, or a string with its bytes."""
    source = "signature_b64" if raw.get("signature_b64") else "signature"
    if not isinstance(raw[name], str) or not isinstance(raw[source], str):
        return False
    decoded = _strict_b64(raw[name])
    return name == source or (decoded is not None and decoded == _strict_b64(raw[source]))


def _flat_member_bound(raw: dict, name: str, key_bytes, key_kid) -> bool:
    """Signed, framing, the verified signature, or the verifying kid."""
    if name in _FLAT_SIGNED or name in _FLAT_FRAMING:
        return True
    if name in ("signature_b64", "signature"):
        return _flat_signature_used(raw, name)
    return name == "key_id" and key_bytes is not None and isinstance(key_kid, str) and raw[name] == key_kid


def _flat_outer_note(raw: dict, key_bytes, key_kid) -> tuple:
    """A flat receipt signs its own fields, so
    it has no copies; list what it signs not."""
    names = sorted(
        (k for k in raw if raw[k] is not None and not _flat_member_bound(raw, k, key_bytes, key_kid)),
        key=_unit_order,
    )
    if not names:
        return "PASS", "no outer copies carried; the signed payload is the whole receipt"
    return "PASS", f"not signed, do not rely on: {', '.join(names)}"


def _outer_copy_outcomes(raw: dict, payload: dict, sig_obj: dict, msg: bytes, key_bytes) -> list:
    out = _text_copy_outcomes(raw, payload) + _time_copy_outcomes(raw, payload)
    out += _envelope_copy_outcomes(raw, sig_obj) + _byte_copy_outcomes(raw, msg, key_bytes)
    if "compliance_mode" in raw:
        out.append(("compliance_mode", raw["compliance_mode"] is True))
    return out + _stray_outcomes(raw, payload, sig_obj)


def _outer_copy_note(outcomes: list, raw: dict) -> tuple:
    """Fold per-member outcomes into the
    (result, note) of the outer_copies axis."""
    bad = [m for m, ok in outcomes if ok is False]
    unattested = [m for m, ok in outcomes if ok is None]
    unattested += [m for m in _OUTER_UNATTESTED if m in raw]
    if bad:
        named = ", ".join(f"{m} (signed: {_OUTER_SIGNED_LABEL.get(m, m.removeprefix('signature.'))})" for m in bad)
        return "FAIL", (
            f"outer copies disagree with the signed payload: {named}; "
            "rely on the signed payload, not on the copies"
        )
    agreed = [m for m, ok in outcomes if ok]
    parts = []
    if agreed:
        parts.append(f"outer copies agree with the signed payload: {', '.join(agreed)}")
    if unattested:
        parts.append(f"not signed, do not rely on: {', '.join(unattested)}")
    return "PASS", "; ".join(parts) or "no outer copies carried; the signed payload is the whole receipt"


def check_outer_copies(raw: dict, msg: bytes, key_bytes=None, key_kid=None):
    """(result, note) for the outer_copies axis. No signature covers an export's
    plain copies, so each is compared with its signed value: mismatch FAILs."""
    payload = raw.get("payload") if isinstance(raw, dict) else None
    if not isinstance(payload, dict):
        # A flat hash-mode receipt has no copies; list what it does not sign,
        # including a key_id other than the verifying key_kid.
        if isinstance(raw, dict) and _is_flat_hash_receipt(raw):
            return _flat_outer_note(raw, key_bytes, key_kid)
        return "PASS", "no signed payload object; no outer copies compared"
    sig_obj = _signature_object(normalise_envelope(raw), payload)
    return _outer_copy_note(_outer_copy_outcomes(raw, payload, sig_obj, msg, key_bytes), raw)


#: What this verifier does not check, carried in every result so silence never
#: reads as coverage. `condition` names the input enabling a check, else None.
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
        "reason": (
            "no live CRL or OCSP lookup offline; revocation status is asserted "
            "at trust-bundle issue time per root, so a later revocation is missed "
            "until the bundle is refreshed (see the revocation policy in "
            "docs/offline-verification.md)"
        ),
        "condition": (
            "a newer trust bundle (bundle_version/issued_at) or freshly re-pinned "
            "--tsa-key material narrows the assertion lag"
        ),
    },
    {
        "check": "tsa_trusted_list_status",
        "requirement": "anchor trust",
        "reason": (
            "no EU Trusted List status lookup as of the token's genTime; "
            "a trust bundle records status asserted at bundle-issue time only"
        ),
        "condition": None,
    },
    {
        "check": "policy_digest_resolution",
        "requirement": "verifier check on policy_digest",
        "reason": (
            "the referenced policy artefact is not fetched, rehashed, or otherwise "
            "evaluated; resolving it needs the Audit Pack manifest"
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


#: The NOT_CHECKED id for resolving the referenced policy artefact, which no
#: caller pin fetches or rehashes.
POLICY_DIGEST_RESOLUTION = next(
    row["check"] for row in NOT_CHECKED if row["check"].startswith(CUTOFF_AXES["policy_digest"])
)


def not_checked_declaration() -> list:
    """Return the non-coverage declaration as a fresh list of fresh dicts.

    Copied on every call so a caller mutating a result cannot edit the module
    constant and quietly narrow what every later result declares.
    """
    return [dict(entry) for entry in NOT_CHECKED]


#: The axis order run_structured reports on the normal path, structure first and
#: expiry last. The coverage block measures "not reached" against this sequence.
_FULL_AXIS_ORDER = (
    "structure", "nonce", "issuer_key", "issuer_bind", "key_status", "signature",
    "key_binding", "counterparty", "payload_digest", "outer_copies", "chain", "anchors",
    "skew", "expiry",
)


def coverage_declaration(axes: list) -> dict:
    """The coverage block every result carries beside its non-coverage list.
    stopped_at is None on a full run; an early return lists not_reached axes."""
    entries = [
        {
            "id": row["check"],
            "reason": "not_implemented",
            "status": "not_implemented",
            "requirement": row["requirement"],
            "condition": row["condition"],
        }
        for row in NOT_CHECKED
    ]
    names = [a["name"] for a in axes]
    if names and names[-1] == _FULL_AXIS_ORDER[-1]:
        return {"stopped_at": None, "checks_not_evaluated": entries}
    stopped_at = "structure"
    after = _FULL_AXIS_ORDER[_FULL_AXIS_ORDER.index(stopped_at) + 1 :]
    entries += [
        {"id": name, "reason": "not_reached", "status": "implemented"}
        for name in after
        if name not in names
    ]
    return {"stopped_at": stopped_at, "checks_not_evaluated": entries}


def check_payload_digest(payload: dict) -> tuple[str, str, str]:
    """Recompute payload_digest from the receipt's context. Absence PASSes:
    hash mode and redacted receipts carry no context, which is ordinary."""
    # Without this, a signer could pair a benign context with a digest that
    # commits to other content and every verifier would pass it.
    if not isinstance(payload, dict):
        return "PASS", "no signed payload; no digest to recompute", "evaluated"
    digest = payload.get("payload_digest")
    if digest is None:
        return "PASS", "receipt binds no payload_digest; nothing to recompute", "not_applicable"
    if not isinstance(digest, dict):
        return "FAIL", f"payload_digest is {type(digest).__name__}, not an object", "evaluated"
    claimed = digest.get("hash")
    if not isinstance(claimed, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed):
        return "FAIL", f"payload_digest.hash {claimed!r} is not 64 lowercase hex", "evaluated"
    claimed_size = digest.get("size")
    # Absence stays allowed; a present null is malformed, not a missing length.
    if "size" in digest and (
        not isinstance(claimed_size, int) or isinstance(claimed_size, bool) or claimed_size < 0
    ):
        return (
            "FAIL",
            f"payload_digest.size {claimed_size!r} is not a non-negative integer",
            "evaluated",
        )
    if payload.get("context") is None:
        # Null reads as absent: the hosted verifier returns payload: null for
        # redacted and hash-only receipts, and null is not a context to hash
        return (
            "PASS",
            "no context carried; payload_digest not recomputable here",
            "not_applicable",
        )
    if is_keyed_digest(payload):
        # A keyed digest (hash_algo outside sha256) is HMAC over the context
        # under holder key material none has; SHA-256 fakes a mismatch.
        return (
            "SKIPPED",
            "keyed payload_digest needs holder key material to recompute; "
            "recorded as unverifiable, never a mismatch",
            "evaluated",
        )

    try:
        encoded = canonical_json(payload["context"])
    except (TypeError, ValueError, RecursionError):
        return (
            "SKIPPED",
            "context is not canonicalisable, so payload_digest cannot be recomputed",
            "evaluated",
        )
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != claimed:
        return (
            "FAIL",
            f"payload_digest_mismatch: receipt binds {claimed[:16]}.., "
            f"its own context hashes to {actual[:16]}..",
            "evaluated",
        )
    if claimed_size is not None and claimed_size != len(encoded):
        return (
            "FAIL",
            f"payload_digest_mismatch: size claims {claimed_size}, "
            f"canonical context is {len(encoded)} bytes",
            "evaluated",
        )
    return (
        "PASS",
        f"payload_digest rederives from the carried context ({len(encoded)} bytes)",
        "evaluated",
    )


def _envelope_hash(envelope: dict) -> str:
    """Hash payload and the exact signature
    object, excluding all outer export members."""
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
        raise ValueError("originating payload is unavailable")
    signature = envelope.get("signature")
    if not isinstance(signature, dict) or any(
        not isinstance(signature.get(key), str) or not signature[key]
        for key in ("alg", "kid", "sig")
    ):
        raise ValueError("originating signature object is unavailable")
    projection = {"payload": envelope["payload"], "signature": signature}
    if _scan_shape(projection, max_depth=MAX_NESTING_DEPTH) is not None:
        raise ValueError("originating canonical bytes are unavailable")
    return base64.b64encode(hashlib.sha256(canonical_json(projection)).digest()).decode()


_COUNTERPARTY_KID_UNSET = object()


def counterparty_acknowledging_kid(envelope: dict):
    """Read an actual wire kid, without
    normalization's synthesized issuer fallback."""
    signature = envelope.get("signature")
    if not isinstance(signature, dict):
        signature = envelope.get("signature_envelope")
    return signature.get("kid") if isinstance(signature, dict) else None


def bound_counterparty_kid(kid, signing_kid, signing_issuer):
    """Require the advertised identifier to name
    the directory's selected signing key."""
    return kid if isinstance(kid, str) and kid in (signing_kid, signing_issuer) else None


def check_counterparty_binding(payload: dict, originator: dict | None = None, *, acknowledging_kid=_COUNTERPARTY_KID_UNSET):
    """Weigh a claimed cross-agent binding instead of trusting it unchecked.
    counterparty_binding is caller-supplied; an issuer can assert it alone."""
    if not isinstance(payload, dict):
        return "PASS", "no signed payload; no counterparty binding to check"
    cpb = payload.get("counterparty_binding")
    # Absence PASSes, the ordinary case. A claim this verifier cannot resolve
    # SKIPs, which blocks, so it never reads as corroborated; malformed FAILs.
    if "counterparty_binding" not in payload:
        return "PASS", "no counterparty binding; content is unilaterally asserted"
    if not isinstance(cpb, dict):
        return "FAIL", f"counterparty_binding is {type(cpb).__name__}, not an object"
    if "scope" not in cpb:
        return "SKIPPED", "legacy_scope: the signed binding omits its digest scope"
    if cpb["scope"] != "envelope_minus_anchors":
        return "SKIPPED", "unrecognised_scope: the signed digest scope is unsupported"
    profile_declared = (
        cpb.get("profile_version") is not None
        or "terms" in cpb
        or "terms_digest" in cpb
    )
    if profile_declared:
        try:
            from ..counterparty import BINDING_MEMBERS, PROFILE_VERSION, terms_digest_of
        except ImportError:
            # A copied standalone file has no package to read the registry from.
            return (
                "SKIPPED",
                "profile_unsupported: the standalone verifier cannot read the profile registry",
            )
        profile_version = cpb.get("profile_version")
        if profile_version is not None:
            if profile_version != PROFILE_VERSION:
                return "SKIPPED", "profile_unsupported: the signed profile version is unsupported"
            if set(cpb) - BINDING_MEMBERS:
                return "FAIL", "counterparty_binding carries a member this verifier does not implement"
        if "terms" in cpb or "terms_digest" in cpb:
            terms = cpb.get("terms")
            terms_digest = cpb.get("terms_digest")
            if not isinstance(terms, dict) or not isinstance(terms_digest, str):
                return "FAIL", "counterparty_binding terms and terms_digest must travel together"
            try:
                recomputed = terms_digest_of(terms)
            except ValueError:
                return "FAIL", "counterparty_binding terms are not canonically encodable"
            if recomputed != terms_digest:
                return "FAIL", "counterparty_terms_mismatch: terms_digest does not recompute over terms"

    receipt_ref = cpb.get("receipt_ref")
    envelope_hash = cpb.get("envelope_hash")
    if not isinstance(receipt_ref, str) or not receipt_ref:
        return "FAIL", "counterparty_binding.receipt_ref missing or not a string"
    if not isinstance(envelope_hash, str) or not envelope_hash:
        return "FAIL", "counterparty_binding.envelope_hash missing or not a string"
    try:
        normalized = envelope_hash.replace("-", "+").replace("_", "/")
        raw = base64.b64decode(normalized + "=" * (-len(normalized) % 4), validate=True)
    except Exception:
        return "FAIL", f"counterparty_binding.envelope_hash {envelope_hash!r} is not base64"
    if len(raw) != 32:
        return "FAIL", (
            f"counterparty_binding.envelope_hash decodes to {len(raw)} bytes, not a sha-256"
        )

    expect_ack_from = cpb.get("expect_ack_from")
    if expect_ack_from is not None and not isinstance(expect_ack_from, str):
        return "FAIL", "counterparty_binding.expect_ack_from is not a string"
    if cpb.get("transport_label") is not None and not isinstance(cpb["transport_label"], str):
        return "FAIL", "counterparty_binding.transport_label is not a string"

    if not isinstance(originator, dict):
        # Structurally sound but nothing corroborates it; blocking on purpose
        return "SKIPPED", (
            f"unresolved: counterparty binding claims {receipt_ref}; no originating receipt supplied, "
            "so the corroboration is unchecked"
        )

    try:
        actual = _envelope_hash(originator)
    except (TypeError, ValueError, RecursionError, UnicodeError):
        return "SKIPPED", "unresolved: exact originating canonical bytes are unavailable"
    if base64.b64decode(actual) != raw:
        return "FAIL", (
            f"counterparty_mismatch: binding commits {envelope_hash[:16]}.., "
            f"supplied originator hashes to {actual[:16]}.."
        )
    kid = payload.get("issuer_id") if acknowledging_kid is _COUNTERPARTY_KID_UNSET else acknowledging_kid
    if expect_ack_from is not None and kid != expect_ack_from:
        return "FAIL", (
            f"kid_mismatch: binding expects an acknowledgment from "
            f"{expect_ack_from}, this receipt names signature.kid {kid!r}"
        )
    return "PASS", f"counterparty binding rederives from the supplied {receipt_ref}"


def check_directory_substitution(payload: dict, jwks: dict, prior_jwks: dict | None):
    """Report directory substitution against retained evidence. Returns None
    with no retained directory or no contradiction, leaving the old path."""
    if prior_jwks is None or not isinstance(payload, dict):
        return None
    claimed = payload.get("key_thumbprint")
    if not isinstance(claimed, str) or not is_well_formed(claimed):
        return None
    prior_keys = prior_jwks.get("keys") if isinstance(prior_jwks, dict) else None
    if not isinstance(prior_keys, list):
        return None
    prior_hits = [
        k for k in prior_keys
        if isinstance(k, dict)
        and k.get("key_thumbprint") == claimed
        and isinstance(k.get("public_key"), str)
    ]
    if len(prior_hits) != 1:
        return None
    (prior_entry,) = prior_hits
    cur_keys = jwks.get("keys") if isinstance(jwks, dict) else None
    cur_keys = cur_keys if isinstance(cur_keys, list) else []
    if any(isinstance(k, dict) and k.get("public_key") == prior_entry.get("public_key")
           for k in cur_keys):
        return None
    kid = prior_entry.get("kid")
    agent = prior_entry.get("agent_id")
    if not any(isinstance(k, dict)
               and (k.get("kid") == kid or k.get("agent_id") == agent)
               and k.get("public_key") != prior_entry.get("public_key")
               for k in cur_keys):
        return None
    return ("SKIPPED",
            "directory_substitution: retained directory binds "
            f"{claimed} to kid {kid!r} agent {agent!r}; the current "
            "directory binds a different key there, so the binding "
            "cannot be recomputed")


def check_key_binding(payload: dict, alg, public_key):
    """Recompute the signed key_thumbprint from the resolved key. A swapped
    key cannot rederive it, a proven binding break that invalidates."""
    if not isinstance(payload, dict):
        return "PASS", "no signed payload; binding not checked"
    claimed = payload.get("key_thumbprint")
    # Legacy receipts bind no key; a skip would block every one, so PASS. An
    # unrecomputable binding SKIPs, which blocks, so it never reads verified.
    if claimed is None:
        return "PASS", "receipt binds no key_thumbprint; binding not checked"
    if not is_well_formed(claimed):
        return "FAIL", f"key_thumbprint {claimed!r} is not sha256:<64 lowercase hex>"
    if not public_key:
        return "SKIPPED", "no signing key resolved, so the bound key_thumbprint cannot be recomputed"
    if not isinstance(alg, str) or not is_akp_public_key(alg=alg, public_key=public_key):
        # A KMS-backed row publishes a PEM here, and a PEM is not the key
        # material an AKP `pub` carries, so the digest is not reconstructible
        return "SKIPPED", (
            "resolved key is not raw ML-DSA material, so the bound key_thumbprint "
            "cannot be recomputed"
        )
    actual = thumbprint_for_key(alg=alg, public_key=public_key)
    if actual == claimed:
        return "PASS", f"key_thumbprint rederives from the resolved {alg} key"
    return "FAIL", f"key_substituted: receipt binds {claimed}, resolved key computes {actual}"


    # Axis-only expiry flag; mirrors signature_expired, never folds verdict.
def check_expiry(payload: dict, *, now: datetime | None = None):
    if not isinstance(payload, dict) or "expires_at" not in payload:
        return "PASS", "receipt declares no expiry"
    raw = payload["expires_at"]
    ts = _parse_stamp(raw)
    if ts is None:
        return "FAIL", f"unreadable expires_at {raw!r}; refused rather than read as no expiry"
    delta = (ts - (now or datetime.now(timezone.utc))).total_seconds()
    if delta < 0:
        return "FAIL", f"expires_at {raw} lapsed {-delta:.0f}s ago"
    return "PASS", f"expires_at {raw} is {delta:.0f}s ahead"


def _is_well_formed_beacon_ref(beacon) -> bool:
    """Shape of an OPTIONAL beacon_ref member (draft 5.5, paragraph 11).

    Source/chain/round/signature/observed_at; mirrors the reference
    platform's structural rule without authenticating the BLS signature.
    """
    if not isinstance(beacon, dict):
        return False
    if beacon.get("source") != "drand":
        return False
    chain = beacon.get("chain")
    if not isinstance(chain, str) or len(chain) != 64:
        return False
    if any(c not in "0123456789abcdef" for c in chain):
        return False
    round_number = beacon.get("round")
    if isinstance(round_number, bool) or not isinstance(round_number, int):
        return False
    if round_number < 1:
        return False
    signature = beacon.get("signature")
    if not isinstance(signature, str) or len(signature) != 96:
        return False
    if any(c not in "0123456789abcdef" for c in signature):
        return False
    return isinstance(beacon.get("observed_at"), str)


#: Optional extensions a selected policy may require this tool to evaluate.
#: Anything else the policy names is unsupported here and fails closed.
SUPPORTED_REQUIRED_EXTENSIONS = frozenset({"beacon_ref"})


def check_required_extensions(payload: dict, required) -> tuple[str, str]:
    """Evaluate caller-required optional extensions (criterion 574-D3). SKIPPED
    if absent or unsupported, so it never reads verified; FAIL if malformed."""
    if not isinstance(payload, dict):
        return "SKIPPED", "no signed payload; required extensions cannot be evaluated"
    for name in required or []:
        if name not in SUPPORTED_REQUIRED_EXTENSIONS:
            return "SKIPPED", f"required extension {name!r} is not supported by this verifier"
        if name == "beacon_ref":
            beacon = payload.get("beacon_ref")
            if beacon is None:
                return "SKIPPED", "required beacon_ref is absent from the signed payload"
            if not _is_well_formed_beacon_ref(beacon):
                return "FAIL", "required beacon_ref is malformed against the 5.5 shape"
    return "PASS", "every required extension is carried and well-formed"


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
    """Replay-candidate axis over the draft 5.7 nonce: a DIFFERENT receipt
    reusing an (issuer_id, nonce) pair fails; an identical one does not."""
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
        # Identity for the seen-nonce index only; never recomputed externally.
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
    # Non-string type is malformed, never unregistered (criterion 524)
    if not isinstance(rt, str):
        return "FAIL", f"malformed type {rt!r}: not a string"
    # Split per draft anchor type: gap, reserved-unknown, foreign
    if rt not in ALLOWED_TYPES:
        if rt in REGISTRY_NAMESPACES:
            return "SKIPPED", (
                f"type {rt!r} is registered but this verifier does not implement it"
            )
        if rt.startswith("protectmcp:"):
            return "FAIL", (
                f"unregistered type {rt!r}: claims the reserved protectmcp namespace"
            )
        return "SKIPPED", (
            f"type {rt!r} outside the known namespace; "
            "profile membership unverifiable, reported not failed"
        )
    # Draft 5.3: absence never reads as v1; only the recognised set verifies.
    # An unrecognised v is unverifiable, not failed; the shape is never guessed.
    if "v" not in payload:
        return "SKIPPED", (
            "no v member: not a Compliance Receipt of this document; "
            "absence is not version 1"
        )
    v = payload["v"]
    if not is_recognised_wire_version(v):
        return "SKIPPED", (
            f"unsupported wire version {v!r}: "
            "unverifiable under this document, reported not failed"
        )
    if "mode" not in payload:
        return "SKIPPED", (
            "no mode member: not a Compliance Receipt of this document; "
            "absence selects no mode"
        )
    mode = payload["mode"]
    if not is_recognised_mode(mode):
        return "SKIPPED", (
            f"unsupported mode {mode!r}: "
            "unverifiable under this document, reported not failed"
        )
    ce_res, ce_note = check_controls_evaluated(payload)
    if ce_res == "FAIL":
        return "FAIL", ce_note
    return "PASS", f"required fields present; type {rt}"


def is_current_profile_version(value) -> bool:
    """True for numeric v=1 (1.0 selects too);
    booleans, v2, missing never select."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == 1
    return isinstance(value, float) and value == 1.0


def is_recognised_wire_version(value) -> bool:
    """True for an integer v in RECOGNISED_WIRE_VERSIONS;
    booleans and floats never select."""
    return type(value) is int and value in RECOGNISED_WIRE_VERSIONS


def is_recognised_mode(value) -> bool:
    """True for a string mode in RECOGNISED_MODES; anything else never selects.
    """
    return type(value) is str and value in RECOGNISED_MODES


def profile_range_note(obj) -> str | None:
    """Note when ``obj`` carries a number beyond +/-(2**53 - 1), else None."""
    stack = [(obj, "$", False)]
    active: set[int] = set()
    done: dict[int, object] = {}
    while stack:
        node, path, exiting = stack.pop()
        if isinstance(node, bool):
            continue
        if isinstance(node, float):
            # A parsed float came from a float lexeme, so 500.0 and -0.0 bind
            # digits the profile never checked as an integer, lexeme or not.
            lexeme = node.lexeme if isinstance(node, _RawFloat) else repr(node)
            if (
                not re.fullmatch(r"-?\d+", lexeme)
                and node.is_integer()
                and -MAX_PROFILE_INTEGER <= node <= MAX_PROFILE_INTEGER
            ):
                return (
                    f"Asqav profile range +/-(2**53 - 1) excludes non-integer "
                    f"lexeme {lexeme!r} at {path}"
                )
            # Fall through: out-of-range magnitudes still fail below.
        if isinstance(node, int):
            if not -MAX_PROFILE_INTEGER <= node <= MAX_PROFILE_INTEGER:
                # str() itself raises past the interpreter digit cap, so an
                # unformattable magnitude is refused by digit count instead.
                if _int_exceeds_str_limit(node):
                    digits = _int_digit_count(node)
                    return (
                        f"Asqav profile range +/-(2**53 - 1) excludes "
                        f"a {digits}-digit integer at {path}"
                    )
                return (
                    f"Asqav profile range +/-(2**53 - 1) excludes integer "
                    f"{node} at {path}"
                )
            continue
        if isinstance(node, float) and node.is_integer():
            if not -MAX_PROFILE_INTEGER <= node <= MAX_PROFILE_INTEGER:
                return (
                    f"Asqav profile range +/-(2**53 - 1) excludes number "
                    f"{node!r} at {path}"
                )
            continue
        if isinstance(node, (dict, list, tuple)):
            if exiting:
                # The dict holds each skipped container alive, so a freed id
                # can never be recycled onto an unchecked container (D2)
                active.discard(id(node))
                done[id(node)] = node
                continue
            # Only a container met while expanding is a cycle; a checked one is
            # shared and clean (the walk stops at the first violation): skip.
            if id(node) in done:
                continue
            if id(node) in active:
                return (
                    "Asqav profile range +/-(2**53 - 1) unverifiable: "
                    f"cyclic reference at {path}"
                )
            active.add(id(node))
            stack.append((node, path, True))
            if isinstance(node, dict):
                stack.extend((v, f"{path}.{k}", False) for k, v in node.items())
            else:
                stack.extend(
                    (v, f"{path}[{i}]", False) for i, v in enumerate(node)
                )
    return None


def _counterparty_commitment_reached(payload, counterparty) -> bool:
    """True when the counterparty path reaches its envelope commitment."""
    cpb = payload.get("counterparty_binding")
    # Only this scope hashes the origin; another one never reads it.
    if not isinstance(cpb, dict) or cpb.get("scope") != "envelope_minus_anchors":
        return False
    receipt_ref = cpb.get("receipt_ref")
    if not isinstance(receipt_ref, str) or not receipt_ref:
        return False
    digest = cpb.get("envelope_hash")
    if not isinstance(digest, str) or not digest:
        return False
    try:
        raw = base64.b64decode(digest, validate=True)
    except Exception:
        return False
    if len(raw) != 32:
        return False
    ack = cpb.get("expect_ack_from")
    if ack is not None and not isinstance(ack, str):
        return False
    return isinstance(counterparty, dict)


    # Refusal note when current-profile digest inputs break the range, or None.
def _profile_range_refusal(payload, sig_obj, predecessor_payload, counterparty, anchors):
    if not is_current_profile_version(payload.get("v")):
        return None
    note = profile_range_note(payload)
    if note is not None:
        return note
    if isinstance(anchors, list) and anchors and isinstance(sig_obj, dict):
        note = profile_range_note(sig_obj)
        if note is not None:
            return note
    # check_chain canonicalizes a supplied predecessor for a non-genesis link
    # the precheck walks it the same way, with no member or registry test here.
    if (
        predecessor_payload is not None
        and payload.get("previousReceiptHash") != FIRST_RECEIPT_SEED
    ):
        note = profile_range_note(predecessor_payload)
        if note is not None:
            return note
    if _counterparty_commitment_reached(payload, counterparty):
        note = profile_range_note(counterparty)
        if note is not None:
            return note
    return None


def _agent_fallback_signature(
    jwks, payload, envelope, msg, sig, alg, jwks_alg, pk, sig_res, eff
):
    """Adopt a verifying agent key, else note exhaustion; returns updated state.
    """
    # The agent's own key signs though kid names the issuer; look it up.
    # agent_id is forgeable, so accept only a key published under that issuer.
    agent_id = payload.get("agent_id") or envelope.get("agent_id")
    org_bind = payload.get("org_id") or envelope.get("org_id")
    entry_a, sig_res_a, exhausted = _select_agent_bound_key(
        jwks, agent_id, payload.get("issuer_id"), org_bind, msg, sig,
        alg or jwks_alg,
    )
    pk_a, alg_a = None, None
    agent_note = ""
    if entry_a is not None:
        pk_a, alg_a = _b64decode(entry_a["public_key"]), entry_a.get("alg")
        if sig_res_a is not None and sig_res_a[0] == "PASS":
            sig_res = sig_res_a
            eff = (
                entry_a.get("status"),
                entry_a.get("kid"),
                key_issuer_of(entry_a),
                revoked_at_of(entry_a),
                pk_a,
                alg_a,
                entry_a.get("revocation_reason"),
            )
        elif exhausted:
            agent_note = "; no key published for this agent verified"
    if sig_res[0] != "PASS":
        cands = [(pk, alg or jwks_alg)]
        if pk_a is not None:
            cands.append((pk_a, alg_a or alg or jwks_alg))
        sig_res = _pre_cutover_diagnostic(payload, sig, cands, sig_res)
    return sig_res, eff, agent_note


def _signature_object(envelope: dict, payload: dict) -> dict:
    """The {alg, kid, sig} object of a normalised
    envelope, derived from a flat string."""
    sig_obj = envelope.get("signature", {})
    if isinstance(sig_obj, str):  # flat-string signature, derive the object
        return {
            "alg": envelope.get("algorithm", "ML-DSA-65"),
            "kid": payload.get("issuer_id", ""),
            "sig": sig_obj,
        }
    # A non-object signature (list, number, ...) carries no kid/sig; key
    # resolution then FAILs cleanly instead of .get on a non-dict.
    return sig_obj if isinstance(sig_obj, dict) else {}


def check_rotation_succession(entry, eff_pk, prior_jwks):
    """Report unevidenced rotation against retained evidence (628 T-b). None if
    unverified, unretained or still published; else SKIPPED, which blocks."""
    if prior_jwks is None or eff_pk is None or not isinstance(entry, dict):
        return None
    prior_keys = prior_jwks.get("keys") if isinstance(prior_jwks, dict) else None
    if not isinstance(prior_keys, list):
        return None
    kid = entry.get("kid")
    agent = entry.get("agent_id")
    retained = [
        k for k in prior_keys
        if isinstance(k, dict)
        and (k.get("kid") == kid or k.get("agent_id") == agent)
        and isinstance(k.get("public_key"), str)
    ]
    if not retained:
        return None
    eff_b64 = base64.b64encode(eff_pk).decode()
    if any(k.get("public_key") == eff_b64 for k in retained):
        return None
    if any(k.get("status") == "revoked" for k in retained):
        case = "removed_old_key"
        why = "the retained key is revoked and cannot authorize succession"
    else:
        case = "forged_transition"
        why = "with no transition authorization from the retained key"
    return ("SKIPPED",
            f"{case}: verifying key differs from every retained "
            f"key for kid {kid!r} agent {agent!r} {why}, so succession "
            "cannot be established")


def _key_and_signature_rows(
    jwks, kid, alg, sig_obj, payload, envelope, msg, anchor_eval, acknowledging_kid,
    prior_jwks=None, status_cutoff=None,
):
    """Rows for issuer_key, issuer_bind, key_status and signature, plus the
    verifying key. Returns ``(rows, eff_pk, eff_alg, acknowledging_kid)``."""
    entry = _signing_key_entry(jwks, kid, payload, envelope)
    pk, status, jwks_alg = _entry_material(entry)
    if pk is None:
        rows = [
            ("issuer_key", "FAIL", f"kid {kid!r} not in jwks directory"),
            ("signature", "SKIPPED", "no issuer key to verify against"),
        ]
        return rows, None, None, acknowledging_kid
    _raw_sig = sig_obj.get("sig", "")
    carrier = _sig_carrier(_raw_sig)
    if carrier is None:
        sig = b""
        sig_res = ("FAIL", "signature string is not decodable base64")
    else:
        decoded = _strict_b64(_raw_sig)
        sig = decoded if decoded is not None else b""
        sig_res = verify_signature(pk, msg, sig, alg or jwks_alg)
    eff_kid = kid if kid and kid in (entry.get("issuer_id"), entry.get("kid")) else entry.get("kid")
    eff = (status, eff_kid, key_issuer_of(entry), revoked_at_of(entry), pk, jwks_alg, entry.get("revocation_reason") if isinstance(entry, dict) else None)
    agent_note = ""
    if sig_res[0] != "PASS" and carrier is not None:
        sig_res, eff, agent_note = _agent_fallback_signature(
            jwks, payload, envelope, msg, sig, alg, jwks_alg, pk, sig_res, eff
        )
    if sig_res[0] == "PASS" and carrier == "legacy":
        sig_res = (
            "PASS",
            sig_res[1] + "; accepted under legacy encoding policy: "
            + LEGACY_ENCODING_POLICY,
        )
    eff_status, eff_kid, eff_issuer, eff_revoked_at, eff_pk, eff_alg, eff_reason = eff
    # Only a cryptographically verified anchor at or before revoked_at proves
    # pre-revocation timing; presence alone never upgrades a revoked key.
    trusted_anchor = _has_trusted_pre_revocation_anchor(anchor_eval.trusted_times, eff_revoked_at)
    rows = [
        ("issuer_key", "PASS", f"resolved signing key {eff_kid} (status={eff_status}){agent_note}"),
        # kid picks the key and the attacker picks the kid, so bind the key that
        # actually verified back to the issuer the receipt claims.
        ("issuer_bind", *check_issuer_binding(eff_issuer, payload.get("issuer_id"))),
        ("key_status", *check_key_status_at(status_cutoff, eff_status, payload.get("issued_at", ""), eff_revoked_at, trusted_anchor, eff_reason)),
        ("signature", *sig_res),
    ]
    # Succession replaces only the resolution report, and only when the
    # signature verified: an unverified key has no succession to evaluate.
    _succession = check_rotation_succession(entry, _verifying_key(rows, eff_pk), prior_jwks)
    if _succession is not None:
        rows = [("issuer_key", *_succession)] + rows[1:]
    return rows, eff_pk, eff_alg, bound_counterparty_kid(acknowledging_kid, eff_kid, eff_issuer)


def _verifying_key(rows, eff_pk):
    """The key bytes the signature row verified
    under, None when it did not PASS."""
    passed = any(name == "signature" and res == "PASS" for name, res, _ in rows)
    return eff_pk if passed else None


def run(
    envelope: dict,
    jwks: dict,
    predecessor_payload: dict | None,
    seen_nonces: set | None = None,
    *,
    trusted_tsa_keys=None,
    bitcoin_headers=None,
    counterparty: dict | None = None,
    required_extensions=None,
    trust_bundle: dict | None = None,
    policy_digest: str | None = None,
    at_time=None,
    status_cutoff=None,
) -> int:
    if not isinstance(envelope, dict):
        print("Asqav receipt verification")
        print(
            f"  [FAIL] input       expected a JSON object receipt, got "
            f"{type(envelope).__name__}"
        )
        print("\n  => unverified (failure_class: unverifiable; no receipt object to verify)")
        return 2
    acknowledging_kid = counterparty_acknowledging_kid(envelope)
    raw_entry = envelope
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
    sig_obj = _signature_object(envelope, payload)
    kid = sig_obj.get("kid", "")
    alg = sig_obj.get("alg", "ML-DSA-65")
    refusal = _profile_range_refusal(
        payload, sig_obj, predecessor_payload, counterparty, envelope.get("anchors")
    )
    if refusal is not None:
        print("Asqav receipt verification")
        print(f"  [FAIL] input       {refusal}")
        print("\n  => unverified (failure_class: unverifiable; never reported verified)")
        return 2
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
        envelope, trusted_tsa_keys=trusted_tsa_keys, bitcoin_headers=bitcoin_headers,
        trust_bundle=trust_bundle,
    )

    rows, eff_pk, eff_alg, acknowledging_kid = _key_and_signature_rows(
        jwks, kid, alg, sig_obj, payload, envelope, msg, anchor_eval, acknowledging_kid,
        status_cutoff=status_cutoff,
    )
    results.extend(rows)

    # Outside the else on purpose: a receipt binding no thumbprint still reports
    # the axis, so the report says the binding was not checked, not silent.
    results.append(("key_binding", *check_key_binding(payload, eff_alg, eff_pk)))
    results.append(("counterparty", *check_counterparty_binding(payload, counterparty, acknowledging_kid=acknowledging_kid)))
    results.append(("payload_digest", *check_payload_digest(payload)[:2]))
    if policy_digest is not None:
        results.append((CUTOFF_AXES["policy_digest"], *check_policy_digest(payload, policy_digest)))
    results.append(("outer_copies", *check_outer_copies(raw_entry, msg, _verifying_key(rows, eff_pk))))
    results.append(("chain", *check_chain(payload, predecessor_payload)))
    results.append(("anchors", anchor_eval.result, anchor_eval.note))
    results.extend(check_at_time(payload, at_time))
    if required_extensions:
        results.append(("required_extensions", *check_required_extensions(payload, required_extensions)))

    print("Asqav receipt verification")
    print(f"  canonical bytes: sha256:{hashlib.sha256(msg).hexdigest()}")
    print(f"  signature.kid:   {kid}")
    # The algorithm is per-receipt, verbatim from the signed envelope: ML-DSA-65
    # for cloud-issued receipts, Ed25519/ES256 for locally signed ones.
    print(f"  signature.alg:   {alg}")
    for name, res, note in results:
        mark = {"PASS": "ok", "FAIL": "FAIL", "SKIPPED": "skip"}[res]
        print(f"  [{mark:>4}] {name:<11} {note}")

    # Expiry is its own axis and never folds the verdict.
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
def _struct_axis(name: str, result: str, note: str, applicability: str | None = None) -> dict:
    axis = {
        "name": name,
        "result": result,
        "note": note,
        "failure_class": _axis_failure_class(name, result, note),
    }
    if applicability is not None:
        axis["applicability"] = applicability
    return axis


#: "anchors": null is the malformed spelling of no anchors (absent or [] ok).
#: Verbatim in all three engines; the cross-language anchor-shape tests pin it.
_ANCHORS_NULL_NOTE = (
    "anchors is null: malformed; absent or [] is the conformant spelling of no anchors"
)


def _holds_unsafe_integer(*inputs) -> bool:
    """True when any input holds an int outside +/-2**53 at any depth.

    A parsed dict must get the refusal raw-text ingest gives the same file.
    """
    stack = list(inputs)
    seen: set = set()
    while stack:
        node = stack.pop()
        if isinstance(node, (dict, list, tuple)):
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.extend(node.values() if isinstance(node, dict) else node)
        elif isinstance(node, int) and not isinstance(node, bool):
            if not -MAX_CANONICAL_INTEGER <= node <= MAX_CANONICAL_INTEGER:
                return True
    return False


def _predecessor_payload(predecessor):
    """The payload of a whole predecessor envelope, else the value as given."""
    if isinstance(predecessor, dict) and isinstance(predecessor.get("payload"), dict):
        return predecessor["payload"]
    return predecessor


#: Input-axis note for a dict holding an integer raw-text ingest refuses.
_UNSAFE_INTEGER_INPUT_NOTE = (
    "integer outside the canonical integer range +/-2**53; serialise it as a JSON "
    "string or an integer-rational pair"
)


def _input_refusal(envelope) -> dict | None:
    """The axis refusing a receipt before any check runs, else None."""
    if not isinstance(envelope, dict):
        note = f"expected a JSON object receipt, got {type(envelope).__name__}"
        return _struct_axis("input", "FAIL", note)
    # A wire "anchors": null is the malformed third spelling of "no anchors"
    # (absent and [] are the conformant two); normalise_envelope would hide it.
    if "anchors" in envelope and envelope["anchors"] is None:
        # Only payload mode spells "no anchors"; others hit the payload gate.
        payload = envelope.get("payload", envelope)
        if not isinstance(payload, dict):
            return None
        # An unrecognised version reports first, so each engine states one fact.
        if not is_recognised_wire_version(payload.get("v")):
            return None
        return _struct_axis("structure", "FAIL", _ANCHORS_NULL_NOTE)
    return None


def _unsafe_integer_refusal(*inputs) -> str | None:
    # The profile check runs first and names a signed member.
    return _UNSAFE_INTEGER_INPUT_NOTE if _holds_unsafe_integer(*inputs) else None


def run_structured(
    envelope: dict,
    jwks: dict,
    predecessor_payload: dict | None = None,
    *,
    trusted_tsa_keys=None,
    bitcoin_headers=None,
    counterparty: dict | None = None,
    seen_nonces: set | None = None,
    required_extensions=None,
    trust_bundle: dict | None = None,
    trust_bundle_source: str | None = None,
    prior_jwks: dict | None = None,
    policy_digest: str | None = None,
    at_time=None,
    status_cutoff=None,
) -> dict:
    """Verify a receipt offline and return a structured result dict, as
    ``run()``. ``predecessor_payload`` may be the whole predecessor envelope."""
    _tb_identity = trust_bundle_identity(trust_bundle, trust_bundle_source)
    refusal = _input_refusal(envelope)
    whole_predecessor = predecessor_payload
    predecessor_payload = _predecessor_payload(predecessor_payload)
    if refusal is not None:
        axes = [refusal]
        return {
            "not_checked": not_checked_declaration(),
            "coverage": coverage_declaration(axes),
            "verdict": VERDICT_UNVERIFIED,
            "failure_class": FAILURE_UNVERIFIABLE,
            "axes": axes,
            "canonical_sha256": None,
            "kid": None,
            "alg": None,
            "registry_edition": REGISTRY_EDITION,
            "registry_version": REGISTRY_VERSION,
            "trust_bundle": _tb_identity,
        }
    acknowledging_kid = counterparty_acknowledging_kid(envelope)
    raw_entry = envelope
    envelope = normalise_envelope(envelope)
    payload = envelope.get("payload", envelope)
    if not isinstance(payload, dict):
        axes = [
            _struct_axis(
                "payload",
                "FAIL",
                "receipt payload not available from this surface "
                f"(got {_describe_value(payload)} instead of an object). "
                "Verify with a saved receipt instead.",
            )
        ]
        return {
            "not_checked": not_checked_declaration(),
            "coverage": coverage_declaration(axes),
            "verdict": VERDICT_UNVERIFIED,
            "failure_class": FAILURE_UNVERIFIABLE,
            "axes": axes,
            "canonical_sha256": None,
            "kid": None,
            "alg": None,
            "registry_edition": REGISTRY_EDITION,
            "registry_version": REGISTRY_VERSION,
            "trust_bundle": _tb_identity,
        }
    shape = _scan_shape(envelope, max_depth=MAX_NESTING_DEPTH)
    if shape is None:
        shape = _scan_shape(predecessor_payload, max_depth=MAX_NESTING_DEPTH)
    if shape is not None:
        axes = [_struct_axis("input", "FAIL", _SHAPE_MESSAGES[shape])]
        return {
            "not_checked": not_checked_declaration(),
            "coverage": coverage_declaration(axes),
            "verdict": VERDICT_UNVERIFIED,
            "failure_class": FAILURE_UNVERIFIABLE,
            "axes": axes,
            "canonical_sha256": None,
            "kid": None,
            "alg": None,
            "registry_edition": REGISTRY_EDITION,
            "registry_version": REGISTRY_VERSION,
            "trust_bundle": _tb_identity,
        }

    sig_obj = _signature_object(envelope, payload)
    kid = sig_obj.get("kid", "")
    alg = sig_obj.get("alg", "ML-DSA-65")
    refusal = _profile_range_refusal(
        payload, sig_obj, predecessor_payload, counterparty, envelope.get("anchors")
    ) or _unsafe_integer_refusal(raw_entry, jwks, whole_predecessor, counterparty)
    if refusal is not None:
        axes = [_struct_axis("input", "FAIL", refusal)]
        return {
            "not_checked": not_checked_declaration(),
            "coverage": coverage_declaration(axes),
            "verdict": VERDICT_UNVERIFIED,
            "failure_class": FAILURE_UNVERIFIABLE,
            "axes": axes,
            "canonical_sha256": None,
            "kid": None,
            "alg": None,
            "registry_edition": REGISTRY_EDITION,
            "registry_version": REGISTRY_VERSION,
            "trust_bundle": _tb_identity,
        }
    try:
        msg = canonical_json(payload)
    except RecursionError:
        # Defense in depth; the shape gate above should have caught this.
        axes = [_struct_axis("input", "FAIL", _SHAPE_MESSAGES["too_deep"])]
        return {
            "not_checked": not_checked_declaration(),
            "coverage": coverage_declaration(axes),
            "verdict": VERDICT_UNVERIFIED,
            "failure_class": FAILURE_UNVERIFIABLE,
            "axes": axes,
            "canonical_sha256": None,
            "kid": None,
            "alg": None,
            "registry_edition": REGISTRY_EDITION,
            "registry_version": REGISTRY_VERSION,
            "trust_bundle": _tb_identity,
        }

    axes: list[dict] = []
    axes.append(_struct_axis("structure", *check_structure(payload)))
    axes.append(_struct_axis("nonce", *check_nonce(payload, seen_nonces)))
    # Evaluated once, up front: the anchors axis reports it, and the key_status
    # axis weighs its cryptographically-proven timing against any revoked_at.
    anchor_eval = evaluate_anchors(
        envelope, trusted_tsa_keys=trusted_tsa_keys, bitcoin_headers=bitcoin_headers,
        trust_bundle=trust_bundle,
    )

    rows, eff_pk, eff_alg, acknowledging_kid = _key_and_signature_rows(
        jwks, kid, alg, sig_obj, payload, envelope, msg, anchor_eval, acknowledging_kid,
        prior_jwks=prior_jwks, status_cutoff=status_cutoff,
    )
    axes.extend(_struct_axis(*row) for row in rows)

    # Outside the else on purpose: a receipt binding no thumbprint still reports
    # the axis, so the report says the binding was not checked, not silent.
    _kb = check_directory_substitution(payload, jwks, prior_jwks)
    if _kb is None:
        _kb = check_key_binding(payload, eff_alg, eff_pk)
    axes.append(_struct_axis("key_binding", *_kb))
    axes.append(
        _struct_axis("counterparty", *check_counterparty_binding(payload, counterparty, acknowledging_kid=acknowledging_kid))
    )
    pd_res, pd_note, pd_applicability = check_payload_digest(payload)
    axes.append(_struct_axis("payload_digest", pd_res, pd_note, pd_applicability))
    if policy_digest is not None:
        axes.append(_struct_axis(CUTOFF_AXES["policy_digest"], *check_policy_digest(payload, policy_digest)))
    axes.append(_struct_axis("outer_copies", *check_outer_copies(raw_entry, msg, _verifying_key(rows, eff_pk))))
    axes.append(_struct_axis("chain", *check_chain(payload, predecessor_payload)))
    axes.append(_struct_axis("anchors", anchor_eval.result, anchor_eval.note))
    axes.extend(_struct_axis(*row) for row in check_at_time(payload, at_time))
    if required_extensions:
        axes.append(_struct_axis("required_extensions", *check_required_extensions(payload, required_extensions)))

    # Expiry is its own axis and never folds the verdict.
    verdict, failure_class = _fold_verdict(
        [(a["name"], a["result"], a["note"]) for a in axes],
        keyed=is_keyed_digest(payload),
    )

    return {
        "not_checked": not_checked_declaration(),
        "coverage": coverage_declaration(axes),
        "verdict": verdict,
        "failure_class": failure_class,
        "axes": axes,
        "canonical_sha256": hashlib.sha256(msg).hexdigest(),
        "kid": kid,
        "alg": alg,
        "registry_edition": REGISTRY_EDITION,
        "registry_version": REGISTRY_VERSION,
        "trust_bundle": _tb_identity,
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
        keys.append(tsa_key_material(blob))
    return keys


def tsa_key_material(blob: bytes) -> bytes:
    """One TSA key or certificate as the anchor
    check takes it: PEM kept, base64 decoded."""
    text = blob.strip()
    if text.startswith(b"-----BEGIN"):
        return text + b"\n"
    try:
        return base64.b64decode(text, validate=True)
    except Exception:
        # Raw binary passes through untouched (574-D3): stripping a binary
        # key with whitespace edge bytes would shorten it and break verifies.
        return bytes(blob)


#: Members a versioned TSA trust bundle must carry; the digest covers every
#: top-level member, so the file self-verifies: an edit skipping it fails below.
TRUST_BUNDLE_REQUIRED_KEYS = (
    "bundle_version",
    "issued_at",
    "digest",
    "digest_algorithm",
    "roots",
)


def load_trust_bundle(path: str) -> dict:
    """Read a versioned TSA trust bundle and reject a digest mismatch.

    Trust material comes from the caller, never from the envelope. A bundle
    whose recorded digest disagrees with its contents raises instead of
    verifying: the anchors axis must report it, never pass on it.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise VerifierInputError(f"{path}: {exc.strerror or exc}") from exc
    try:
        bundle = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise VerifierInputError(f"{path}: not a JSON trust bundle") from exc
    if not isinstance(bundle, dict) or any(k not in bundle for k in TRUST_BUNDLE_REQUIRED_KEYS):
        raise VerifierInputError(f"{path}: trust bundle misses a required member")
    ok, _ = verify_trust_bundle(bundle)
    if not ok:
        raise VerifierInputError(f"{path}: trust bundle digest mismatch")
    return bundle


def verify_trust_bundle(bundle: dict) -> tuple:
    """Self-digest check of a trust bundle dict.

    Returns (ok, recorded_digest). The digest is SHA-256 over the JCS
    canonical bytes of every top-level member except `digest` itself, so a
    caller holding only the dict can re-derive it without the file.
    """
    if not isinstance(bundle, dict):
        return False, ""
    recorded = bundle.get("digest")
    if not isinstance(recorded, str) or not recorded:
        return False, ""
    try:
        body = {k: v for k, v in bundle.items() if k != "digest"}
        computed = hashlib.sha256(canonical_json(body)).hexdigest()
    except (TypeError, ValueError):
        return False, recorded
    return (computed == recorded), recorded


def trust_bundle_identity(bundle: dict | None, source: str | None) -> dict | None:
    """Identity block for structured results; None when no bundle was supplied.
    """
    if bundle is None:
        return None
    ok, recorded = verify_trust_bundle(bundle)
    return {
        "version": bundle.get("bundle_version"),
        "digest": recorded,
        "source": source,
        "valid": ok,
    }


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
        "--trust-bundle",
        metavar="FILE",
        help="versioned TSA trust bundle JSON (verifier/trust/tsa-trust-bundle.json "
        "shape: bundle_version, issued_at, digest, digest_algorithm, roots with "
        "subject/sha256_der/not_after); the bundle self-digest is verified and "
        "a mismatch makes the anchors axis unverifiable. The bundle identity "
        "travels on every structured result; --tsa-key keeps working without one",
    )
    p.add_argument(
        "--bitcoin-headers",
        metavar="FILE",
        help="JSON map of bitcoin block height to {\"merkle_root\": <display hex>, "
        "\"time\": <ISO-8601>} for the OpenTimestamps block check; without it the "
        "block portion reports unverifiable",
    )
    p.add_argument(
        "--counterparty",
        metavar="FILE",
        help="complete originating signing envelope JSON for the counterparty binding check",
    )
    p.add_argument(
        "--require-extension",
        action="append",
        default=[],
        metavar="NAME",
        help="optional extension the selected policy requires (repeatable; "
        "supported: beacon_ref). A required member that is absent or "
        "unsupported reports unverifiable; a malformed one fails invalid",
    )
    p.add_argument(
        "--policy-digest",
        metavar="DIGEST",
        help="caller-pinned policy digest compared with the signed payload's "
        "policy_digest; a mismatch reports invalid, a receipt carrying none "
        "without one reports unverifiable",
    )
    p.add_argument(
        "--at-time",
        metavar="ISO8601",
        help="instant that replaces the wall clock in the skew and expiry checks; "
        "a value that is not ISO 8601 is refused",
    )
    p.add_argument(
        "--status-cutoff",
        metavar="ISO8601",
        help="instant the signing key status is judged at; a revocation after it "
        "is ignored and a value that is not ISO 8601 is refused",
    )
    p.add_argument("--offline", action="store_true", help="never reach the network")
    args = p.parse_args()

    for flag, value in (("--at-time", args.at_time), ("--status-cutoff", args.status_cutoff)):
        if value is not None and _parse_stamp(value) is None:
            p.error(f"{flag} {value!r} is not an ISO-8601 instant")

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
        counterparty = _load(args.counterparty) if args.counterparty else None
        trust_bundle = load_trust_bundle(args.trust_bundle) if args.trust_bundle else None
    except VerifierInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return run(
        envelope,
        jwks,
        predecessor_payload,
        trusted_tsa_keys=trusted_tsa_keys,
        bitcoin_headers=bitcoin_headers,
        counterparty=counterparty,
        required_extensions=args.require_extension or None,
        trust_bundle=trust_bundle,
        policy_digest=args.policy_digest,
        at_time=args.at_time,
        status_cutoff=args.status_cutoff,
    )


if __name__ == "__main__":
    sys.exit(main())
