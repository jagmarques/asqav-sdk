"""Mint minimal RFC 3161 TimeStampResp tokens for the anchor-verification tests.

A real TSA token is a CMS SignedData wrapping a TSTInfo; the verifier parses
the DER directly, so the tests need tokens built byte by byte rather than by a
TSA service. The shape mirrors the production anchor in
verifier/docs/fixtures/published-receipt.json: no signedAttrs (the TSA signs
the eContent bytes directly) and a [0] subjectKeyIdentifier sid.
"""

from __future__ import annotations

import base64


def _len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    body = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(body)]) + body


def _tlv(tag: int, content: bytes) -> bytes:
    return bytes([tag]) + _len(len(content)) + content


def _seq(*items: bytes) -> bytes:
    return _tlv(0x30, b"".join(items))


def _set(*items: bytes) -> bytes:
    return _tlv(0x31, b"".join(items))


def _int(n: int) -> bytes:
    body = n.to_bytes(max(1, (n.bit_length() + 7) // 8), "big")
    if body[0] & 0x80:
        body = b"\x00" + body
    return _tlv(0x02, body)


def _oid(dotted: str) -> bytes:
    arcs = [int(x) for x in dotted.split(".")]
    out = bytearray([arcs[0] * 40 + arcs[1]])
    for arc in arcs[2:]:
        stack = [arc & 0x7F]
        arc >>= 7
        while arc:
            stack.append(0x80 | (arc & 0x7F))
            arc >>= 7
        out.extend(reversed(stack))
    return _tlv(0x06, bytes(out))


def _oct(content: bytes) -> bytes:
    return _tlv(0x04, content)


_OID_SHA256 = "2.16.840.1.101.3.4.2.1"
_OID_SIGNED_DATA = "1.2.840.113549.1.7.2"
_OID_TSTINFO = "1.2.840.113549.1.9.16.1.4"
_OID_ML_DSA_65 = "2.16.840.1.101.3.4.3.18"


def make_tst_info(digest32: bytes, gen_time: str = "20260601000000Z") -> bytes:
    """A TSTInfo committing digest32, signed-content form the verifier parses."""
    return _seq(
        _int(1),
        _oid("1.2.3.4"),  # policy OID, opaque to the check
        _seq(_seq(_oid(_OID_SHA256), _tlv(0x05, b"")), _oct(digest32)),
        _int(42),
        _tlv(0x18, gen_time.encode("ascii")),
    )


_OID_CONTENT_TYPE = "1.2.840.113549.1.9.3"
_OID_MESSAGE_DIGEST = "1.2.840.113549.1.9.4"


def make_signed_attrs(
    tst: bytes,
    *,
    include_content_type: bool = True,
    content_type_oid: str = _OID_TSTINFO,
    message_digest: bytes | None = None,
) -> bytes:
    """The signedAttrs CONTENT bytes (attributes concatenated, no outer tag).

    Public TSAs all emit signedAttrs; the production fixture does not, so this
    is what exercises the `_cms_message_digest_ok` / `_cms_signed_bytes` branch.
    The knobs exist to build the RFC 5652 s11.1 violations the verifier must
    reject: a missing contentType, and a contentType naming other content.
    """
    import hashlib

    digest = hashlib.sha256(tst).digest() if message_digest is None else message_digest
    attrs = b""
    if include_content_type:
        attrs += _seq(_oid(_OID_CONTENT_TYPE), _set(_oid(content_type_oid)))
    attrs += _seq(_oid(_OID_MESSAGE_DIGEST), _set(_oct(digest)))
    return attrs


def signed_attrs_signing_input(attrs: bytes) -> bytes:
    """Bytes a CMS signature covers when signedAttrs is present: SET OF re-tag."""
    return _tlv(0x31, attrs)


def make_timestamp_resp(
    tst: bytes,
    signature: bytes,
    *,
    status: int = 0,
    sig_alg_oid: str = _OID_ML_DSA_65,
    certs: list[bytes] | None = None,
    sid: bytes | None = None,
    signed_attrs: bytes | None = None,
    econtent_type_oid: str = _OID_TSTINFO,
) -> bytes:
    """Wrap a TSTInfo and its CMS signature in a TimeStampResp.

    ``signed_attrs`` carries the signedAttrs CONTENT bytes (see
    make_signed_attrs); omit it for the no-signedAttrs production shape.
    """
    signer_fields = [
        _int(3),
        sid if sid is not None else _tlv(0xA0, b"\x00" * 20),  # [0] SKI, as in production
        _seq(_oid(_OID_SHA256), _tlv(0x05, b"")),
    ]
    if signed_attrs is not None:
        signer_fields.append(_tlv(0xA0, signed_attrs))  # [0] IMPLICIT SignedAttributes
    signer_fields += [
        _seq(_oid(sig_alg_oid)),
        _oct(signature),
    ]
    sd_items = [
        _int(3),
        _set(_seq(_oid(_OID_SHA256), _tlv(0x05, b""))),
        _seq(_oid(econtent_type_oid), _tlv(0xA0, _oct(tst))),
    ]
    if certs:
        sd_items.append(_tlv(0xA0, b"".join(certs)))  # [0] IMPLICIT CertificateSet
    sd_items.append(_set(_seq(*signer_fields)))
    content_info = _seq(_oid(_OID_SIGNED_DATA), _tlv(0xA0, _seq(*sd_items)))
    return _seq(_seq(_int(status)), content_info)


def mint_ml_dsa_anchor(digest32: bytes, gen_time: str = "20260601000000Z"):
    """(base64 token, raw public key) for a fresh ML-DSA-65 TSA keypair."""
    from dilithium_py.ml_dsa import ML_DSA_65

    pk, sk = ML_DSA_65.keygen()
    tst = make_tst_info(digest32, gen_time)
    token = make_timestamp_resp(tst, ML_DSA_65.sign(sk, tst))
    return base64.b64encode(token).decode(), pk
