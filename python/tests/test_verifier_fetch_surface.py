"""Fetch-surface guard: the verifier fetches only https same-host JSON (628 T3).

A verifier library must never turn a URL into a local-file read or a
cleartext request. These tests pin that fetch_jwks rejects file:// and
http:// with a scheme error (not a downstream parse/network error) and
that the redirect handler refuses cross-host hops while allowing
same-host ones.
"""

import re
import urllib.request

import pytest

from asqav.verifier.verify_receipt import (
    VerifierInputError,
    _SameHostRedirectHandler,
    fetch_jwks,
)


def test_file_url_rejected_as_scheme() -> None:
    """file:// must fail on the scheme, not on whatever the file parses to."""
    with pytest.raises(VerifierInputError, match=r"[Hh]ttps|scheme"):
        fetch_jwks("file:///etc/passwd")


def test_http_url_rejected_as_scheme() -> None:
    """http:// must fail on the scheme, even when nothing listens there."""
    with pytest.raises(VerifierInputError, match=r"[Hh]ttps|scheme"):
        fetch_jwks("http://127.0.0.1:1/.well-known/jwks.json", timeout=5)


def test_redirect_handler_refuses_cross_host() -> None:
    """A hop to another host raises instead of following."""
    handler = _SameHostRedirectHandler()
    req = urllib.request.Request("https://issuer.example/.well-known/jwks.json")
    with pytest.raises(VerifierInputError, match=r"[Cc]ross-host|redirect"):
        handler.redirect_request(
            req, None, 302, "Found", {}, "https://other.example/keys"
        )


def test_redirect_handler_allows_same_host() -> None:
    """A same-host hop still builds the follow-up request."""
    handler = _SameHostRedirectHandler()
    req = urllib.request.Request("https://issuer.example/.well-known/jwks.json")
    nxt = handler.redirect_request(
        req, None, 302, "Found", {}, "https://issuer.example/v2/keys"
    )
    assert nxt is not None
    assert re.search(r"issuer\.example/v2/keys", nxt.get_full_url())
