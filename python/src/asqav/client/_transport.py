"""API configuration, retries and the HTTP calls every client function makes."""

from __future__ import annotations

import json as _json
import os
import time
from collections.abc import Callable
from http.client import HTTPException
from typing import Any

from .._useragent import USER_AGENT
from ..credentials import resolve_api_base, resolve_api_key
from ..retry import with_retry
from ._errors import (
    RESPONSE_ERROR_DOCS_URL,
    APIError,
    AsqavResponseError,
    AuthenticationError,
    RateLimitError,
    _connection_error,
    _error_message,
    _one_line,
)

# Retry configuration for API calls
_MAX_RETRIES = 5


_RETRY_DELAYS = [0.5, 1.0, 2.0, 4.0, 8.0]  # Exponential backoff


    # Strict ingest (419): API responses carrying receipts/records never parse
    # last-wins; a duplicated member name is a terminal failure, raised loud.
def _strict_response_json(response: Any) -> dict[str, Any]:
    from .. import strict_json

    try:
        return strict_json.loads(response.text)
    except (_json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _non_json_error(exc) from exc


    # A 2xx body that is not JSON (e.g. a proxy page) is an AsqavResponseError.
def _non_json_error(exc: Exception) -> AsqavResponseError:
    return AsqavResponseError(
        f"the API answered with a non-JSON body ({_one_line(str(exc))}); "
        "check the API base and any proxy in between",
        docs_url=RESPONSE_ERROR_DOCS_URL,
    )


    # Run with exponential backoff retry on rate-limit and network errors.
def _with_retry(func: Callable[[], Any]) -> Any:
    last_error: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS):
        try:
            return func()
        except (RateLimitError, ConnectionError, TimeoutError) as e:
            last_error = e
            if attempt < len(_RETRY_DELAYS) - 1:
                time.sleep(delay)
    raise last_error


try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore


# API configuration
_api_key: str | None = None


# Same order as init(); see credentials.resolve_api_base.
_api_base: str = resolve_api_base()


# Last base passed to init(); a later init() without one keeps it.
_explicit_api_base: str | None = None


def _join_url(base: str, path: str) -> str:
    """Join ``base`` and ``path`` keeping the base path (``/api/v1``).
    ``urljoin`` drops it for leading-slash paths; urllib needs it."""
    return base.rstrip("/") + "/" + path.lstrip("/")


_client: Any = None


_NO_BODY: Any = object()


# Hybrid GDPR mode: resolved at init() time, overridable per-call. See _mode.py.
_mode: str = "full-payload"


_org_salt: bytes | None = None


#: Per-request HTTP timeout in seconds; init(timeout=...) overrides it.
_timeout: float = 30.0


def health_check() -> dict[str, Any]:
    """Check API connectivity and return the server status dict.
    Raises AuthenticationError (bad key) or APIError (unreachable)."""
    return _get("/health")


def init(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    mode: str = "auto",
    org_salt: bytes | None = None,
    timeout: float | None = None,
) -> None:
    """Initialize the Asqav SDK.

    Args:
        api_key: Your Asqav API key. Can also be set via ASQAV_API_KEY env var.
        base_url: API base URL. When omitted, a base given to a previous
            init() call in this process stays; otherwise it resolves in this
            order: ``ASQAV_API_URL``, ``ASQAV_API_BASE``, the base saved by
            ``asqav login --api-base``, then ``https://api.asqav.com/api/v1``.
        mode: Wire format for ``Agent.sign``. One of ``"auto"`` (default),
            ``"hash-only"``, or ``"full-payload"``. ``auto`` resolves to
            ``hash-only`` for the Asqav cloud (``*.asqav.com``) and
            ``full-payload`` for self-hosted deployments. The
            ``ASQAV_MODE`` env var is consulted when ``mode="auto"``.
        org_salt: Optional 32-byte salt that you generate and hold. When
            set, hash-only requests use HMAC-SHA-256 keyed by it instead
            of plain SHA-256. Asqav never receives it and cannot recover
            it for you. Leave it unset and the digest is an unsalted
            SHA-256, which can be guessed for a predictable context.
            Salted digests still travel labelled ``sha256``, so a third
            party recomputing per ``docs/fingerprint-spec.md`` would see a
            mismatch; the current backend rejects the body before signing
            (see CAVEAT below), so no signature verifies for this flow.
            CAVEAT: the current backend rejects caller-computed-HMAC
            hash-only bodies (``keyed_digest_requires_context``). Keyed
            digests are server-keyed: send ``context`` (``verified_keyed``).
        timeout: Per-request HTTP timeout in seconds (default 30). The
            harness hook passes its own deadline here so a hung signer
            cannot outlive the gate.

    Raises:
        AuthenticationError: If no API key is provided.

    Example:
        import asqav

        # Using parameter
        asqav.init(api_key="sk_...")

        # Using environment variable
        os.environ["ASQAV_API_KEY"] = "sk_..."
        asqav.init()

        # Force hash-only against a custom URL
        asqav.init(api_key="sk_...", mode="hash-only", org_salt=bytes.fromhex("..."))
    """
    global _api_key, _api_base, _explicit_api_base, _client, _mode, _org_salt, _timeout

    _api_key = resolve_api_key(api_key)

    if not _api_key:
        raise AuthenticationError(
            "API key required. Run `asqav login`, set ASQAV_API_KEY, or pass api_key "
            "to init(). A saved key is read from ~/.asqav/credentials. Get yours at asqav.com"
        )

    _explicit_api_base = base_url or _explicit_api_base
    _api_base = resolve_api_base(_explicit_api_base)

    from .._mode import _resolve_mode

    _mode = _resolve_mode(
        api_base_url=_api_base,
        env=os.environ.get("ASQAV_MODE"),
        explicit=mode,
    )
    _org_salt = org_salt
    _timeout = float(timeout) if timeout else 30.0

    if _HTTPX_AVAILABLE:
        _client = httpx.Client(
            base_url=_api_base,
            headers={"X-API-Key": _api_key, "User-Agent": USER_AGENT},
            timeout=_timeout,
        )


    # Make a GET request to the API.
@with_retry()
def _get(path: str) -> dict[str, Any]:
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        return _httpx_call("get", path)
    else:
        # Use stdlib urllib if httpx not installed
        return _urllib_request("GET", path)


    # Make a POST request to the API.
@with_retry()
def _post(path: str, data: dict[str, Any]) -> dict[str, Any]:
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        return _httpx_call("post", path, data)
    else:
        return _urllib_request("POST", path, data)


    # Make a PATCH request to the API.
@with_retry()
def _patch(path: str, data: dict[str, Any]) -> dict[str, Any]:
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        return _httpx_call("patch", path, data)
    else:
        return _urllib_request("PATCH", path, data)


    # Make a PUT request to the API.
@with_retry()
def _put(path: str, data: dict[str, Any]) -> dict[str, Any]:
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        return _httpx_call("put", path, data)
    else:
        return _urllib_request("PUT", path, data)


    # Make a DELETE request to the API.
@with_retry()
def _delete(path: str) -> dict[str, Any]:
    _ensure_initialized()

    if _HTTPX_AVAILABLE and _client:
        return _httpx_call("delete", path)
    else:
        return _urllib_request("DELETE", path)


    # Send one request on the shared httpx client; transport errors raise.
def _httpx_call(verb: str, path: str, data: Any = _NO_BODY) -> dict[str, Any]:
    kwargs = {} if data is _NO_BODY else {"json": data}
    try:
        response = getattr(_client, verb)(path, **kwargs)
    except httpx.RequestError as exc:
        raise _connection_error(_join_url(_api_base, path), exc) from exc
    _handle_response(response)
    result: dict[str, Any] = _strict_response_json(response)
    return result


    # Ensure the SDK is initialized.
def _ensure_initialized() -> None:
    if not _api_key:
        raise AuthenticationError("Call asqav.init() first. Get your API key at asqav.com")


    # Handle API response errors.
def _handle_response(response: Any) -> None:
    if response.status_code == 401:
        raise AuthenticationError("Invalid API key")
    elif response.status_code == 429:
        raise RateLimitError("Rate limit exceeded. Upgrade at asqav.com/pricing")
    elif response.status_code >= 400:
        try:
            body = response.json()
        except ValueError:
            body = None
        text = response.text if isinstance(response.text, str) else ""
        raise APIError(_error_message(response.status_code, text, body), response.status_code)


    # HTTP client using stdlib urllib (used when httpx not installed).
def _urllib_request(
    method: str,
    path: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import json
    import urllib.error
    import urllib.request

    url = _join_url(_api_base, path)
    headers = {
        "X-API-Key": _api_key,
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    body = json.dumps(data).encode("utf-8") if data else None

    def _do_request() -> dict[str, Any]:
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=_timeout) as response:
            # Strict ingest (419): a duplicated member name fails the response.
            from .. import strict_json

            try:
                result: dict[str, Any] = strict_json.loads(response.read().decode("utf-8"))
            except (_json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise _non_json_error(exc) from exc
            return result

    try:
        return _with_retry(_do_request)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise AuthenticationError("Invalid API key") from e
        elif e.code == 429:
            raise RateLimitError("Rate limit exceeded") from e
        raise APIError(_error_message(e.code, _http_error_text(e)), e.code) from e
    except (urllib.error.URLError, OSError) as e:
        raise _connection_error(url, e) from e


    # Body of a urllib HTTPError as text ("" if unreadable) to keep the reason.
def _http_error_text(err: Any) -> str:
    try:
        raw = err.read()
    except (OSError, HTTPException):
        return ""
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else ""
