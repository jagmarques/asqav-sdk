"""Client exceptions and the messages they carry."""

from __future__ import annotations

import json as _json
from typing import Any

from .._sdk_constants import SDK_CONSTANTS


    # Base exception for asqav errors.
class AsqavError(Exception):

    pass


    # Raised when API key is missing or invalid.
class AuthenticationError(AsqavError):

    pass


    # Raised when rate limit is exceeded.
class RateLimitError(AsqavError):

    pass


    # Raised for general API errors.
class APIError(AsqavError):

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


    # API unreachable (refused, DNS, timeout, TLS): APIError status 0.
    # Not retried, so a caller's deadline (the hook gate's timeout) still holds.
class AsqavConnectionError(APIError):

    def __init__(self, message: str) -> None:
        super().__init__(message, 0)


    # One-line connection error naming the URL and a next step.
def _connection_error(url: str, exc: BaseException) -> AsqavConnectionError:
    reason = getattr(exc, "reason", None) or exc
    return AsqavConnectionError(
        f"Network error: {_one_line(str(reason))} (could not reach the Asqav API at {url}; "
        "check ASQAV_API_URL or `asqav login --api-base`)"
    )


    # Collapse whitespace and cap length so an error message stays on one line.
def _one_line(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


    # Server reason from a JSON error body: detail plus code.
def _reason_from_body(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    detail = body.get("detail") or body.get("error") or body.get("message")
    code = body.get("code")
    if isinstance(detail, dict):
        code = code or detail.get("code")
        detail = detail.get("message") or detail.get("detail")
    if isinstance(detail, list):
        detail = _validation_list_text(detail)
    text = _one_line(str(detail)) if detail else ""
    if code and str(code) not in text:
        text = f"{text} ({code})" if text else str(code)
    return text


    # FastAPI validation list as "field: message; field: message".
def _validation_list_text(items: list[Any]) -> str:
    return "; ".join(
        f"{e.get('loc', ['?'])[-1]}: {e.get('msg', '')}".strip(": ")
        for e in items
        if isinstance(e, dict)
    ) or str(items)


    # One-line APIError message for an HTTP error status and raw body text.
def _error_message(status: int, text: str, body: Any = None) -> str:
    if body is None:
        try:
            body = _json.loads(text)
        except ValueError:
            body = None
    reason = _reason_from_body(body)
    if reason:
        return reason
    flat = _one_line(text or "")
    if not flat:
        return f"HTTP {status}"
    if "<" not in flat:
        return f"HTTP {status}: {flat}"
    return f"HTTP {status} with a non-JSON body (often a proxy or gateway error page)"


class AsqavValidationError(ValueError):
    """Client-side wire-vocabulary rejection with a docs pointer (rule 3.9).
    A ValueError; ``docs_url`` names the rule page; the value is cut to 64."""

    def __init__(self, message: str, *, docs_url: str) -> None:
        super().__init__(message)
        self.docs_url = docs_url


class AsqavResponseError(AsqavError):
    """Raised when a cloud response omits a field the SDK requires (rule 3.9).

    Replaces a raw ``KeyError`` so callers can catch one asqav-owned type
    with a clear message naming the missing field, instead of a bare
    traceback pointing at an internal dict index.
    """

    def __init__(self, message: str, *, docs_url: str) -> None:
        super().__init__(message)
        self.docs_url = docs_url


#: Docs page anchoring missing-required-field response errors (rule 3.9).
RESPONSE_ERROR_DOCS_URL: str = SDK_CONSTANTS["RESPONSE_ERROR_DOCS_URL"]


def require_field(data: dict[str, Any], name: str) -> Any:
    """Return ``data[name]`` or raise ``AsqavResponseError`` if absent.

    Centralizes required-field response parsing so a cloud response
    missing e.g. ``action_id`` raises one typed, catchable error instead
    of a raw ``KeyError`` at a random dict index.
    """
    if name not in data:
        raise AsqavResponseError(
            f"response missing required field '{name}'",
            docs_url=RESPONSE_ERROR_DOCS_URL,
        )
    return data[name]
