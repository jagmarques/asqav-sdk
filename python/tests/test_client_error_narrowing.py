"""Client excepts catch only what they expect; the rest raise."""

from __future__ import annotations

import logging
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from asqav.client import APIError, Span, _spans, _transport, span, verify_output


def test_span_logs_a_failed_sign(caplog: pytest.LogCaptureFixture) -> None:
    agent = MagicMock()
    agent.sign.side_effect = APIError("cloud down", 503)
    with patch.object(_spans, "get_agent", return_value=agent), caplog.at_level(logging.WARNING):
        with span("api:call"):
            pass
    assert "asqav span 'api:call' was not signed" in caplog.text


def test_span_raises_an_unexpected_sign_error() -> None:
    agent = MagicMock()
    agent.sign.side_effect = RuntimeError("bug in the signer")
    with patch.object(_spans, "get_agent", return_value=agent):
        with pytest.raises(RuntimeError, match="bug in the signer"):
            with span("api:call"):
                pass


def test_flush_spans_logs_a_failed_export(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(_spans, "_otel_endpoint", "http://otel.invalid/v1/traces")
    monkeypatch.setattr(_spans, "_completed_spans", [Span(span_id="s1", name="n", start_time=0.0)])
    refused = urllib.error.URLError("connection refused")
    with patch("urllib.request.urlopen", side_effect=refused), caplog.at_level(logging.WARNING):
        _spans.flush_spans()
    assert "asqav span export to http://otel.invalid/v1/traces failed" in caplog.text


def test_handle_response_raises_an_unexpected_body_error() -> None:
    response = MagicMock(status_code=500, text="boom")
    response.json.side_effect = RuntimeError("bug in the decoder")
    with pytest.raises(RuntimeError, match="bug in the decoder"):
        _transport._handle_response(response)


def test_http_error_text_raises_an_unexpected_read_error() -> None:
    err = MagicMock()
    err.read.side_effect = RuntimeError("bug in the reader")
    with pytest.raises(RuntimeError, match="bug in the reader"):
        _transport._http_error_text(err)


def test_verify_output_raises_on_an_unhashable_output() -> None:
    verification = MagicMock(verified=True, payload={"context": {"output_hash": "abc"}})
    with patch("asqav.client._actions.verify_signature", return_value=verification):
        with pytest.raises(TypeError):
            verify_output("sig_1", {1: "a", "b": 2})


def test_negctl_handle_response_still_maps_an_unreadable_body_to_api_error() -> None:
    response = MagicMock(status_code=500, text="boom")
    response.json.side_effect = ValueError("not json")
    with pytest.raises(APIError, match="boom"):
        _transport._handle_response(response)


def test_http_error_text_names_an_unreadable_body() -> None:
    err = MagicMock()
    err.read.side_effect = OSError("socket closed")
    assert _transport._http_error_text(err) == "<unreadable body: OSError>"
