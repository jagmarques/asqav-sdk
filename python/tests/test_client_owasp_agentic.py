"""Bare ASI identifiers through validation and real HTTP serialization."""
import json
from unittest.mock import patch

import httpx
import pytest

import asqav.client as client
from tests.test_client_threat_framework_mappings import _agent, _ok_response


@pytest.mark.parametrize("value", [
    ["ASI00"], ["ASI11"], ["ASI1"], ["asi01"], ["ASI01:2026"],
    ["ASI01-2026"], ["ASI01\n"], [" ASI01"], ["ASI01 "],
    [""], [42], [True], [None], [{}], [["ASI01"]], ["A" * 129],
    ["ASI01", "LLM01"],
])
def test_agentic_invalid_identifier_rejected_before_http(value):
    with patch("asqav.client._post", return_value=_ok_response()) as post:
        with pytest.raises(ValueError, match="owasp_agentic_top10_entry_invalid"):
            _agent().sign("api:call", {}, owasp_agentic_top10=value)
        post.assert_not_called()


@pytest.mark.parametrize("compliance", [True, False])
def test_agentic_serialized_bytes_and_omission(monkeypatch, compliance):
    requests = []

    def receive(request):
        assert request.url.host == "api.example.com"
        requests.append(json.loads(request.content))
        return httpx.Response(201, json=_ok_response())

    values = [f"ASI{n:02d}" for n in range(1, 11)] + ["ASI01"]
    monkeypatch.setattr(client, "_api_key", "asq_test_fixture")
    with httpx.Client(
        base_url="https://api.example.com/api/v1", transport=httpx.MockTransport(receive)
    ) as transport:
        monkeypatch.setattr(client, "_client", transport)
        _agent().sign("api:call", {}, owasp_agentic_top10=values, compliance_mode=compliance)
        _agent().sign("api:call", {}, owasp_agentic_top10=None, compliance_mode=compliance)
    assert len(requests) == 2
    if compliance:
        assert requests[0]["owasp_agentic_top10"] == values
    else:
        assert "owasp_agentic_top10" not in requests[0]
    assert "owasp_agentic_top10" not in requests[1]
    assert all("framework_mappings_self_declared" not in request for request in requests)


@pytest.mark.parametrize("value", ["ASI01", 42, {"ids": ["ASI01"]}])
def test_agentic_non_list_rejected_before_http(value):
    with patch("asqav.client._post", return_value=_ok_response()) as post:
        with pytest.raises(ValueError, match="owasp_agentic_top10_must_be_non_empty_list"):
            _agent().sign("api:call", {}, owasp_agentic_top10=value)
        post.assert_not_called()
