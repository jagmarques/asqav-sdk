"""Signing preparation order and validation before any HTTP request."""
from unittest.mock import patch

import pytest

from tests.test_client_threat_framework_mappings import _agent, _ok_response


@pytest.mark.parametrize("arguments,token", [
    ({"receipt_type": "unknown:receipt"}, "invalid_receipt_type"),
    ({"iso_42001": ["A" * 129]}, "iso_42001_entry_invalid"),
    ({"witness_policy": {"required": 0, "witnesses": ["rfc3161"]}},
     "witness_policy_required_out_of_range"),
])
def test_sign_checks_inputs_before_http(arguments, token):
    with patch("asqav.client._post", return_value=_ok_response()) as post:
        with pytest.raises(ValueError, match=token):
            _agent().sign("api:call", {"value": 1}, **arguments)
        post.assert_not_called()


def test_sign_prepares_context_before_schema_and_detectors():
    steps = []

    def before(action, context):
        steps.append("before")
        assert context["_trace_id"] == "trace_fixture"
        return dict(context, hook_value=2)

    def schema(context):
        steps.append("schema")
        assert context["hook_value"] == 2

    def detectors(action, context):
        steps.append("detectors")
        assert list(context) == sorted(context)
        return [{"detector": "fixture", "verdict": "allow"}]

    with patch("asqav.hooks._dispatch_before", side_effect=before), \
         patch("asqav._detectors.run_detectors", side_effect=detectors), \
         patch("asqav.client._post", return_value=_ok_response()) as post:
        _agent().sign("api:call", {"z": 1}, trace_id="trace_fixture", context_schema=schema)
    assert steps == ["before", "schema", "detectors"]
    context = post.call_args.args[1]["context"]
    assert context["_trace_id"] == "trace_fixture" and context["hook_value"] == 2
    assert context["_detectors"] == [{"detector": "fixture", "verdict": "allow"}]
