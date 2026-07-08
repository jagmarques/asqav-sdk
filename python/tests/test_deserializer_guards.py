"""Required-field deserializer guards raise AsqavResponseError, not a
raw KeyError, when a cloud response omits a field the SDK requires.

Companion to test_verify_response_fields.py, which pins the positive
case (all fields present). This file pins the negative case: a
response missing a required field must surface one typed, catchable
``AsqavResponseError`` naming the field, never a bare ``KeyError`` at
a random dict index (rule 3.9).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asqav.client import AsqavResponseError, verify_signature


def _mock_httpx(payload: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    return response


def _full_payload() -> dict:
    """A complete `/verify` response; each test deletes one required key."""
    return {
        "signature_id": "sig_test",
        "agent_id": "agent_test",
        "agent_name": "test-agent",
        "action_id": "act_test",
        "action_type": "x.y",
        "payload": None,
        "signature": "ZmFrZQ==",
        "algorithm": "Ed25519",
        "signed_at": "2026-05-10T12:00:00+00:00",
        "verified": True,
        "verification_url": "https://verify.example/sig_test",
    }


@pytest.mark.parametrize("missing_field", ["action_id", "signature_id"])
def test_missing_required_field_raises_asqav_response_error(missing_field: str) -> None:
    """A response missing action_id or signature_id raises
    AsqavResponseError naming the field, not a raw KeyError."""
    payload = _full_payload()
    del payload[missing_field]
    with patch("asqav.client.httpx.get", return_value=_mock_httpx(payload)):
        with pytest.raises(AsqavResponseError, match=missing_field):
            verify_signature("sig_test")


@pytest.mark.parametrize("missing_field", ["action_id", "signature_id"])
def test_missing_required_field_never_leaks_a_bare_key_error(missing_field: str) -> None:
    """Regression guard: callers should only ever need to catch
    AsqavResponseError, never a bare KeyError, on a missing field."""
    payload = _full_payload()
    del payload[missing_field]
    with patch("asqav.client.httpx.get", return_value=_mock_httpx(payload)):
        try:
            verify_signature("sig_test")
        except KeyError:
            pytest.fail(f"verify_signature leaked a raw KeyError for '{missing_field}'")
        except AsqavResponseError:
            pass
