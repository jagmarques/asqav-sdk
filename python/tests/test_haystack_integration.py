"""Tests for Haystack pipeline component integration."""

from __future__ import annotations

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from asqav.client import AsqavError, SignatureResponse

# ---------------------------------------------------------------------------
# Mock haystack before importing the integration module
# ---------------------------------------------------------------------------

_mock_haystack = ModuleType("haystack")


class _MockComponent:
    """Mock for haystack.component decorator."""

    def __call__(self, cls):
        """Decorator: return the class unchanged."""
        return cls

    @staticmethod
    def output_types(**kwargs):
        """Return a no-op decorator for output type declarations."""
        def decorator(func):
            return func
        return decorator


_mock_haystack.component = _MockComponent()  # type: ignore[attr-defined]
sys.modules.setdefault("haystack", _mock_haystack)

from asqav.extras.haystack import AsqavComponent  # noqa: E402

MOCK_SIGN_RESPONSE: dict = {
    "signature": "sig_abc123",
    "signature_id": "sid_abc123",
    "action_id": "act_abc123",
    "timestamp": 1700000001.0,
    "verification_url": "https://api.asqav.com/verify/sid_abc123",
}


def _make_component() -> AsqavComponent:
    """Create a component with mocked Agent."""
    with patch("asqav.client._api_key", "sk_test"), \
         patch("asqav.extras._base.Agent") as mock_agent_cls:
        mock_agent_cls.create.return_value = MagicMock()
        return AsqavComponent(agent_name="test-pipeline")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_signs_action(self):
        comp = _make_component()
        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        comp._sign_action = MagicMock(return_value=mock_sig)

        comp.run(data="hello world")

        comp._sign_action.assert_called_once_with(
            "pipeline:process",
            {
                "data_length": 11,
                "metadata_keys": [],
            },
        )

    def test_run_returns_output_dict(self):
        comp = _make_component()
        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        comp._sign_action = MagicMock(return_value=mock_sig)

        result = comp.run(data="test data")

        assert result == {
            "data": "test data",
            "metadata": {},
            "signature_id": "sid_abc123",
        }

    def test_run_with_metadata(self):
        comp = _make_component()
        mock_sig = SignatureResponse(**MOCK_SIGN_RESPONSE)
        comp._sign_action = MagicMock(return_value=mock_sig)

        metadata = {"source": "file.txt", "page": 1}
        result = comp.run(data="content", metadata=metadata)

        assert result["metadata"] == metadata
        call_context = comp._sign_action.call_args[0][1]
        assert sorted(call_context["metadata_keys"]) == ["page", "source"]

    def test_run_signature_id_none_on_error(self):
        comp = _make_component()
        comp._sign_action = MagicMock(return_value=None)

        result = comp.run(data="test")

        assert result["signature_id"] is None

    def test_fail_open_on_sign_error(self):
        """run() still returns data even when signing raises."""
        comp = _make_component()
        comp._sign_action = MagicMock(return_value=None)

        result = comp.run(data="important data", metadata={"key": "val"})

        assert result["data"] == "important data"
        assert result["metadata"] == {"key": "val"}
        assert result["signature_id"] is None
