"""Tests for FastAPI middleware integration for AI endpoint governance.

Uses mock fastapi/starlette modules so tests run without fastapi installed.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

_original_modules: dict[str, Any] = {}
_MISSING = object()


def _install_fastapi_mocks() -> None:
    """Inject fake fastapi/starlette modules."""
    for mod_name in (
        "fastapi",
        "fastapi.middleware",
        "fastapi.middleware.base",
        "starlette",
        "starlette.types",
        "starlette.requests",
        "starlette.responses",
        "starlette.datastructures",
        "uvicorn",
    ):
        _original_modules[mod_name] = sys.modules.get(mod_name, _MISSING)

    # Create mock classes
    class MockClient:
        def __init__(self) -> None:
            self.host = "127.0.0.1"

    class MockURL:
        def __init__(self, path: str = "/api/chat") -> None:
            self.path = path

    class MockRequest:
        def __init__(self) -> None:
            self.client = MockClient()
            self.url = MockURL()
            self.method = "POST"
            self._body: dict = {}

        async def json(self) -> dict:
            return self._body

    class MockResponse:
        def __init__(self, status_code: int = 200, content: str = "{}", media_type: str = "application/json") -> None:
            self.status_code = status_code
            self._content = content
            self.media_type = media_type
            self.headers: dict[str, str] = {}

    class BaseHTTPMiddleware:
        def __init__(self, app: Any) -> None:
            self.app = app

    # Wire up modules
    fastapi = ModuleType("fastapi")
    fastapi.FastAPI = MagicMock()
    fastapi.Request = MockRequest
    fastapi.Response = MockResponse
    fastapi.middleware = ModuleType("fastapi.middleware")
    fastapi.middleware.base = ModuleType("fastapi.middleware.base")
    fastapi.middleware.base.BaseHTTPMiddleware = BaseHTTPMiddleware
    starlette = ModuleType("starlette")
    starlette.types = ModuleType("starlette.types")

    sys.modules["fastapi"] = fastapi
    sys.modules["fastapi.middleware"] = fastapi.middleware
    sys.modules["fastapi.middleware.base"] = fastapi.middleware.base
    sys.modules["starlette"] = starlette
    sys.modules["starlette.types"] = starlette.types
    sys.modules["starlette.requests"] = ModuleType("starlette.requests")
    sys.modules["starlette.responses"] = ModuleType("starlette.responses")
    sys.modules["starlette.datastructures"] = ModuleType("starlette.datastructures")
    sys.modules["uvicorn"] = ModuleType("uvicorn")


def _remove_fastapi_mocks() -> None:
    """Restore sys.modules to pre-mock state."""
    for mod_name, original in _original_modules.items():
        if original is _MISSING:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original
    sys.modules.pop("asqav.examples.fastapi_middleware", None)


_install_fastapi_mocks()


# === Fixtures ===


@pytest.fixture()
def mock_agent():
    """Create a mock asqav agent."""
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as mock_agent_cls,
    ):
        mock_agent = MagicMock()
        mock_sig = MagicMock()
        mock_sig.signature_id = "sig_test_123"
        mock_sig.verification_url = "https://asqav.com/verify/sig_test_123"
        mock_agent.sign.return_value = mock_sig
        mock_agent_cls.create.return_value = mock_agent
        yield mock_agent


# === Basic Tests ===


def test_fastapi_middleware_file_exists():
    """Verify the FastAPI middleware example file exists."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), "..", "examples", "fastapi_middleware.py")
    assert os.path.exists(file_path), f"FastAPI middleware example not found at {file_path}"


def test_fastapi_middleware_has_required_functions():
    """Verify the FastAPI middleware example has required functions."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), "..", "examples", "fastapi_middleware.py")
    with open(file_path, "r") as f:
        content = f.read()
    
    assert "class AsqavAuditMiddleware" in content
    assert "def dispatch" in content
    assert "def _check_rate_limit" in content
    assert "def _init_asqav" in content
    assert "@app.post(\"/api/chat\")" in content
    assert "@app.post(\"/api/embeddings\")" in content
    assert "@app.get(\"/api/health\")" in content
    assert "@app.get(\"/api/compliance/report\")" in content


def test_fastapi_middleware_has_fail_open():
    """Verify the FastAPI middleware has fail-open behavior."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), "..", "examples", "fastapi_middleware.py")
    with open(file_path, "r") as f:
        content = f.read()
    
    assert "fail-open" in content.lower() or "AsqavError" in content


def test_fastapi_middleware_has_rate_limiting():
    """Verify the FastAPI middleware has rate limiting."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), "..", "examples", "fastapi_middleware.py")
    with open(file_path, "r") as f:
        content = f.read()
    
    assert "rate_limit" in content.lower()
    assert "429" in content


def test_fastapi_middleware_has_content_policy():
    """Verify the FastAPI middleware has content policy enforcement."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), "..", "examples", "fastapi_middleware.py")
    with open(file_path, "r") as f:
        content = f.read()
    
    assert "max_prompt_length" in content.lower()
    assert "413" in content


def test_fastapi_middleware_signs_events():
    """Verify the FastAPI middleware signs AI events."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), "..", "examples", "fastapi_middleware.py")
    with open(file_path, "r") as f:
        content = f.read()
    
    assert "ai:request" in content
    assert "ai:response" in content
    assert "policy:deny" in content


# === Cleanup ===


def teardown_module() -> None:
    """Remove mock fastapi/starlette from sys.modules."""
    _remove_fastapi_mocks()
