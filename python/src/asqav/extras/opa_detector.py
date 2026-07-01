"""OpaDetector: posts action/context to an OPA REST endpoint for policy decisions.

Install: no extra dep needed; uses httpx if available (pip install asqav[httpx])
or falls back to stdlib urllib.

Usage::

    from asqav.extras.opa_detector import OpaDetector
    import asqav

    asqav.register_detector(OpaDetector(
        opa_url="http://localhost:8181",
        policy_path="asqav/action/allow",
    ))

OPA endpoint: ``POST {opa_url}/v1/data/{policy_path}``
Request body: ``{"input": {"action_type": ..., "context": ...}}``
Expected response: ``{"result": true}`` or ``{"result": false}``
A missing ``result`` key is treated as deny (fail-closed).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .._detectors import DetectorResult

logger = logging.getLogger("asqav")

try:
    import httpx as _httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    _httpx = None  # type: ignore


class OpaDetector:
    """DetectorPlugin that consults an OPA REST endpoint for allow/deny decisions.

    Args:
        opa_url:     Base URL of the OPA server, e.g. ``"http://localhost:8181"``.
        policy_path: OPA policy path, e.g. ``"myapp/action/allow"``.
                     Posted to ``{opa_url}/v1/data/{policy_path}``.
        timeout:     HTTP request timeout in seconds. Default 5.
    """

    name = "opa"

    def __init__(
        self,
        *,
        opa_url: str = "http://localhost:8181",
        policy_path: str = "asqav/action/allow",
        timeout: float = 5.0,
    ) -> None:
        self._opa_url = opa_url.rstrip("/")
        self._policy_path = policy_path.lstrip("/")
        self._timeout = timeout

    def _endpoint(self) -> str:
        return f"{self._opa_url}/v1/data/{self._policy_path}"

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST payload as JSON; return parsed JSON response dict."""
        endpoint = self._endpoint()
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if _HTTPX_AVAILABLE and _httpx is not None:
            resp = _httpx.post(
                endpoint,
                content=body,
                headers=headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()

        # Fallback to stdlib urllib.
        import urllib.request

        req = urllib.request.Request(
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def inspect(self, action_type: str, context: dict[str, Any]) -> DetectorResult:
        """POST to OPA; map the boolean result to allow/deny."""
        payload = {"input": {"action_type": action_type, "context": context}}
        try:
            data = self._post_json(payload)
        except Exception as exc:
            # Let the caller (run_detectors) handle fail-open / fail-closed.
            raise RuntimeError(
                f"opa_request_failed: could not reach OPA at {self._endpoint()}: {exc}"
            ) from exc

        # OPA returns {"result": <value>}; missing result = fail-closed deny.
        result_val = data.get("result")
        allow = bool(result_val) if result_val is not None else False

        if allow:
            return DetectorResult(allow=True, detector=self.name)

        return DetectorResult(
            allow=False,
            confidence=1.0,
            labels=["opa_deny"],
            detector=self.name,
            reason=(
                f"OPA policy '{self._policy_path}' returned deny "
                f"(result={result_val!r})"
            ),
        )
