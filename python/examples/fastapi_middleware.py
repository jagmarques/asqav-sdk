"""
FastAPI Middleware Integration for AI Endpoint Governance

This example demonstrates how to use asqav with FastAPI to add audit trails,
policy enforcement, and compliance reporting for AI-serving endpoints.

Requirements:
    pip install asqav fastapi httpx

Usage:
    # Set your API key
    export ASQAV_API_KEY=sk_your_api_key_here

    # Run the server
    uvicorn fastapi_middleware:app --reload

    # Test the endpoint
    curl -X POST http://localhost:8000/api/chat \
         -H "Content-Type: application/json" \
         -d '{"model": "gpt-4", "prompt": "Hello, AI!"}'
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

import asqav
from asqav.client import Agent, AsqavError
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger("asqav")

# === Configuration ===

ASQAV_API_KEY = os.getenv("ASQAV_API_KEY", "sk_your_api_key_here")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "4096"))

# === FastAPI App ===

app = FastAPI(
    title="AI Governance Demo",
    description="FastAPI endpoints with asqav audit trails and policy enforcement",
    version="1.0.0",
)

# === Audit Middleware ===


class AsqavAuditMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that signs every AI request/response via asqav.

    This middleware:
    1. Signs each request as an audit event
    2. Enforces rate limits and content policies
    3. Records response metadata for compliance reporting
    """

    def __init__(
        self,
        app: ASGIApp,
        agent: Agent,
        rate_limit: int = RATE_LIMIT_PER_MINUTE,
        max_prompt_length: int = MAX_PROMPT_LENGTH,
    ) -> None:
        super().__init__(app)
        self._agent = agent
        self._rate_limit = rate_limit
        self._max_prompt_length = max_prompt_length
        self._request_counts: dict[str, list[float]] = {}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process each request: audit, enforce policy, sign response."""
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # Skip non-AI endpoints
        if not path.startswith("/api/") or request.method == "OPTIONS":
            return await call_next(request)

        # --- Rate limit enforcement ---
        if not self._check_rate_limit(client_ip):
            sig = self._agent.sign(
                "policy:deny",
                {
                    "reason": "rate_limit_exceeded",
                    "client_ip": client_ip,
                    "limit": self._rate_limit,
                },
            )
            return Response(
                status_code=429,
                content=f'{{"error": "Rate limit exceeded", "signature_id": "{sig.signature_id if sig else "N/A"}"}}',
                media_type="application/json",
            )

        # --- Content policy enforcement ---
        try:
            body = await request.json()
        except Exception:
            body = {}

        prompt = body.get("prompt", "")
        if len(prompt) > self._max_prompt_length:
            self._agent.sign(
                "policy:deny",
                {
                    "reason": "prompt_too_long",
                    "prompt_length": len(prompt),
                    "max_length": self._max_prompt_length,
                },
            )
            return Response(
                status_code=413,
                content='{"error": "Prompt exceeds maximum length"}',
                media_type="application/json",
            )

        # --- Sign the request ---
        start_time = time.time()
        sig = self._agent.sign(
            "ai:request",
            {
                "path": path,
                "method": request.method,
                "client_ip": client_ip,
                "model": body.get("model", "unknown"),
                "prompt_length": len(prompt),
            },
        )

        # --- Call the actual endpoint ---
        response = await call_next(request)
        elapsed_ms = (time.time() - start_time) * 1000

        # --- Sign the response ---
        self._agent.sign(
            "ai:response",
            {
                "path": path,
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed_ms, 2),
                "request_signature": sig.signature_id if sig else None,
            },
        )

        # --- Add signature header to response ---
        if sig:
            response.headers["X-Asqav-Signature"] = sig.signature_id
            response.headers["X-Asqav-Verify"] = sig.verify_url

        return response

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit."""
        now = time.time()
        window_start = now - 60  # 1-minute window

        if client_ip not in self._request_counts:
            self._request_counts[client_ip] = []

        # Clean old entries
        self._request_counts[client_ip] = [
            t for t in self._request_counts[client_ip] if t > window_start
        ]

        if len(self._request_counts[client_ip]) >= self._rate_limit:
            return False

        self._request_counts[client_ip].append(now)
        return True


# === Initialize asqav and attach middleware ===

def _init_asqav() -> Agent | None:
    """Initialize asqav and return the audit agent."""
    if ASQAV_API_KEY.startswith("sk_your"):
        logger.warning(
            "ASQAV_API_KEY not set - audit middleware disabled. "
            "Set ASQAV_API_KEY to enable governance."
        )
        return None

    try:
        asqav.init(api_key=ASQAV_API_KEY)
        agent = asqav.Agent.create("fastapi-governance-agent")
        app.add_middleware(AsqavAuditMiddleware, agent=agent)
        logger.info("asqav audit middleware attached to FastAPI app")
        return agent
    except AsqavError as exc:
        logger.warning("Failed to initialize asqav (fail-open): %s", exc)
        return None


_init_asqav()

# === AI Endpoints ===


@app.post("/api/chat")
async def chat(request: Request) -> dict[str, Any]:
    """AI chat endpoint - all requests are audited by asqav middleware."""
    body = await request.json()
    model = body.get("model", "gpt-4")
    prompt = body.get("prompt", "")

    # Simulate AI response (replace with actual model call)
    return {
        "model": model,
        "response": f"AI response to: {prompt[:50]}...",
        "status": "success",
    }


@app.post("/api/embeddings")
async def embeddings(request: Request) -> dict[str, Any]:
    """AI embeddings endpoint - all requests are audited by asqav middleware."""
    body = await request.json()
    model = body.get("model", "text-embedding-3-small")
    text = body.get("text", "")

    # Simulate embeddings response
    return {
        "model": model,
        "embedding_length": 1536,
        "status": "success",
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Health check endpoint - not audited (non-AI endpoint)."""
    return {"status": "healthy", "asqav": "active" if ASQAV_API_KEY else "inactive"}


# === Compliance Report Endpoint ===


@app.get("/api/compliance/report")
async def compliance_report() -> dict[str, Any]:
    """Generate a compliance report of recent AI activity."""
    return {
        "report_type": "ai_governance_summary",
        "period": "last_24_hours",
        "metrics": {
            "total_requests": "see_asqav_dashboard",
            "policy_denials": "see_asqav_dashboard",
            "avg_latency_ms": "see_asqav_dashboard",
        },
        "verification_url": "https://asqav.com/dashboard",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
