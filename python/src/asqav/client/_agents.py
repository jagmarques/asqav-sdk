"""Module-level agent access: the global agent, secure() and govern()."""

from __future__ import annotations

import functools
import os
import sys
import threading
from typing import Any, cast

from . import _transport
from ._agent import Agent
from ._common import F, _parse_timestamp


def emergency_halt(org_id: str, reason: str = "emergency") -> dict:
    """Raise an org-wide emergency halt. Use in emergencies only.

    Calls POST /orgs/{org_id}/emergency-halt. Requires an admin/owner session
    token (ASQAV_SESSION_TOKEN), not an agent API key.
    """
    return _transport._post(f"/orgs/{org_id}/emergency-halt", {"reason": reason})


def govern(
    api_key: str | None = None,
    agent_name: str = "default-agent",
    agent_id: str | None = None,
    *,
    algorithm: str = "ml-dsa-65",
    capabilities: list[str] | None = None,
    base_url: str | None = None,
    mode: str = "auto",
    org_salt: bytes | None = None,
) -> "Agent":
    """One-call governance entrypoint: init + Agent.create (or Agent.get).

    Composes init() and Agent.create() so the minimum setup is one call.
    Pass api_key (or set ASQAV_API_KEY) and an agent name; everything else
    is optional. Existing agents can be retrieved by passing agent_id.

    Args:
        api_key: Asqav API key. Defaults to ASQAV_API_KEY env var.
        agent_name: Human-readable name for the new agent.
        agent_id: If supplied, retrieves the existing agent instead of creating.
        algorithm: Signing algorithm. Defaults to "ml-dsa-65" (FIPS 204).
        capabilities: Capability strings for the new agent.
        base_url: Override API base URL (for testing).
        mode: Wire mode - "auto" (default), "hash-only", or "full-payload".
        org_salt: Optional 32-byte HMAC salt for hash-only mode. See
            init() for the ``keyed_digest_requires_context`` caveat.

    Returns:
        A ready-to-use Agent instance.

    Raises:
        AuthenticationError: If no API key is provided.
        APIError: If agent creation or retrieval fails.

    Example:
        import asqav

        agent = asqav.govern(api_key="sk_...", agent_name="my-agent")
        sig = agent.sign("api:call", {"model": "gpt-4"})
    """
    _transport.init(api_key=api_key, base_url=base_url, mode=mode, org_salt=org_salt)
    if agent_id is not None:
        return Agent.get(agent_id)
    return Agent.create(agent_name, algorithm=algorithm, capabilities=capabilities)


# Global agent for decorators
_global_agent: Agent | None = None


_global_agent_lock = threading.Lock()


def get_agent() -> Agent:
    """Get the global agent for decorator use; raises AsqavError if unset."""
    global _global_agent
    if _global_agent is None:
        # Serialise auto-creation: racing first calls would register duplicates.
        with _global_agent_lock:
            if _global_agent is None:
                name = _auto_generate_name()
                _global_agent = Agent.create(name)
    return _global_agent


def list_agents() -> list[Agent]:
    """Return every agent visible to the current API key.

    Public surface for the same data ``asqav agents list`` shows. Use this
    in dashboards, scripts, and notebooks instead of reaching into
    ``asqav.client._get('/agents')``.
    """
    data = _transport._get("/agents")
    raw = data if isinstance(data, list) else data.get("agents", [])
    out: list[Agent] = []
    for a in raw:
        out.append(
            Agent(
                agent_id=a["agent_id"],
                name=a["name"],
                public_key=a.get("public_key", ""),
                key_id=a.get("key_id", ""),
                algorithm=a.get("algorithm", ""),
                capabilities=a.get("capabilities", []),
                created_at=_parse_timestamp(a.get("created_at", 0)),
            )
        )
    return out


    # Generate an agent name from environment.
def _auto_generate_name() -> str:
    env_name = os.environ.get("ASQAV_AGENT_NAME")
    if env_name:
        return env_name

    if hasattr(sys, "argv") and sys.argv:
        import pathlib

        script_path = pathlib.Path(sys.argv[0])
        return f"agent-{script_path.stem}"

    return "asqav-agent"


def secure(func: F) -> F:
    """Decorator to secure function calls with cryptographic signing.

    Wraps the function in an asqav session, signing the call as an action.
    All ML-DSA cryptography happens server-side.

    Trust contract (fail-closed): the call is signed BEFORE the wrapped body runs.
    If signing is refused (a blocked policy, a non-2xx response, or a network error),
    the error propagates and the body never executes. Unsigned actions abort by
    default - the caller does not need to inspect a return value to stay safe.

    Args:
        func: The function to secure.

    Returns:
        The wrapped function.

    Example:
        import asqav

        asqav.init(api_key="sk_...")

        @asqav.secure
        def process_data(data: dict) -> dict:
            # This call is signed with cryptographic proof
            return {"processed": True}
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        agent = get_agent()

        # Start session if not active
        if agent._session_id is None:
            agent.start_session()

        # Log the function call
        agent.sign(
            action_type="function:call",
            context={
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            },
        )

        try:
            result = func(*args, **kwargs)

            # Log success
            agent.sign(
                action_type="function:result",
                context={
                    "function": func.__name__,
                    "success": True,
                },
                result=result,
            )

            return result

        except Exception as e:
            # Log error
            agent.sign(
                action_type="function:error",
                context={
                    "function": func.__name__,
                    "error": str(e),
                },
                result=str(e),
            )
            raise

    return cast(F, wrapper)


def secure_async(func: F) -> F:
    """Async version of the @secure decorator.

    Args:
        func: The async function to secure.

    Returns:
        The wrapped async function.

    Example:
        import asqav

        asqav.init(api_key="sk_...")

        @asqav.secure_async
        async def fetch_data(url: str) -> dict:
            # This call is signed with cryptographic proof
            return {"data": "..."}
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        agent = get_agent()

        if agent._session_id is None:
            agent.start_session()

        agent.sign(
            action_type="function:call",
            context={
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            },
        )

        try:
            result = await func(*args, **kwargs)

            agent.sign(
                action_type="function:result",
                context={
                    "function": func.__name__,
                    "success": True,
                },
                result=result,
            )

            return result

        except Exception as e:
            agent.sign(
                action_type="function:error",
                context={
                    "function": func.__name__,
                    "error": str(e),
                },
                result=str(e),
            )
            raise

    return cast(F, wrapper)
