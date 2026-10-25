"""Each framework extra names its framework and its adapter works on it.
NEGCTL: the base install carries no framework and a stub is not an install."""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from importlib.machinery import ModuleSpec
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"

# extra -> (distribution the extra must require, importable framework module)
FRAMEWORKS = {
    "llamaindex": ("llama-index-core", "llama_index.core"),
    "dspy": ("dspy", "dspy"),
    "smolagents": ("smolagents", "smolagents"),
    "crewai": ("crewai", "crewai"),
}


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _requirement_names(requirements: list[str]) -> set[str]:
    return {re.split(r"[<>=!~\[; ]", r, maxsplit=1)[0].lower() for r in requirements}


def _module_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ValueError, ImportError):
        return False


def _unavailable(module: str, reason: str) -> None:
    """Fail when ASQAV_EXTRA_REQUIRED names the module, else skip."""
    if os.environ.get("ASQAV_EXTRA_REQUIRED") == module:
        pytest.fail(f"{module} is required but {reason}")
    pytest.skip(f"{module} {reason}")


def _real_framework(module: str) -> ModuleType:
    """Import the installed framework; a stub or absent one is unavailable."""
    if not _module_installed(module):
        _unavailable(module, "is not installed")
    found = importlib.import_module(module)
    if getattr(found, "__file__", None) is None:
        _unavailable(module, "is a stub, not an install")
    return found


def _adapter(extra: str) -> ModuleType:
    """Import the adapter fresh, not bound to an earlier test's stub."""
    name = f"asqav.extras.{extra}"
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.fixture()
def agent_calls():
    """Patch the Agent behind the adapter base; yield the signed actions."""
    signed: list[tuple[str, dict]] = []
    agent = MagicMock()
    agent.sign.side_effect = lambda action, context=None, **kw: signed.append((action, context))
    with (
        patch("asqav.client._api_key", "sk_test"),
        patch("asqav.extras._base.Agent") as agent_cls,
    ):
        agent_cls.create.return_value = agent
        yield signed


@pytest.mark.parametrize("extra", sorted(FRAMEWORKS))
def test_extra_requires_its_framework(extra):
    distribution, _ = FRAMEWORKS[extra]
    extras = _project()["optional-dependencies"]
    assert distribution in _requirement_names(extras[extra])


@pytest.mark.parametrize("extra", sorted(FRAMEWORKS))
def test_extra_floor_is_declared(extra):
    distribution, _ = FRAMEWORKS[extra]
    requirement = next(
        r for r in _project()["optional-dependencies"][extra]
        if _requirement_names([r]) == {distribution}
    )
    assert re.search(r">=\s*\d", requirement), requirement


def test_base_install_has_no_framework_negctl():
    """NEGCTL: the base dependencies and the all extra name no framework."""
    project = _project()
    frameworks = {distribution for distribution, _ in FRAMEWORKS.values()}
    assert project["dependencies"] == []
    assert not frameworks & _requirement_names(project["optional-dependencies"]["all"])


def _stub(name: str) -> ModuleType:
    stub = ModuleType(name)
    stub.__spec__ = ModuleSpec(name, None)
    return stub


def test_required_framework_fails_when_absent_negctl(monkeypatch):
    """NEGCTL: with the env set, an absent module fails; unset, it skips."""
    name = "asqav_absent_framework"
    with pytest.raises(pytest.skip.Exception, match="not installed"):
        _real_framework(name)
    monkeypatch.setenv("ASQAV_EXTRA_REQUIRED", name)
    with pytest.raises(pytest.fail.Exception, match="not installed"):
        _real_framework(name)


def test_required_framework_fails_on_a_stub_negctl(monkeypatch):
    """NEGCTL: a stub with a spec but no file skips, and fails when required."""
    name = "asqav_stub_framework"
    with patch.dict(sys.modules, {name: _stub(name)}):
        with pytest.raises(pytest.skip.Exception, match="stub"):
            _real_framework(name)
        monkeypatch.setenv("ASQAV_EXTRA_REQUIRED", name)
        with pytest.raises(pytest.fail.Exception, match="stub"):
            _real_framework(name)


def test_llamaindex_handler_signs_on_real_callback_manager(agent_calls):
    _real_framework(FRAMEWORKS["llamaindex"][1])
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType

    adapter = _adapter("llamaindex")
    handler = adapter.AsqavLlamaIndexHandler()
    assert isinstance(handler, BaseCallbackHandler)
    manager = CallbackManager([handler])
    with manager.event(CBEventType.QUERY, payload={"query_str": "q"}):
        pass
    assert [a for a, _ in agent_calls] == [
        f"llamaindex.{CBEventType.QUERY.value}.start",
        f"llamaindex.{CBEventType.QUERY.value}.end",
    ]


def test_dspy_callback_signs_a_real_predict(agent_calls):
    dspy = _real_framework(FRAMEWORKS["dspy"][1])
    from dspy.utils import DummyLM

    adapter = _adapter("dspy")
    callback = adapter.AsqavDSPyCallback()
    with dspy.context(lm=DummyLM([{"answer": "x"}]), callbacks=[callback]):
        dspy.Predict("question -> answer")(question="q")
    actions = [a for a, _ in agent_calls]
    assert "dspy.module.start" in actions
    assert "dspy.module.end" in actions
    assert "dspy.lm.start" in actions


def test_smolagents_hook_signs_a_real_tool(agent_calls):
    smolagents = _real_framework(FRAMEWORKS["smolagents"][1])

    class Shout(smolagents.Tool):
        name = "shout"
        description = "Upper-case the text."
        inputs = {"text": {"type": "string", "description": "the text"}}
        output_type = "string"

        def forward(self, text: str) -> str:
            return text.upper()

    shout = Shout()

    adapter = _adapter("smolagents")
    hook = adapter.AsqavSmolagentsHook()
    wrapped = hook.wrap_tool(shout)
    assert wrapped(text="a") == "A"
    assert [a for a, _ in agent_calls] == ["tool:start", "tool:end"]


def test_crewai_governance_wires_a_real_crew(agent_calls, monkeypatch):
    crewai = _real_framework(FRAMEWORKS["crewai"][1])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from asqav.extras.crewai import enable_crew_governance

    worker = crewai.Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
    task = crewai.Task(description="d", expected_output="e", agent=worker)
    crew = crewai.Crew(agents=[worker], tasks=[task])
    hook = enable_crew_governance(crew)
    assert crew.step_callback is not None
    assert crew.task_callback is not None
    crew.task_callback(MagicMock(description="d", raw="out"))
    assert [a for a, _ in agent_calls] == ["task:complete"]
    assert enable_crew_governance(crew) is hook
