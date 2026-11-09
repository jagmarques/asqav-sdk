"""Pin the public names and signatures of asqav and asqav.client.

The committed list is tests/fixtures/public_api_surface.json."""

from __future__ import annotations
import __future__

import importlib
import inspect
import json
import re
import typing
from pathlib import Path

import pytest

SNAPSHOT = Path(__file__).parent / "fixtures" / "public_api_surface.json"
MODULES = ("asqav", "asqav.client")
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_RELEASE = re.compile(r"(asqav-python/)[^ ]+")
_PLAIN_VALUES = (str, int, float, bool, tuple, list, dict, type(None))


def _signature(obj: object) -> str:
    try:
        return _ADDRESS.sub(" at 0x?", str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return "<no signature>"


def _value(obj: object) -> str:
    if isinstance(obj, (frozenset, set)):
        return f"{type(obj).__name__}{sorted(map(repr, obj))}"
    if isinstance(obj, _PLAIN_VALUES):
        return _RELEASE.sub(r"\1<release>", f"{type(obj).__name__} {obj!r}")
    return type(obj).__qualname__


def _member(obj: object) -> str:
    if isinstance(obj, property):
        return "property"
    if isinstance(obj, (classmethod, staticmethod)):
        return f"{type(obj).__name__} {_signature(obj.__func__)}"
    if inspect.isroutine(obj):
        return _signature(obj)
    return _value(obj)


def _owner(cls: type, name: str) -> type:
    return next(klass for klass in cls.__mro__ if name in vars(klass))


def _owned(cls: type, name: str) -> bool:
    return _owner(cls, name).__module__.startswith("asqav")


def _describe(obj: object) -> object:
    if inspect.isclass(obj):
        members = {
            name: _member(inspect.getattr_static(obj, name))
            for name in dir(obj)
            if not name.startswith("_") and _owned(obj, name)
        }
        init = _signature(obj) if _owned(obj, "__init__") else "<inherited>"
        return {"class": init, "members": members}
    if inspect.isroutine(obj):
        return {"function": _signature(obj)}
    return {"value": _value(obj)}


def _is_public(name: str, obj: object) -> bool:
    if name.startswith("_") or inspect.ismodule(obj):
        return False
    if getattr(typing, name, None) is obj or getattr(__future__, name, None) is obj:
        return False
    if inspect.isclass(obj) or inspect.isroutine(obj):
        return obj.__module__.startswith("asqav")
    return True


def surface(module_name: str) -> dict[str, object]:
    module = importlib.import_module(module_name)
    return {
        name: _describe(getattr(module, name))
        for name in sorted(dir(module))
        if _is_public(name, getattr(module, name))
    }


@pytest.mark.parametrize("module_name", MODULES)
def test_public_surface_matches_snapshot(module_name: str) -> None:
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))[module_name]
    assert surface(module_name) == expected


def test_negctl_surface_sees_a_dropped_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEGCTL: a name deleted from asqav.client shows in surface()."""
    client = importlib.import_module("asqav.client")
    monkeypatch.delattr(client, "verify_signature")
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["asqav.client"]
    assert "verify_signature" not in surface("asqav.client")
    assert surface("asqav.client") != expected


def test_negctl_surface_sees_a_changed_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEGCTL: a verify_signature with another signature shows in surface()."""
    client = importlib.import_module("asqav.client")

    def verify_signature(signature_id: str, strict: bool) -> None:
        return None

    verify_signature.__module__ = "asqav.client"
    monkeypatch.setattr(client, "verify_signature", verify_signature)
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["asqav.client"]
    assert surface("asqav.client")["verify_signature"] != expected["verify_signature"]
