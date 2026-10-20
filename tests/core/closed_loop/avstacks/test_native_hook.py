"""Native registration stays explicit and outside the generic hook dispatcher."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from avsectester.closed_loop.avstacks.native_hook import NativeHook
from avsectester.closed_loop.hooks import CallInputs
from pathlib import Path
from avsectester.experiment.config import _hooks


@pytest.fixture
def registry(monkeypatch):
    module = ModuleType("avstack.config")
    module.HOOKS = SimpleNamespace(build=Mock())
    monkeypatch.setitem(sys.modules, "avstack.config", module)
    return module.HOOKS


def test_native_hook_preserves_tuple_bridge_and_owns_one_native_lifecycle(registry):
    handler = Mock(return_value=(12,))
    registry.build.return_value = handler
    options = {"type": "RegisteredHook", "values": [1]}
    native = NativeHook(options)
    options["values"].append(2)
    native.reset(None)
    registry.build.assert_called_once_with({"type": "RegisteredHook", "values": [1]})
    assert native.transform(3, None) == 12
    native.close()
    handler.close.assert_called_once_with()
    assert native.native is None

    handler.return_value = ((1, 2), {"flag": True})
    pre = NativeHook({"type": "RegisteredHook"}, phase="pre")
    pre.reset(None)
    assert pre.transform(CallInputs((0,), {"flag": False}), None) == CallInputs(
        (1, 2), {"flag": True},
    )


def test_native_registration_uses_only_declared_imports(registry, monkeypatch):
    imported = []
    monkeypatch.setattr("avsectester.experiment.config.import_module", imported.append)
    native = NativeHook({"type": "CustomHook"})
    native.reset(None)
    assert imported == []
    hooks = _hooks([{
        "stage": "perception.post", "native_hook": {"type": "CustomHook"},
        "imports": ["user.registration"],
    }], Path("."))(0)
    hooks.hooks[0].handler.reset(None)
    assert imported == ["user.registration"]
    assert registry.build.call_args.args == ({"type": "CustomHook"},)
