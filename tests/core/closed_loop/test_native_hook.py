"""Native registration stays explicit and outside the generic hook dispatcher."""

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from avsectester.closed_loop.avstacks.native_hook import NativeHook
from avsectester.closed_loop.hooks import CallInputs
from avsectester.interventions import LegacyNativeHook, runtime_from_config


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


def test_legacy_native_registration_uses_only_declared_imports(registry, monkeypatch):
    imported = []
    monkeypatch.setattr("avsectester.interventions.import_module", imported.append)
    legacy = LegacyNativeHook({"type": "PhantomInjection"})
    legacy.reset(None)
    assert imported == []  # Native names do not secretly import an attack implementation.

    runtime = runtime_from_config({"hooks": [{
        "stage": "perception.post", "avstack": {"type": "CustomHook"},
        "imports": ["user.registration"],
    }]})
    runtime.hooks[0].handler.reset(None)
    assert imported == ["user.registration"]
    assert registry.build.call_args.args == ({"type": "CustomHook"},)


@pytest.mark.parametrize("imports", ["user.registration", None, [None], [""]])
def test_legacy_native_imports_reject_non_module_names(imports):
    with pytest.raises(ValueError, match="module names"):
        LegacyNativeHook({"type": "CustomHook"}, imports=imports)


def test_simulator_imports_do_not_load_legacy_renderers_or_concrete_attacks():
    code = '''
import importlib.abc
import sys
class NoLegacyRendering(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("avsectester.attacks", "avsectester.rendering.legacy_")) or fullname == "avsectester.rendering.compositor":
            raise AssertionError("Framework imported " + fullname)
sys.meta_path.insert(0, NoLegacyRendering())
import avsectester.closed_loop.simulators.carla
import avsectester.closed_loop.simulators.nurec
import avsectester.closed_loop.hooks
import avsectester.closed_loop.avstacks.native_hook
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
