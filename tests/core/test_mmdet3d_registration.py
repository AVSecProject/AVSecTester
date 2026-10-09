"""Custom dataset registration without installing the detector stack."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from avsectester.stacks.mmdet3d import register_carla_dataset


@pytest.fixture
def detector_packages(monkeypatch, tmp_path):
    class Registry(dict):
        def register_module(self):
            def register(cls):
                self[cls.__name__] = cls
                return cls
            return register

    registry = Registry()
    installed = tmp_path / "installed" / "datasets"
    installed.mkdir(parents=True)
    checkout = tmp_path / "avstack-core"
    vendored = checkout / "third_party/mmdetection3d/mmdet3d/datasets"
    vendored.mkdir(parents=True)
    packages = {
        "mmdet3d": {},
        "mmdet3d.registry": {"DATASETS": registry},
        "mmdet3d.datasets": {"__path__": [str(installed)]},
        "mmdet3d.datasets.det3d_dataset": {"Det3DDataset": type("Det3DDataset", (), {})},
        "avstack": {"__file__": str(checkout / "avstack/__init__.py")},
    }
    for name, attrs in packages.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    # Isolate the dynamically imported module from other tests in this process.
    monkeypatch.delitem(sys.modules, "mmdet3d.datasets.carla_dataset", raising=False)
    yield SimpleNamespace(registry=registry, installed=installed, vendored=vendored)
    sys.modules.pop("mmdet3d.datasets.carla_dataset", None)


DECLARATION = """from mmdet3d.registry import DATASETS
from .det3d_dataset import Det3DDataset
@DATASETS.register_module()
class CarlaDataset(Det3DDataset):
    METAINFO = {'classes': ('car', 'bicycle', 'truck', 'motorcycle')}
"""


@pytest.mark.parametrize("location", ["installed", "vendored"])
def test_registration_uses_installed_base_and_is_repeatable(detector_packages, location):
    source = getattr(detector_packages, location) / "carla_dataset.py"
    source.write_text(DECLARATION)
    paths_before = list(sys.modules["mmdet3d.datasets"].__path__)
    registered = register_carla_dataset()
    assert registered is register_carla_dataset()
    assert registered is detector_packages.registry["CarlaDataset"]
    assert registered.__bases__ == (sys.modules["mmdet3d.datasets.det3d_dataset"].Det3DDataset,)
    assert sys.modules["mmdet3d.datasets"].__path__ == paths_before


def test_registration_preserves_user_class(detector_packages):
    custom = type("UserCarlaDataset", (), {})
    detector_packages.registry["CarlaDataset"] = custom
    assert register_carla_dataset() is custom


def test_missing_declaration_explains_required_submodule(detector_packages):
    with pytest.raises(FileNotFoundError, match="Initialize the nested"):
        register_carla_dataset()


def test_failed_registration_does_not_leave_a_partial_import(detector_packages):
    source = detector_packages.vendored / "carla_dataset.py"
    source.write_text("raise RuntimeError('invalid declaration')")
    with pytest.raises(RuntimeError, match="invalid declaration"):
        register_carla_dataset()
    assert "mmdet3d.datasets.carla_dataset" not in sys.modules
    source.write_text(DECLARATION)
    assert register_carla_dataset() is detector_packages.registry["CarlaDataset"]


@pytest.mark.parametrize("perception, expected", [
    ({"type": "MMDetObjectDetector3D", "dataset": "carla-vehicle"}, True),
    ({"type": "MMDetObjectDetector3D", "dataset": "carla-vehicle", "deploy": True}, False),
    ({"type": "MMDetObjectDetector3D", "dataset": "kitti"}, False),
    ({"type": "Passthrough3DObjectDetector"}, False),
])
def test_stack_registers_only_when_needed_and_before_model_construction(monkeypatch, perception, expected):
    from avsectester.stacks import mmdet3d, modular

    calls = []
    monkeypatch.setattr(modular, "_register_avstack_modules", lambda: None)
    monkeypatch.setattr(mmdet3d, "register_carla_dataset", lambda: calls.append("register"))
    config_module = ModuleType("avstack.config")
    config_module.PIPELINE = SimpleNamespace(build=lambda cfg: calls.append("build"))
    monkeypatch.setitem(sys.modules, "avstack.config", config_module)
    modular.ModularAVStack({"perception": perception})
    assert calls == (["register", "build"] if expected else ["build"])
