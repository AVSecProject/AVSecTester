"""Register avstack's CARLA dataset with the installed MMDetection3D package."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def register_carla_dataset() -> type:
    """Return the CARLA dataset class, registering its vendored declaration if needed.

    CARLA-trained PointPillars configurations require this class even for inference,
    where MMDetection3D builds a lazy dataset to obtain metadata. Existing registrations
    are preserved. Only the dataset module is loaded, leaving the installed detector
    package and its CUDA operators unchanged.
    """
    from mmdet3d.registry import DATASETS

    existing = DATASETS.get("CarlaDataset")
    if existing is not None:
        return existing

    # The installed package supplies Det3DDataset and the module's relative imports.
    importlib.import_module("mmdet3d.datasets")
    name = "mmdet3d.datasets.carla_dataset"
    if importlib.util.find_spec(name) is not None:
        importlib.import_module(name)
    else:
        import avstack

        source = (
            Path(avstack.__file__).resolve().parents[1]
            / "third_party/mmdetection3d/mmdet3d/datasets/carla_dataset.py"
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"CARLA dataset declaration not found: {source}. Initialize the nested "
                "third_party/mmdetection3d submodule in third_party/avstack-core."
            )
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise

    registered = DATASETS.get("CarlaDataset")
    if registered is None:
        raise RuntimeError(f"{name} did not register CarlaDataset")
    return registered
