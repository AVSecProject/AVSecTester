"""Execution backends — thin adapters over the avstack stack.

- ``CarlaBackend``   -> lib-avstack-carla (avcarla): closed-loop CARLA 0.9.13.
- ``DatasetBackend`` -> avstack-api (avapi): offline KITTI / nuScenes eval + replay.
- perception wrapping -> avstack-core MMDetObjectDetector3D (mmdetection3d).

Imports are lazy so ``avsectester.core`` stays importable without the heavy stack.
"""

__all__ = ["CarlaBackend", "DatasetBackend", "MockBackend"]


def __getattr__(name: str):  # PEP 562 lazy import
    if name == "CarlaBackend":
        from .carla_backend import CarlaBackend

        return CarlaBackend
    if name == "DatasetBackend":
        from .dataset_backend import DatasetBackend

        return DatasetBackend
    if name == "MockBackend":
        from .mock_backend import MockBackend

        return MockBackend
    raise AttributeError(name)
