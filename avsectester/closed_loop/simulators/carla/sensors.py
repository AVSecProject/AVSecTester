"""CARLA sensor alignment, pinhole registration, depth decoding and owned SDK captures."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from avsectester.serialization import DEFAULT_REGISTRY


@dataclass(frozen=True)
class SensorCapture:
    """Owned bytes from a CARLA sensor capture; contains no simulator handle.

    ``raw_data`` retains the memoryview interface used by AVStack's LiDAR
    exporter. Coordinates remain in their original convention; the surrounding
    LidarData keeps its ``flipy`` setting. No serialization uses SDK pickle.
    """

    native_type: str
    raw_bytes: bytes
    frame: int
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_data(self) -> memoryview:
        """Native sensor payload before conversion to framework records."""
        return memoryview(self.raw_bytes)


def _sdk_capture(value: Any) -> SensorCapture:
    metadata = {}
    for name in ("width", "height", "fov", "channels", "horizontal_angle"):
        if hasattr(value, name):
            metadata[name] = getattr(value, name)
    transform = getattr(value, "transform", None)
    if transform is not None:
        metadata["transform"] = {
            "location": [transform.location.x, transform.location.y, transform.location.z],
            "rotation_pitch_yaw_roll": [transform.rotation.pitch, transform.rotation.yaw,
                                         transform.rotation.roll],
        }
    return SensorCapture(
        native_type=f"{type(value).__module__}.{type(value).__qualname__}",
        raw_bytes=bytes(value.raw_data), frame=int(value.frame), timestamp=float(value.timestamp),
        metadata=metadata,
    )


def _sdk_capture_data(value: Any) -> dict[str, Any]:
    """Encode a native SDK measurement as ordinary fields, without SDK reconstruction."""
    return asdict(_sdk_capture(value))


class SensorAlignmentError(RuntimeError):
    """A configured sensor has no capture for the current world frame."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = deepcopy(dict(report))
        super().__init__("Sensor capture is not aligned: " + "; ".join(report["failures"]))


_PINHOLE_ATTRIBUTES = {"lens_k": "0", "lens_kcube": "0", "lens_circle_multiplier": "0"}


def register_pinhole_camera():
    """Register the undistorted RGB sensor used with a depth camera, without importing CARLA eagerly.

    The class stays in this repository. The ordinary avcarla RGB sensor keeps its original
    behavior. Sensor factories reuse the registered subclass and its calibration.
    """
    from avcarla.config import CARLA
    from avcarla.sensors import CarlaRgbCamera

    existing = CARLA.module_dict.get("PinholeRgbCamera")
    if existing is not None:
        if existing.__module__ != __name__:
            raise ValueError("PinholeRgbCamera is already registered by another module")
        return existing

    @CARLA.register_module()
    class PinholeRgbCamera(CarlaRgbCamera):
        """An ideal pinhole RGB sensor for calibrated depth-based insertion."""

        def __init__(self, *args, do_spawn=True, **kwargs):
            super().__init__(*args, do_spawn=False, **kwargs)
            for name, value in _PINHOLE_ATTRIBUTES.items():
                self.bp.set_attribute(name, value)
            if do_spawn:
                self.spawn()

    return PinholeRgbCamera


def _use_pinhole_cameras(scenario):
    """Replace stock RGB cameras with the undistorted subclass; custom classes are kept."""
    for sensor in scenario.get("ego", {}).get("sensors", ()):
        if sensor.get("type") == "CarlaRgbCamera":
            sensor["type"] = "PinholeRgbCamera"
    return scenario


def _require_pinhole_sensor(sensor):
    """Reject camera geometry that the pinhole visibility provider cannot represent."""
    for name in _PINHOLE_ATTRIBUTES:
        value = float(sensor.attributes.get(name, -1 if name == "lens_k" else 0))
        if not np.isfinite(value) or value != 0:
            raise ValueError(
                f"Selection requires an undistorted camera ({name}=0), got {value}. "
                "Use PinholeRgbCamera or supply a custom calibrated visibility provider."
            )


def decode_carla_depth(image):
    """Decode CARLA's 24-bit depth image to forward camera depth in meters."""
    bgra = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    channels = bgra.astype(np.float64)
    return (channels[..., 2] + 256 * channels[..., 1] + 65536 * channels[..., 0]) * (
        1000.0 / 16777215
    )


DEFAULT_REGISTRY.register_native(
    SensorCapture.__module__, SensorCapture.__qualname__, asdict,
)


for _module in ("carla", "carla.libcarla"):
    for _name in ("Image", "LidarMeasurement", "SemanticLidarMeasurement", "RadarMeasurement"):
        DEFAULT_REGISTRY.register_native(
            _module, _name, _sdk_capture_data,
            snapshot=lambda value, capture_child: _sdk_capture(value),
        )


__all__ = [
    'SensorCapture',
    'SensorAlignmentError',
    'decode_carla_depth',
]
