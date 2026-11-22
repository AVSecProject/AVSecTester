"""CARLA sensor alignment, pinhole registration, depth decoding and owned SDK captures."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from inspect import Parameter, signature
from typing import Any

import numpy as np

from avsectester.serialization import DEFAULT_REGISTRY


@dataclass(frozen=True)
class SensorCapture:
    """Owned bytes from a CARLA sensor capture. It contains no simulator handle.

    ``raw_data`` retains the memoryview interface used by AVStack's LiDAR
    exporter. Coordinates remain in their original convention. The surrounding
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


def validate_native_sensor_tick(value) -> None:
    if value != 0.:
        raise ValueError("CARLA sensors acquire every physics tick. Set sensor_tick to 0 and "
                         "configure acquisition with runs.timing.sensors")


def _native_attribute_values(attributes):
    return {name: str(value).lower() if isinstance(value, bool) else str(value)
            for name, value in attributes.items()}


def _set_blueprint_attributes(blueprint, attributes):
    for name, value in _native_attribute_values(attributes).items():
        if not blueprint.has_attribute(name):
            raise ValueError(f"CARLA blueprint {blueprint.id!r} has no attribute {name!r}")
        blueprint.set_attribute(name, value)


def _verify_blueprint_attributes(actor, attributes, description):
    reported = actor.attributes
    for name, value in _native_attribute_values(attributes).items():
        if reported.get(name) != value:
            raise ValueError(
                f"CARLA {description} attribute {name!r} does not match its configured value "
                f"{value!r}. Actor reported {reported.get(name)!r}.")


class _BlueprintWorld:
    def __init__(self, world, attributes):
        self.world = world
        self.attributes = attributes

    def __getattr__(self, name):
        return getattr(self.world, name)

    def try_spawn_actor(self, blueprint, transform, *args, **kwargs):
        _set_blueprint_attributes(blueprint, self.attributes)
        return self.world.try_spawn_actor(blueprint, transform, *args, **kwargs)


class _BlueprintClient:
    def __init__(self, client, attributes):
        self.client = client
        self.world = _BlueprintWorld(client.world, attributes)

    def __getattr__(self, name):
        return getattr(self.client, name)


def _configured_type(name, *, sensor):
    from avcarla.config import CARLA

    registered_name = f"Configured{name}"
    existing = CARLA.module_dict.get(registered_name)
    if existing is not None:
        if existing.__module__ != __name__:
            raise ValueError(f"{registered_name} is already registered by another module")
        return registered_name
    original = CARLA.module_dict[name]
    if getattr(original, "_native_blueprint_wrapper", False):
        return name
    if sensor:
        constructor_options = {
            parameter.name
            for base in original.__mro__
            for parameter in signature(base.__init__).parameters.values()
            if parameter.name != "self" and parameter.kind in (
                Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        }

        class ConfiguredSensor(original):
            _native_blueprint_wrapper = True

            def __init__(self, *args, blueprint_attributes=None, ros=False, do_spawn=True, **kwargs):
                self._publish_ros = ros
                attributes = blueprint_attributes or {}
                conflicts = attributes.keys() & constructor_options
                if conflicts:
                    raise ValueError(
                        f"CARLA sensor blueprint_attributes contains constructor options "
                        f"{sorted(conflicts)}. Use sensor options for these values because "
                        "the constructor also determines calibration."
                    )
                validate_native_sensor_tick(float(attributes.get("sensor_tick", 0.)))
                super().__init__(*args, do_spawn=False, **kwargs)
                _set_blueprint_attributes(self.bp, attributes)
                if do_spawn:
                    self.spawn()

            def spawn(self):
                super().spawn()
                if self._publish_ros:
                    try:
                        self.object.enable_for_ros()
                    except BaseException:
                        self.object.destroy()
                        raise

        configured = ConfiguredSensor
    else:
        class ConfiguredActor(original):
            _native_blueprint_wrapper = True

            def __init__(self, *args, client, blueprint_attributes=None, **kwargs):
                attributes = blueprint_attributes or {}
                super().__init__(*args, client=_BlueprintClient(client, attributes), **kwargs)
                # Read back from the spawned actor, independently of how the base class spawned it.
                try:
                    _verify_blueprint_attributes(self.actor, attributes, "ego")
                except ValueError:
                    self.destroy()
                    raise

        configured = ConfiguredActor
    CARLA.register_module(name=registered_name)(configured)
    return registered_name


def configure_recipe_blueprints(scenario):
    """Register recipe wrappers that apply native attributes before actors are spawned."""
    ego = scenario["ego"]
    if "blueprint_attributes" in ego:
        ego["type"] = _configured_type(ego["type"], sensor=False)
    for specification in ego.get("sensors", ()):
        if "blueprint_attributes" in specification or specification.get("ros", False):
            specification["type"] = _configured_type(specification["type"], sensor=True)
    return scenario


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
    """Replace stock RGB cameras with the undistorted subclass. Custom classes are kept."""
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
            snapshot=lambda value, capture_child: capture_child(_sdk_capture(value)),
        )


__all__ = [
    'SensorCapture',
    'SensorAlignmentError',
    'decode_carla_depth',
]
