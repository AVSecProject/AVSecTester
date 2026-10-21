"""Explicit data-only RunLog codecs for the supported driving adapters.

Creating the registry never imports AVStack, AlpaSim, Torch or a simulator. Types
are registered from already loaded adapter modules, including modules loaded by
a model's first reset. Unknown objects still fail instead of exposing arbitrary
attributes or native handles. Decoding deliberately returns ordinary data.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
import sys
from typing import Any

from avsectester.runlog.log import CodecRegistry


@dataclass(frozen=True)
class SensorCapture:
    """Private bytes from a CARLA sensor capture; contains no simulator handle.

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


def _snapshot_sensor(value: Any, capture_child: Callable[[Any], Any]) -> Any:
    result = copy(value)
    # The allowlist is also used by persistence. Copy wrappers and references,
    # but snapshot the explicitly supported SDK payload through its own codec.
    for name, child in _sensor(value).items():
        if name != "native_type":
            setattr(result, name, capture_child(child))
    # These are optional derived image caches owned by native wrappers.
    for name in ("_rgb_image", "depth_in_meters"):
        if hasattr(value, name):
            setattr(result, name, capture_child(getattr(value, name)))
    return result


def _snapshot_native(value: Any, capture_child: Callable[[Any], Any]) -> Any:
    """Keep native in-memory behavior while capturing children through adapter codecs.

    ``deepcopy`` of an outer DataContainer bypasses the ReferenceFrame codec and follows
    every reverse reference in the live global frame. Native record fields are captured
    recursively here instead. Persistence continues to use explicit field allowlists.
    """
    result = copy(value)
    for name, child in vars(value).items():
        vars(result)[name] = capture_child(child)
    return result


def _snapshot_reference(value: Any, capture_child: Callable[[Any], Any]) -> Any:
    """Copy only an ancestor chain; descendants and integrated caches are not pose data."""
    result = copy(value)
    state = vars(result)
    for name, child in vars(value).items():
        if name == "_point_from":
            state[name] = []
        elif name == "_global_integrated":
            state[name] = None
        else:
            state[name] = capture_child(child)
    return result


def _fields(value: Any, *names: str) -> dict[str, Any]:
    """Copy an explicit field allowlist, never an object's attribute dictionary."""
    return {
        "native_type": f"{type(value).__module__}.{type(value).__qualname__}",
        **{name: getattr(value, name) for name in names},
    }


def _quaternion(value: Any) -> list[float]:
    return [float(value.w), float(value.x), float(value.y), float(value.z)]


def _reference(value: Any) -> dict[str, Any]:
    result = _fields(value, "reference", "handedness", "n_prec", "from_frame", "to_frame",
                     "timestamp")
    result.update({name: getattr(value, name).tolist() for name in ("x", "v", "acc")})
    result.update(q_wxyz=_quaternion(value.q), angular_wxyz=_quaternion(value.ang))
    # Never follow _point_from: it contains reverse links and potentially an
    # entire live sensor tree. The parent chain alone defines the coordinates.
    return result


def _vector(value: Any) -> dict[str, Any]:
    return {**_fields(value, "reference", "n_prec"), "x": value.x.tolist()}


def _rotation(value: Any) -> dict[str, Any]:
    return {**_fields(value, "reference", "n_prec"), "q_wxyz": _quaternion(value.q)}


def _box3d(value: Any) -> dict[str, Any]:
    return {**_fields(value, "ID", "obj_type", "position", "attitude", "where_is_t"),
            "hwl": [float(value.h), float(value.w), float(value.l)]}


def _camera(value: Any) -> dict[str, Any]:
    return {**_fields(value, "reference", "fov_horizontal", "fov_vertical", "square_pixels",
                      "channel_order"),
            "P": value.P.tolist(), "img_shape": list(value.img_shape)}


def _semantic_camera(value: Any) -> dict[str, Any]:
    return {**_camera(value), "colors": value.colors,
            "tags": [{"id": key, "name": name} for key, name in value.tags.items()]}


def _sensor(value: Any) -> dict[str, Any]:
    result = _fields(value, "timestamp", "frame", "source_ID", "source_name",
                     "source_identifier", "data", "calibration")
    # These are declared SensorData/subclass fields, not arbitrary kwargs.
    for name in ("depth", "in_front", "levar", "encoding", "n_features", "flipy"):
        if hasattr(value, name):
            result[name] = getattr(value, name)
    # Image and point arrays stay arrays: RunLog places them in payload shards.
    return result


def _detection(value: Any) -> dict[str, Any]:
    return {**_fields(value, "ID", "source_identifier", "obj_type", "score", "data",
                      "reference"), "noise": value.noise.tolist()}


def _track(value: Any) -> dict[str, Any]:
    return {
        **_fields(value, "ID", "obj_type", "t0", "t", "dt_coast", "n_updates", "n_missed",
                  "score", "reference", "attitude"),
        "x": value.x.tolist(), "P": value.P.tolist(),
    }


def _box_track(value: Any) -> dict[str, Any]:
    return {**_track(value), "box": value.box, "velocity": value.velocity}


def _waypoints(value: Any) -> dict[str, Any]:
    # Reading the heap must not pop entries or alter the planner's state.
    entries = sorted(value.heap, key=lambda pair: pair[0])
    return {**_fields(value, "min_waypoints", "dist_arrived", "max_dist", "max_size"),
            "waypoints": [{"priority": value.mult * priority, "waypoint": item}
                          for priority, item in entries]}


def _model_prediction(value: Any) -> dict[str, Any]:
    # NVIDIA AlpaSim models/base.py, inspected at
    # affc2eab209fa43bdfa2f26c0f8d437922d78a68. Arrays remain separate payloads.
    result = _fields(value, "candidate_positions", "candidate_rotations", "selected_index",
                     "reasoning_text", "selected_plan")
    for name in ("model_t0_us", "pose_local_to_rig_t0", "waypoint_timestamps_us"):
        result[name] = getattr(value, name, None)
    return result


def _alpasim_pose(value: Any) -> dict[str, Any]:
    position, quaternion = value.to_proto()
    return {"position": list(position), "q_wxyz": list(quaternion)}


def _alpasim_trajectory(value: Any) -> dict[str, Any]:
    # AlpaSim geometry uses scipy's xyzw quaternion convention.
    return {"timestamps_us": value.timestamps_us, "positions": value.positions,
            "quaternions_xyzw": value.quaternions}


_SPECS: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    ("avstack.datastructs", "DataContainer",
     lambda v: _fields(v, "frame", "timestamp", "source_identifier", "source_reference", "data")),
    ("avstack.geometry.refchoc", "ReferenceFrame", _reference),
    ("avstack.geometry.refchoc", "PassiveReferenceFrame",
     lambda v: _fields(v, "frame_id", "timestamp")),
    ("avstack.geometry.refchoc", "Vector", _vector),
    ("avstack.geometry.refchoc", "Rotation", _rotation),
    ("avstack.geometry.datastructs", "Pose", lambda v: _fields(v, "position", "attitude")),
    ("avstack.geometry.datastructs", "Twist", lambda v: _fields(v, "linear", "angular")),
    ("avstack.geometry.datastructs", "PointMatrix2D", lambda v: _fields(v, "x", "calibration")),
    ("avstack.geometry.datastructs", "PointMatrix3D", lambda v: _fields(v, "x", "calibration")),
    ("avstack.geometry.bbox", "Box3D", _box3d),
    ("avstack.geometry.bbox", "Box2D",
     lambda v: {**_fields(v, "ID", "obj_type", "calibration"), "xyxy": v.box2d.tolist()}),
    ("avstack.geometry.bbox", "SegMask2D", lambda v: _fields(v, "ID", "mask", "calibration")),
    ("avstack.environment.objects", "ObjectState",
     lambda v: _fields(v, "ID", "obj_type", "score", "t", "position", "velocity", "acceleration",
                      "attitude", "angular_velocity", "box", "occlusion", "visible_fraction")),
    ("avstack.calibration", "Calibration", lambda v: _fields(v, "reference")),
    ("avstack.calibration", "CameraCalibration", _camera),
    ("avstack.calibration", "SemanticSegmentationCalibration", _semantic_camera),
    ("avstack.calibration", "RadarCalibration",
     lambda v: _fields(v, "reference", "fov_horizontal", "fov_vertical")),
    ("avstack.sensors", "SensorData", _sensor),
    ("avstack.modules.perception.detections", "Detection_", _detection),
    ("avstack.modules.tracking.tracks", "TrackBase", _track),
    ("avstack.modules.tracking.tracks", "BasicBoxTrack2D", _box_track),
    ("avstack.modules.tracking.tracks", "BasicBoxTrack3D", _box_track),
    ("avstack.modules.tracking.tracks", "BasicJointBoxTrack",
     lambda v: _fields(v, "track_2d", "track_3d")),
    ("avstack.modules.tracking.tracks", "GroupTrack", lambda v: _fields(v, "state", "members")),
    ("avstack.modules.planning.types", "WaypointPlan", _waypoints),
    ("avstack.modules.planning.types", "Waypoint",
     lambda v: _fields(v, "target_point", "target_speed")),
    ("avstack.modules.control.types", "VehicleControlSignal",
     lambda v: _fields(v, "timestamp", "throttle", "steer", "brake", "hand_brake", "reverse",
                      "manual_gear_shift", "bounded")),
    ("alpasim_driver.models.base", "ModelPrediction", _model_prediction),
    ("alpasim_driver.models.base", "CameraFrame", lambda v: _fields(v, "timestamp_us", "image")),
    ("alpasim_utils.geometry", "Pose", _alpasim_pose),
    ("alpasim_utils.geometry", "Trajectory", _alpasim_trajectory),
    ("carla", "Image", _sdk_capture),
    ("carla", "LidarMeasurement", _sdk_capture),
    ("carla", "SemanticLidarMeasurement", _sdk_capture),
    ("carla", "RadarMeasurement", _sdk_capture),
)


class _AdapterCodecs(CodecRegistry):
    def __init__(self) -> None:
        super().__init__()
        self._scanned_types: set[type] = set()

    def encoder(self, value: Any) -> tuple[str, Callable[[Any], Any]] | None:
        value_type = type(value)
        if value_type in self._scanned_types:
            return super().encoder(value)
        # Register before MRO resolution: a newly loaded subclass can require a
        # more specific codec than an already registered base class.
        for module_name, class_name, encode in _SPECS:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            native_type = getattr(module, class_name, None)
            if isinstance(native_type, type) and native_type not in self._encoders:
                snapshot = None
                if module_name == "avstack.sensors":
                    snapshot = _snapshot_sensor
                elif module_name == "avstack.geometry.refchoc" and class_name == "ReferenceFrame":
                    snapshot = _snapshot_reference
                elif module_name.startswith("avstack."):
                    snapshot = _snapshot_native
                elif module_name == "carla":
                    snapshot = lambda value, capture_child: _sdk_capture(value)
                self.register(native_type, encode, name=f"{module_name}.{class_name}",
                              snapshot=snapshot)
        self._scanned_types.add(value_type)
        return super().encoder(value)


def default_codecs() -> CodecRegistry:
    """Return lazy codecs for data produced by Modular and Alpamayo adapters.

    Custom codecs can be registered before recording. No decoder imports or
    reconstructs native classes: persisted logs remain readable on a core-only
    installation. In-memory Recorder snapshots retain their original types.
    """
    return _AdapterCodecs()
