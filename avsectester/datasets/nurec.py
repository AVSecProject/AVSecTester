"""Read NuRec USDZ annotations and recorded initial sequences without a rendering service.

Actor states and rig trajectories share the reconstruction world frame. Native annotations and
archive access remain available without importing scene filters or simulator code.
"""

from __future__ import annotations

import json
import io
import math
import zipfile
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from avsectester.datasets.base import ActorState, Dataset, Unavailable

if TYPE_CHECKING:
    from avsectester.maps.scenario_map import ScenarioMap

# Classification controls target selection, never which cuboids can occlude an insertion.
_VEHICLE_LABELS = ("automobile", "car", "truck", "bus", "trailer", "van", "vehicle")


def _quat_to_rot(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix from an ``xyzw`` (scipy / NRE-export order) unit quaternion."""
    x, y, z, w = (float(v) for v in quat)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _se3(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = np.asarray(trans, dtype=np.float64)
    return m


@dataclass
class _CameraCalibration:
    """Native recorded camera parameters and its rig-from-optical-camera pose."""

    width: int
    height: int
    principal_point: np.ndarray
    angle_to_pixeldist: np.ndarray  # polynomial coeffs, low order first
    pixeldist_to_angle: np.ndarray
    max_angle: float
    t_sensor_rig: np.ndarray
    linear_cde: tuple[float, float, float] = (1.0, 0.0, 0.0)

    @classmethod
    def from_calib(cls, calib: dict) -> _CameraCalibration:
        p = calib["camera_model"]["parameters"]
        w, h = p["resolution"]
        return cls(
            width=int(w),
            height=int(h),
            principal_point=np.asarray(p["principal_point"], dtype=np.float64),
            angle_to_pixeldist=np.asarray(p["angle_to_pixeldist_poly"], dtype=np.float64),
            pixeldist_to_angle=np.asarray(p.get("pixeldist_to_angle_poly", ()), dtype=np.float64),
            max_angle=float(p["max_angle"]),
            t_sensor_rig=np.asarray(calib["T_sensor_rig"], dtype=np.float64),
            linear_cde=tuple(p.get("linear_cde", (1.0, 0.0, 0.0))),
        )


def _interpolate_pose(timestamps, poses, timestamp_us: int) -> np.ndarray | None:
    """Linearly interpolate translation and slerp xyzw orientation without extrapolating."""
    ts = np.asarray(timestamps, dtype=np.int64)
    if not len(ts) or timestamp_us < ts[0] or timestamp_us > ts[-1]:
        return None
    i = int(np.searchsorted(ts, timestamp_us, side="left"))
    values = np.asarray(poses, dtype=float)
    if i == 0 or ts[i] == timestamp_us:
        p = values[i]
        return _se3(_quat_to_rot(p[3:]), p[:3])
    a = float(timestamp_us - ts[i - 1]) / float(ts[i] - ts[i - 1])
    p0, p1 = values[i - 1], values[i]
    q0, q1 = p0[3:] / np.linalg.norm(p0[3:]), p1[3:] / np.linalg.norm(p1[3:])
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = (1 - a) * q0 + a * q1
    else:
        theta = math.acos(np.clip(dot, -1, 1))
        q = (math.sin((1 - a) * theta) * q0 + math.sin(a * theta) * q1) / math.sin(theta)
    return _se3(_quat_to_rot(q), (1 - a) * p0[:3] + a * p1[:3])


def _matrix_quaternion(matrix):
    """Stable xyzw quaternion from an SO(3) matrix, including rotations near pi."""
    m = np.asarray(matrix)
    k = np.array([
        [m[0, 0]-m[1, 1]-m[2, 2], m[1, 0]+m[0, 1], m[2, 0]+m[0, 2], m[2, 1]-m[1, 2]],
        [m[1, 0]+m[0, 1], m[1, 1]-m[0, 0]-m[2, 2], m[2, 1]+m[1, 2], m[0, 2]-m[2, 0]],
        [m[2, 0]+m[0, 2], m[2, 1]+m[1, 2], m[2, 2]-m[0, 0]-m[1, 1], m[1, 0]-m[0, 1]],
        [m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1], np.trace(m)],
    ]) / 3
    return np.linalg.eigh(k)[1][:, -1]


def _interpolate_rig(timestamps, transforms, timestamp_us):
    if timestamp_us < timestamps[0] or timestamp_us > timestamps[-1]:
        raise ValueError("Rig timestamp exceeds recorded support")
    i = int(np.searchsorted(timestamps, timestamp_us))
    if timestamps[i] == timestamp_us:
        return transforms[i].copy()
    poses = [np.r_[m[:3, 3], _matrix_quaternion(m[:3, :3])] for m in transforms[i-1:i+1]]
    return _interpolate_pose(timestamps[i-1:i+1], poses, timestamp_us)


def _map_points(values):
    """Read one clipgt polyline; placeholder rows contain no geometry."""
    if not values:
        return np.empty((0, 3), dtype=float)
    return np.asarray([[point[axis] for axis in ("x", "y", "z")] for point in values], dtype=float)


def _matched_rails(left, right):
    """Pair rail points by normalized arc length before taking their midpoint."""
    count = max(len(left), len(right))
    fractions = np.linspace(0., 1., count)

    def resample(points):
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if distance[-1] == 0:
            return np.repeat(points[:1], count, axis=0)
        return np.column_stack([np.interp(fractions, distance / distance[-1], points[:, axis])
                                for axis in range(3)])

    return (resample(left) + resample(right)) / 2


def _speed_multiplier(archive):
    """Use explicit OpenDRIVE units; bare clipgt speed numbers do not declare a unit."""
    if "map.xodr" not in archive.namelist():
        return None
    root = ElementTree.fromstring(archive.read("map.xodr"))
    units = {element.get("unit") for element in root.findall(".//speed")
             if element.get("max") not in {None, "0", "undefined", "no limit"}}
    if len(units) != 1:
        return None
    return {"mph": .44704, "km/h": 1 / 3.6, "m/s": 1.}.get(next(iter(units)))


def _lane_kind(record):
    uses = set(record.get("use_types") or ())
    for token, kind in (("SHOULDER_LANE", "shoulder"), ("PARKING_LANE", "parking")):
        if token in uses:
            return kind
    vehicles = set(record.get("vehicle_types") or ())
    if "CAR" in vehicles:
        return "driving"
    if vehicles == {"BICYCLE"}:
        return "bike"
    return "other"


def _control_kind(category):
    # These source categories denote MUTCD R1-1, R1-2 and R2-1. Other signs that
    # mention stopping or yielding impose different rules and retain kind="other".
    return {
        "TRAFFIC_SIGN_REGULATORY_R1_STOP": "stop_sign",
        "TRAFFIC_SIGN_REGULATORY_R1_YIELD": "yield_sign",
        "TRAFFIC_SIGN_REGULATORY_R2_SPEED_LIMIT": "speed_limit",
    }.get(category, "other")


def _read_clipgt_map(archive):
    import pyarrow.parquet as parquet

    from avsectester.maps.scenario_map import (
        STOP_LINE_KIND_BY_CONTROL_KIND, Lane, ScenarioMap, StopLine, TrafficControl,
    )

    unavailable = {}
    missing_tables = {}

    def records(layer):
        name = f"clipgt/{layer}.parquet"
        if name not in archive.namelist():
            missing_tables[layer] = f"NuRec archive has no {name}"
            return ()
        rows = parquet.read_table(io.BytesIO(archive.read(name))).to_pylist()
        return tuple((row.get("key") or {}, row.get(layer) or {}) for row in rows)

    lane_records = {}
    for key, record in records("lane"):
        left, right = _map_points(record.get("left_rail")), _map_points(record.get("right_rail"))
        if key.get("map_id") is not None and len(left) >= 2 and len(right) >= 2:
            lane_records[str(key["map_id"])] = (record, left, right)

    control_records = {}
    for layer in ("traffic_sign", "traffic_light"):
        valid_controls = 0
        for key, record in records(layer):
            center, dimensions = record.get("center") or {}, record.get("dimensions") or {}
            orientation = record.get("orientation") or {}
            values = [*(center.get(axis) for axis in ("x", "y", "z")),
                      *(dimensions.get(axis) for axis in ("x", "y", "z")),
                      *(orientation.get(axis) for axis in ("x", "y", "z", "w"))]
            if all(value is None for value in values):
                continue
            if key.get("map_id") is None or any(value is None for value in values):
                unavailable["traffic_controls.partial"] = (
                    "Some NuRec controls lack an identity, pose or dimensions")
                continue
            category = str(record.get("category") or "")
            pose = _se3(_quat_to_rot(np.array([orientation[axis] for axis in ("x", "y", "z", "w")])),
                        np.array([center[axis] for axis in ("x", "y", "z")]))
            extent = tuple(float(dimensions[axis]) for axis in ("x", "y", "z"))
            control_records[str(key["map_id"])] = (
                "traffic_light" if layer == "traffic_light" else _control_kind(category),
                category, pose, extent,
            )
            valid_controls += 1
        if layer == "traffic_light" and not valid_controls:
            unavailable["traffic_controls.traffic_light"] = missing_tables.get(layer) or (
                "NuRec traffic_light source contains no valid light geometry")

    line_records = {}
    source_line_flags = []
    for key, record in records("wait_line"):
        if key.get("map_id") is not None:
            source_line_flags.append(record.get("is_implicit"))
        points = _map_points(record.get("location"))
        if (record.get("is_implicit") is False and key.get("map_id") is not None
                and len(points) >= 2):
            # STOP is a paint label shared by signal-controlled and stop-controlled
            # approaches. The lane's controlling light/sign determines the line kind.
            line_records[str(key["map_id"])] = (points, str(record.get("category") or ""))
    implicit_only = bool(source_line_flags) and all(flag is True for flag in source_line_flags)

    neighbors = {identity: {name: [] for name in ("predecessors", "successors", "left", "right")}
                 for identity in lane_records}
    control_lanes = {identity: set() for identity in control_records}
    line_lanes = {identity: set() for identity in line_records}
    lane_relations = {"PREVIOUS_LANE": "predecessors", "NEXT_LANE": "successors",
                      "LEFT_LANE": "left", "RIGHT_LANE": "right"}
    for key, record in records("association"):
        relation = lane_relations.get(key.get("kind"))
        for subject in record.get("subjects") or ():
            for target in record.get("objects") or ():
                subject, target = str(subject), str(target)
                if relation and subject in neighbors and target in neighbors:
                    neighbors[subject][relation].append(target)
                # Real clipgt SIGN_TO_LANE/LIGHT_TO_LANE/WAIT_LINE_TO_LANE rows put the
                # lane in subjects. Identify endpoints by their layer instead of the label.
                for first, second in ((subject, target), (target, subject)):
                    if first in lane_records and second in control_records:
                        control_lanes[second].add(first)
                    if first in lane_records and second in line_records:
                        line_lanes[second].add(first)

    lane_controls = {identity: set() for identity in lane_records}
    control_lines = {identity: set() for identity in control_records}
    for control_id, lane_ids in control_lanes.items():
        if control_records[control_id][0] in STOP_LINE_KIND_BY_CONTROL_KIND:
            for lane_id in lane_ids:
                lane_controls[lane_id].add(control_id)
    line_controls = {identity: set().union(*(lane_controls[lane_id] for lane_id in lane_ids))
                     for identity, lane_ids in line_lanes.items()}
    for line_id, control_ids in line_controls.items():
        for control_id in control_ids:
            control_lines[control_id].add(line_id)

    multiplier = _speed_multiplier(archive)
    lanes = {}
    for identity, (record, left, right) in lane_records.items():
        speed = None
        value = record.get("speed_limit")
        if value in (None, "", "0", 0):
            unavailable["lanes.speed_limit_mps"] = "Some NuRec lane speed limits are missing or unspecified"
        else:
            if multiplier is None:
                unavailable["lanes.speed_limit_mps"] = "NuRec lane speed values have no confirmed unit"
            else:
                speed = float(value) * multiplier
        relation = neighbors[identity]
        lanes[identity] = Lane(
            id=identity, centerline=_matched_rails(left, right), left_boundary=left,
            right_boundary=right, lane_type=_lane_kind(record),
            predecessors=tuple(dict.fromkeys(relation["predecessors"])),
            successors=tuple(dict.fromkeys(relation["successors"])),
            left_neighbor=next(iter(relation["left"]), None),
            right_neighbor=next(iter(relation["right"]), None), speed_limit_mps=speed,
        )
    controls = {identity: TrafficControl(
        id=identity, kind=kind, native_category=category, pose=pose, extent=extent,
        pose_reference="face_center",
        lane_ids=tuple(sorted(control_lanes[identity])),
        stop_line_id=next(iter(control_lines[identity])) if len(control_lines[identity]) == 1 else None,
    ) for identity, (kind, category, pose, extent) in control_records.items()}
    lines = {}
    for identity, (points, native_category) in line_records.items():
        controller_ids = line_controls[identity]
        kinds = {STOP_LINE_KIND_BY_CONTROL_KIND[control_records[control_id][0]] for control_id in controller_ids}
        lines[identity] = StopLine(
            id=identity, points=points, native_category=native_category,
            kind=next(iter(kinds)) if len(kinds) == 1 else "unknown",
            lane_ids=tuple(sorted(line_lanes[identity])),
            control_id=next(iter(controller_ids)) if len(controller_ids) == 1 else None,
        )

    def geometries(layer, minimum):
        return tuple(points for _key, record in records(layer)
                     if len(points := _map_points(record.get("location"))) >= minimum)

    values: dict[str, Any] = {"lanes": lanes, "traffic_controls": controls, "stop_lines": lines}
    layers: dict[str, tuple[str, ...]] = {
        "lanes": ("lane",), "traffic_controls": ("traffic_sign", "traffic_light"),
        "stop_lines": ("wait_line",),
    }
    for name, layer, minimum in (("drivable_area", "drivable_space", 3),
            ("road_boundaries", "road_boundary", 2), ("crosswalks", "crosswalk", 3),
            ("intersections", "intersection_area", 3)):
        values[name] = geometries(layer, minimum)
        layers[name] = (layer,)
    for name, records_in_layer in values.items():
        if not records_in_layer:
            if name == "stop_lines" and implicit_only:
                continue
            reasons = [missing_tables[layer] for layer in layers[name] if layer in missing_tables]
            unavailable[name] = "; ".join(reasons) or "NuRec source layer contains no valid geometry"
    return ScenarioMap(**values, unavailable_layers=unavailable)


def read_scene_mesh(usdz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Merge mesh.ply and mesh_ground.ply without loading camera or trajectory metadata.

    The indexed vertices use the archive's metre-based, Z-up scene-world frame.
    The neural volume's separate ``world_to_nre`` offset is not applied.
    """
    try:
        import trimesh
    except ImportError as error:
        raise ImportError("NuRec scene meshes require trimesh; install avsectester[nurec]") from error
    vertices, faces = [], []
    offset = 0
    with zipfile.ZipFile(usdz_path, "r") as archive:
        for name in ("mesh.ply", "mesh_ground.ply"):
            if name not in archive.namelist():
                raise ValueError(f"NuRec archive {str(usdz_path)!r} has no {name}")
            mesh = trimesh.load(io.BytesIO(archive.read(name)), file_type="ply", process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"NuRec {name} is not a triangle mesh")
            points, triangles = np.asarray(mesh.vertices), np.asarray(mesh.faces)
            if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                    or triangles.ndim != 2 or triangles.shape[1] != 3
                    or (triangles < 0).any() or (triangles >= len(points)).any()):
                raise ValueError(f"NuRec {name} has invalid vertices or triangle indices")
            vertices.append(points)
            faces.append(triangles if offset == 0 else triangles + offset)
            offset += len(points)
    return np.concatenate(vertices), np.concatenate(faces)


class NuRecDataset(Dataset):
    """Labeled NuRec scenes on a decision timeline generated at ``time_step``.

    Each USDZ file is one scenario, identified by NuRec's own scene ID (``clipgt-<uuid>``), the
    ID the NuRec renderer loads. Frame data and scenario starts interpolate the recorded rig
    trajectory and actor tracks within recorded support. ``camera_id`` is NuRec's camera ID
    (for example ``camera_front_wide_120fov``); ``camera_alias`` is the sensor name used in
    observations and scene calibration. Native annotations are available per file through
    ``raw_metadata`` and ``open_archive``.
    """

    def __init__(
        self,
        usdz_paths: list[str],
        camera_id: str = "camera_front_wide_120fov",
        camera_alias: str = "front",
        vehicle_labels: tuple[str, ...] = _VEHICLE_LABELS,
        time_step: float = 0.1,
    ) -> None:
        if isinstance(time_step, bool) or not math.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be positive and finite")
        self._step_us = round(time_step * 1e6)
        if self._step_us < 1 or abs(time_step * 1e6 - self._step_us) > 1e-6:
            raise ValueError("time_step must be representable in integer microseconds")
        self.time_step = self._step_us / 1e6
        self.usdz_paths = [str(p) for p in usdz_paths]
        self.camera_id, self.camera_alias = camera_id, camera_alias
        self.vehicle_labels = tuple(vehicle_labels)
        self._paths_by_id: dict[str, str] | None = None

    @classmethod
    def from_glob(cls, pattern: str, **kwargs) -> NuRecDataset:
        import glob

        return cls(sorted(glob.glob(pattern, recursive=True)), **kwargs)

    @staticmethod
    def open_archive(path: str) -> zipfile.ZipFile:
        """Open the scene archive read-only. The caller closes it, preferably with ``with``."""
        return zipfile.ZipFile(path, "r")

    @lru_cache(maxsize=8)
    def _raw_metadata(self, path: str) -> dict[str, Any]:
        """Return uncropped annotations, including fields outside the unified scene schema.

        Cached dictionaries are shared between contexts. Treat them as dataset records rather than
        mutating them. ``open_archive`` exposes additional files without loading large assets.
        """
        with self.open_archive(path) as archive:
            return {
                name: json.loads(archive.read(name))
                for name in ("rig_trajectories.json", "sequence_tracks.json")
            }

    def raw_metadata(self, path: str) -> dict[str, Any]:
        from copy import deepcopy

        return deepcopy(self._raw_metadata(path))

    def _camera_data(self, path):
        raw = self._raw_metadata(str(path))
        rig = raw["rig_trajectories.json"]
        traj = rig["rig_trajectories"][0]
        key = next(k for k in rig["camera_calibrations"] if k.split("@")[0] == self.camera_id)
        cam = _CameraCalibration.from_calib(rig["camera_calibrations"][key])
        ts = np.asarray(traj["cameras_frame_timestamps_us"][key], dtype=np.int64)
        transforms = np.asarray(traj["cameras_frame_T_rig_worlds"][key], dtype=float)
        return raw, traj, key, cam, ts, transforms

    def camera_geometry(self, path: str):
        """Return an independent optical camera model and its rig-from-camera pose."""
        from avsectester.geometry.cameras import FThetaCamera as Lens

        _raw, _trajectory, _key, camera, _times, _poses = self._camera_data(str(path))
        return Lens(
            cx=float(camera.principal_point[0]), cy=float(camera.principal_point[1]),
            angle_to_pixeldist=tuple(camera.angle_to_pixeldist), width=camera.width,
            height=camera.height, pixeldist_to_angle=tuple(camera.pixeldist_to_angle),
            max_angle=camera.max_angle, linear_cde=camera.linear_cde,
        ), camera.t_sensor_rig.copy()

    def camera_timestamps(self, path: str) -> np.ndarray:
        """Recorded video-frame timestamps in seconds, distinct from decision positions."""
        return self._camera_data(str(path))[-2][:, 0].astype(float) / 1e6

    def scene_id(self, path: str) -> str:
        """NuRec's scene ID of a USDZ file, as used by the NuRec renderer."""
        _raw, traj, key, *_ = self._camera_data(path)
        scene_id = key.split("@")[1] if "@" in key else traj.get("sequence_id")
        if not scene_id:
            raise ValueError(f"No NuRec scene ID in {path}")
        return scene_id

    def _paths(self) -> dict[str, str]:
        if self._paths_by_id is None:
            paths: dict[str, str] = {}
            for path in self.usdz_paths:
                scene_id = self.scene_id(path)
                if scene_id in paths:
                    raise ValueError(f"NuRec scene {scene_id!r} occurs in {paths[scene_id]} and {path}")
                paths[scene_id] = path
            self._paths_by_id = paths
        return self._paths_by_id

    def _path(self, scenario_id):
        try:
            return self._paths()[scenario_id]
        except KeyError:
            raise KeyError(f"Unknown NuRec scenario: {scenario_id!r}") from None

    def usdz_path(self, scenario_id: str) -> str:
        """Return this scenario's source archive path without creating a simulator."""
        return self._path(scenario_id)

    def scene_mesh(self, scenario_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Merge the static and ground PLY meshes in canonical scene-world coordinates."""
        return read_scene_mesh(self.usdz_path(scenario_id))

    def scenario_map(self, scenario_id: str) -> ScenarioMap | Unavailable:
        """Read canonical-world clipgt geometry, preserving missing-layer reasons.

        Bare lane speed values are converted only when the archive's OpenDRIVE declares
        one common unit. An OpenDRIVE-only archive cannot establish alignment to the rig
        coordinates from its geographic projection alone, so that fallback is unavailable.
        """
        with self.open_archive(self.usdz_path(scenario_id)) as archive:
            if not any(name.startswith("clipgt/") and name.endswith(".parquet")
                       for name in archive.namelist()):
                if "map.xodr" in archive.namelist():
                    return Unavailable("NuRec map.xodr coordinate alignment to the scenario is unconfirmed")
                return Unavailable("NuRec archive provides neither clipgt map layers nor map.xodr")
            return _read_clipgt_map(archive)

    def scenarios(self):
        from .base import ScenarioDescriptor

        for scene_id, path in self._paths().items():
            yield ScenarioDescriptor(scene_id, "NuRecDataset", {"usdz_path": path})

    def frames(self, scenario_id):
        from .base import FrameReference

        *_, timestamps, _ = self._camera_data(self._path(scenario_id))
        first, last = int(timestamps[0, 0]), int(timestamps[-1, 0])
        return tuple(FrameReference(scenario_id, i, i, t / 1e6)
                     for i, t in enumerate(range(first, last + 1, self._step_us)))

    def frame_data(self, scenario_id, frame):
        from .base import FrameData

        self.checked_frame(scenario_id, frame)
        path = self._path(scenario_id)
        ego = self.ego_state(path, round(frame.timestamp * 1e6))
        actors = self.actor_states(path, round(frame.timestamp * 1e6))
        return FrameData(
            frame, actors=actors, ego=ego,
            provenance={"kind": "interpolated", "usdz_path": path,
                        "time_step": self.time_step,
                        "interpolation": "linear_translation_slerp_rotation"},
        )

    def native(self, scenario_id):
        return self.raw_metadata(self._path(scenario_id))

    def scenario_start(self, scenario_id, frame):
        from .base import ScenarioStart

        self.checked_frame(scenario_id, frame)
        path = self._path(scenario_id)
        timeline = self.frames(scenario_id)
        return ScenarioStart("NuRecDataset", scenario_id, frame, {
            "usdz_path": path, "camera_id": self.camera_id, "camera_alias": self.camera_alias,
            "scene_id": scenario_id, "ego": self.ego_state(path, round(frame.timestamp * 1e6)),
            "time_step": self.time_step, "interpolation": "linear_translation_slerp_rotation",
            "first_timestamp": timeline[0].timestamp,
            "terminal_timestamp": timeline[-1].timestamp,
        })

    def ego_state(self, path: str, timestamp_us: int) -> ActorState:
        """Interpolate the recorded rig pose and world velocity within its support."""
        _raw, trajectory, _key, _cam, timestamps, transforms = self._camera_data(str(path))
        times, poses = timestamps[:, 0], transforms[:, 0]
        pose = _interpolate_rig(times, poses, timestamp_us)
        velocity = np.zeros(3)
        if len(times) > 1:
            i = min(max(1, int(np.searchsorted(times, timestamp_us))), len(times) - 1)
            velocity = (poses[i, :3, 3] - poses[i - 1, :3, 3]) / ((times[i] - times[i - 1]) / 1e6)
        bbox = trajectory.get("rig_bbox") or {}
        dimensions = bbox.get("dim")
        return ActorState(
            pose, tuple(float(v) for v in velocity), "vehicle",
            None if dimensions is None else tuple(float(v) for v in dimensions),
            tuple(float(v) for v in bbox.get("centroid", (0, 0, 0))),
        )

    def actor_poses(self, path: str, timestamp_us: int):
        """World poses and full cuboid dimensions for every actor present at this timestamp."""
        tracks = self._raw_metadata(str(path))["sequence_tracks.json"]
        return {actor_id: ActorState(pose, extent=extent).box_pose()
                for actor_id, _label, extent, pose in self._track_states(tracks, timestamp_us)}

    def actor_states(self, path: str, timestamp_us: int) -> dict[str, ActorState]:
        """Recorded actors at this timestamp: world pose, velocity and category, by track ID.

        Velocity is the position difference over one ``time_step``, backward where the track
        exists one step earlier, otherwise forward; it is ``None`` for a single-sample track.
        """
        tracks = self._raw_metadata(str(path))["sequence_tracks.json"]
        step_us = round(self.time_step * 1e6)
        before = {tid: pose for tid, _, _, pose in self._track_states(tracks, timestamp_us - step_us)}
        after = {tid: pose for tid, _, _, pose in self._track_states(tracks, timestamp_us + step_us)}
        states = {}
        for tid, label, extent, pose in self._track_states(tracks, timestamp_us):
            velocity = None
            if tid in before:
                velocity = (pose[:3, 3] - before[tid][:3, 3]) / self.time_step
            elif tid in after:
                velocity = (after[tid][:3, 3] - pose[:3, 3]) / self.time_step
            vehicle = any(v in label.lower() for v in self.vehicle_labels)
            states[tid] = ActorState(
                np.array(pose, dtype=float),
                None if velocity is None else tuple(float(v) for v in velocity),
                "vehicle" if vehicle else label,
                tuple(float(value) for value in extent),
            )
        return states

    @staticmethod
    def _track_states(tracks, timestamp_us):
        for sequence in tracks.values():
            data = sequence["tracks_data"]
            for tid, label, extent, ts, poses in zip(
                data["tracks_id"],
                data["tracks_label_class"],
                sequence["cuboidtracks_data"]["cuboids_dims"],
                data["tracks_timestamps_us"],
                data["tracks_poses"],
            ):
                pose = _interpolate_pose(ts, poses, timestamp_us)
                if pose is not None:
                    yield str(tid), str(label), tuple(float(v) for v in extent), pose
