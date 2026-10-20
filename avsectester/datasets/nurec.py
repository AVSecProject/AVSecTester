"""Read NuRec USDZ annotations and recorded initial sequences without a rendering service.

Actor states and rig trajectories share the reconstruction world frame. Native annotations and
archive access remain available without importing scene filters or simulator code.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from avsectester.datasets.base import ActorState, Dataset

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
            height=camera.height, max_angle=camera.max_angle, linear_cde=camera.linear_cde,
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
