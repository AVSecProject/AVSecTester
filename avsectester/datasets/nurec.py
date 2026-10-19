"""Read NuRec USDZ annotations and recorded initial sequences without a rendering service.

Actor poses and rig trajectories share the reconstruction world frame. Unified object geometry is
expressed in the rig frame, with complete SE(3) poses retained for host attachment and occlusion.
Raw annotations and archive access remain available to custom filters.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, SceneGT
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
class FThetaCamera:
    """The nuRec f-theta (fisheye-polynomial) camera: projects rig-frame points to pixels.

    ``t_sensor_rig`` is the sensor pose *in the rig* (sensor axes: x right, y down, z forward), so
    ``inv(t_sensor_rig)`` maps rig -> sensor. ``angle_to_pixeldist`` (low->high coeffs) maps the ray angle
    from the optical axis to a radial pixel distance from ``principal_point``. ``linear_cde`` is the small
    affine screen correction ``[[c, d], [e, 1]]``. Points beyond ``max_angle`` or behind the sensor do not
    project."""

    width: int
    height: int
    principal_point: np.ndarray
    angle_to_pixeldist: np.ndarray  # polynomial coeffs, low order first
    max_angle: float
    t_sensor_rig: np.ndarray
    linear_cde: tuple[float, float, float] = (1.0, 0.0, 0.0)

    @classmethod
    def from_calib(cls, calib: dict) -> FThetaCamera:
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

    def project(self, pts_rig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project ego/rig-frame points ``(N, 3)`` -> pixels ``(N, 2)`` + a validity mask ``(N,)``."""
        from avsectester.geometry.cameras import FThetaCamera as Lens

        pts = np.atleast_2d(np.asarray(pts_rig, dtype=np.float64))
        homo = np.c_[pts, np.ones(len(pts))]
        cam = (np.linalg.inv(self.t_sensor_rig) @ homo.T).T[
            :, :3
        ]  # sensor frame (x right, y down, z fwd)
        lens = Lens(
            cx=self.principal_point[0],
            cy=self.principal_point[1],
            angle_to_pixeldist=tuple(self.angle_to_pixeldist),
            width=self.width,
            height=self.height,
            max_angle=self.max_angle,
            linear_cde=tuple(self.linear_cde),
        )
        return lens.project(cam), lens.in_view(cam)


def _box2d_from_corners(cam: FThetaCamera, corners_rig: np.ndarray) -> tuple | None:
    """Tight 2-D box (clipped to the frame) from a cuboid's 8 rig-frame corners, or None if it does not
    project (needs >= 2 valid corners in front of the camera)."""
    px, valid = cam.project(corners_rig)
    px = px[valid]
    if len(px) < 2:
        return None
    x1, y1 = (
        float(np.clip(px[:, 0].min(), 0, cam.width)),
        float(np.clip(px[:, 1].min(), 0, cam.height)),
    )
    x2, y2 = (
        float(np.clip(px[:, 0].max(), 0, cam.width)),
        float(np.clip(px[:, 1].max(), 0, cam.height)),
    )
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


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


def _rig_speed(timestamps, transforms, timestamp_us):
    if len(timestamps) < 2:
        return 0.0
    i = min(max(1, int(np.searchsorted(timestamps, timestamp_us))), len(timestamps) - 1)
    return float(np.linalg.norm(transforms[i, :3, 3] - transforms[i-1, :3, 3])
                 / ((timestamps[i] - timestamps[i-1]) / 1e6))


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
        cam = FThetaCamera.from_calib(rig["camera_calibrations"][key])
        ts = np.asarray(traj["cameras_frame_timestamps_us"][key], dtype=np.int64)
        transforms = np.asarray(traj["cameras_frame_T_rig_worlds"][key], dtype=float)
        return raw, traj, key, cam, ts, transforms

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
        from .base import ActorState, FrameData

        self.checked_frame(scenario_id, frame)
        path = self._path(scenario_id)
        scene = self._timeline_scene(scenario_id, frame.position)
        actors = {"ego": ActorState(scene.ego.pose, category="vehicle", extent=scene.ego.extent)}
        actors.update(self.actor_states(path, round(frame.timestamp * 1e6)))
        return FrameData(frame, scene, actors,
                              provenance={"kind": "interpolated", "usdz_path": path,
                                          "time_step": self.time_step,
                                          "interpolation": "linear_translation_slerp_rotation"})

    def native(self, scenario_id):
        return self.raw_metadata(self._path(scenario_id))

    def scenario_start(self, scenario_id, frame):
        from .base import ScenarioStart

        self.checked_frame(scenario_id, frame)
        scene = self._timeline_scene(scenario_id, frame.position)
        return ScenarioStart("NuRecDataset", scenario_id, frame, {
            "scene": scene, "camera_id": self.camera_id, "camera_alias": self.camera_alias,
            "time_step": self.time_step, "interpolation": "linear_translation_slerp_rotation",
            "first_timestamp": self.frames(scenario_id)[0].timestamp,
            "terminal_timestamp": self.frames(scenario_id)[-1].timestamp,
        })

    def _timeline_scene(self, scenario_id: str, position: int) -> SceneGT:
        """The scene at a decision-timeline position, interpolated within recorded support."""
        path = self._path(scenario_id)
        _raw, _traj, _key, _cam, timestamps, transforms = self._camera_data(path)
        timeline = self.frames(scenario_id)
        if not 0 <= position < len(timeline):
            raise IndexError("decision position is outside the recorded sequence")
        timestamp_us = round(timeline[position].timestamp * 1e6)
        return self._scene(
            path, position, timestamp_us,
            _interpolate_rig(timestamps[:, 0], transforms[:, 0], timestamp_us),
            _rig_speed(timestamps[:, 0], transforms[:, 0], timestamp_us),
            {"time_step": self.time_step, "interpolation": "linear_translation_slerp_rotation"},
        )

    def _scene(self, path, frame, timestamp_us, world_from_rig, speed, extra_source=None) -> SceneGT:
        raw, traj, key, cam, _timestamps, _transforms = self._camera_data(path)
        rig_from_world = np.linalg.inv(world_from_rig)
        objects = self._objects(raw["sequence_tracks.json"], timestamp_us, rig_from_world, cam)
        calib = CameraCalib(
            self.camera_alias, cam.width, cam.height, model=cam, cam_to_ego=cam.t_sensor_rig
        )
        return SceneGT(
            frame,
            timestamp_us / 1e6,
            EgoState(
                speed,
                np.asarray(world_from_rig).tolist(),
                center=tuple((traj.get("rig_bbox") or {}).get("centroid", (0, 0, 0))),
                extent=tuple((traj.get("rig_bbox") or {}).get("dim", (0, 0, 0))),
            ),
            {self.camera_alias: calib},
            objects,
            source={
                "dataset": "nurec",
                "usdz_path": str(path),
                "scene_id": key.split("@")[1] if "@" in key else traj.get("sequence_id"),
                "camera_id": self.camera_id,
                "frame_index": frame,
                **(extra_source or {}),
                "timestamp_us": timestamp_us,
            },
        )

    def actor_poses(self, path: str, timestamp_us: int):
        """World poses and full cuboid dimensions for every actor present at this timestamp."""
        from avsectester.geometry import ActorPose

        tracks = self._raw_metadata(str(path))["sequence_tracks.json"]
        return {
            tid: ActorPose(pose, extent)
            for tid, _label, extent, pose in self._track_states(tracks, timestamp_us)
        }

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

    def _objects(self, tracks, timestamp_us, rig_from_world, cam):
        objects = []
        for tid, label, extent, pose in self._track_states(tracks, timestamp_us):
            local_pose = rig_from_world @ pose
            center = local_pose[:3, 3]
            half = np.asarray(extent) / 2
            local = np.array(
                [
                    [x, y, z]
                    for x in (-half[0], half[0])
                    for y in (-half[1], half[1])
                    for z in (-half[2], half[2])
                ]
            )
            corners = local @ local_pose[:3, :3].T + center
            box = _box2d_from_corners(cam, corners)
            category = "vehicle" if any(v in label.lower() for v in self.vehicle_labels) else label
            objects.append(
                ObjectGT(
                    track_id=tid,
                    category=category,
                    center=tuple(center),
                    extent=extent,
                    yaw=math.atan2(local_pose[1, 0], local_pose[0, 0]),
                    box2d={self.camera_alias: box} if box else {},
                    pose=local_pose.tolist(),
                )
            )
        return objects

    @staticmethod
    def _ego_speed(origins_world, ts_us, frame):
        if len(ts_us) < 2:
            return 0.0
        other = frame + 1 if frame + 1 < len(ts_us) else frame - 1
        dt = abs(float(ts_us[other]) - float(ts_us[frame])) / 1e6
        return (
            float(np.linalg.norm(origins_world[other] - origins_world[frame]) / dt) if dt else 0.0
        )
