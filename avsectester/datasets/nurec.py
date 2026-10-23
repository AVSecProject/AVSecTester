"""Read NuRec USDZ annotations and recorded initial sequences without a rendering service.

Actor states and rig trajectories share the reconstruction world frame. Native annotations and
archive access remain available without importing scene filters or simulator code.
"""

from __future__ import annotations

import json
import io
import math
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from avsectester.data_types import ActorState, Unavailable
from avsectester.datasets.base import Dataset
from avsectester.geometry.poses import interpolate_transform, make_pose, quaternion_to_rotation
from avsectester.maps.clipgt import read_clipgt

if TYPE_CHECKING:
    from avsectester.maps.scenario_map import ScenarioMap

# Classification controls target selection, never which cuboids can occlude an insertion.
_VEHICLE_LABELS = ("automobile", "car", "truck", "bus", "trailer", "van", "vehicle")


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
    """Bracket native xyzw track poses by timestamp without extrapolating."""
    ts = np.asarray(timestamps, dtype=np.int64)
    if not len(ts) or timestamp_us < ts[0] or timestamp_us > ts[-1]:
        return None
    i = int(np.searchsorted(ts, timestamp_us, side="left"))
    values = np.asarray(poses, dtype=float)
    if i == 0 or ts[i] == timestamp_us:
        p = values[i]
        return make_pose(quaternion_to_rotation(p[6], p[3], p[4], p[5]), p[:3])
    fraction = float(timestamp_us - ts[i - 1]) / float(ts[i] - ts[i - 1])
    first, second = values[i - 1], values[i]
    first_pose = make_pose(
        quaternion_to_rotation(first[6], first[3], first[4], first[5]), first[:3],
    )
    second_pose = make_pose(
        quaternion_to_rotation(second[6], second[3], second[4], second[5]), second[:3],
    )
    return interpolate_transform(first_pose, second_pose, fraction)


def _interpolate_rig(timestamps, transforms, timestamp_us):
    if timestamp_us < timestamps[0] or timestamp_us > timestamps[-1]:
        raise ValueError("Rig timestamp exceeds recorded support")
    i = int(np.searchsorted(timestamps, timestamp_us))
    if timestamps[i] == timestamp_us:
        return transforms[i].copy()
    fraction = float(timestamp_us - timestamps[i - 1]) / float(timestamps[i] - timestamps[i - 1])
    return interpolate_transform(transforms[i - 1], transforms[i], fraction)


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
    (for example ``camera_front_wide_120fov``). Sensor names remain native here; simulators
    configure any observation aliases. Native annotations are available per scenario through
    ``raw_metadata``; ``open_archive`` provides read-only archive access.
    """

    def __init__(
        self,
        usdz_paths: list[str | Path],
        camera_id: str = "camera_front_wide_120fov",
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
        self.camera_id = camera_id
        self.vehicle_labels = tuple(vehicle_labels)
        self._paths_by_id: dict[str, str] | None = None

    @classmethod
    def from_glob(cls, pattern: str | Path, **kwargs) -> NuRecDataset:
        import glob

        return cls(sorted(glob.glob(str(pattern), recursive=True)), **kwargs)

    @staticmethod
    def open_archive(path: str | Path) -> zipfile.ZipFile:
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

    def raw_metadata(self, scenario_id: str) -> dict[str, Any]:
        """Return an independent copy of this scenario's native annotations."""
        from copy import deepcopy

        return deepcopy(self._raw_metadata(self._path(scenario_id)))

    def _camera_data(self, path):
        raw = self._raw_metadata(str(path))
        rig = raw["rig_trajectories.json"]
        traj = rig["rig_trajectories"][0]
        key = next(k for k in rig["camera_calibrations"] if k.split("@")[0] == self.camera_id)
        cam = _CameraCalibration.from_calib(rig["camera_calibrations"][key])
        ts = np.asarray(traj["cameras_frame_timestamps_us"][key], dtype=np.int64)
        transforms = np.asarray(traj["cameras_frame_T_rig_worlds"][key], dtype=float)
        return raw, traj, key, cam, ts, transforms

    def camera_geometry(self, scenario_id: str):
        """Return an independent optical camera model and its rig-from-camera pose."""
        return self._camera_geometry(self._path(scenario_id))

    def _camera_geometry(self, path: str):
        from avsectester.geometry.cameras import FThetaCamera as Lens

        _raw, _trajectory, _key, camera, _times, _poses = self._camera_data(str(path))
        return Lens(
            cx=float(camera.principal_point[0]), cy=float(camera.principal_point[1]),
            angle_to_pixeldist=tuple(camera.angle_to_pixeldist), width=camera.width,
            height=camera.height, pixeldist_to_angle=tuple(camera.pixeldist_to_angle),
            max_angle=camera.max_angle, linear_cde=camera.linear_cde,
        ), camera.t_sensor_rig.copy()

    def camera_timestamps(self, scenario_id: str) -> np.ndarray:
        """Recorded video-frame timestamps in seconds, distinct from decision positions."""
        return self._camera_timestamps(self._path(scenario_id))

    def _camera_timestamps(self, path: str) -> np.ndarray:
        return self._camera_data(str(path))[-2][:, 0].astype(float) / 1e6

    def _scene_id(self, path: str) -> str:
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
                scene_id = self._scene_id(path)
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

    @lru_cache(maxsize=8)
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
            return read_clipgt(archive)

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
        ego = self.ego_state(scenario_id, round(frame.timestamp * 1e6))
        actors = self.actor_states(scenario_id, round(frame.timestamp * 1e6))
        return FrameData(
            frame, actors=actors, ego=ego,
            provenance={"kind": "interpolated", "usdz_path": path,
                        "time_step": self.time_step,
                        "interpolation": "linear_translation_slerp_rotation"},
        )

    def native(self, scenario_id):
        return self.raw_metadata(scenario_id)

    def scenario_start(self, scenario_id, frame):
        from .base import ScenarioStart

        self.checked_frame(scenario_id, frame)
        path = self._path(scenario_id)
        timeline = self.frames(scenario_id)
        return ScenarioStart("NuRecDataset", scenario_id, frame, {
            "usdz_path": path, "camera_id": self.camera_id,
            "scene_id": scenario_id, "ego": self.ego_state(scenario_id, round(frame.timestamp * 1e6)),
            "time_step": self.time_step, "interpolation": "linear_translation_slerp_rotation",
            "first_timestamp": timeline[0].timestamp,
            "terminal_timestamp": timeline[-1].timestamp,
        }, recorded_ego=tuple(self.ego_state(scenario_id, round(item.timestamp * 1e6))
                              for item in timeline), scenario_map=self.scenario_map(scenario_id))

    def ego_state(self, scenario_id: str, timestamp_us: int) -> ActorState:
        """Interpolate the recorded rig pose and world velocity within its support."""
        return self._ego_state(self._path(scenario_id), timestamp_us)

    def _ego_state(self, path: str, timestamp_us: int) -> ActorState:
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

    def actor_states(self, scenario_id: str, timestamp_us: int) -> dict[str, ActorState]:
        """Recorded actors at this timestamp: world pose, velocity and category, by track ID.

        Velocity is the position difference over one ``time_step``, backward where the track
        exists one step earlier, otherwise forward; it is ``None`` for a single-sample track.
        """
        return self._actor_states(self._path(scenario_id), timestamp_us)

    def _actor_states(self, path: str, timestamp_us: int) -> dict[str, ActorState]:
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
