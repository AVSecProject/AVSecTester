"""Read NuRec USDZ annotations and recorded initial sequences without a rendering service.

Actor poses and rig trajectories share the reconstruction world frame. Unified object geometry is
expressed in the rig frame, with complete SE(3) poses retained for host attachment and occlusion.
Raw annotations and archive access remain available to custom filters.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from avsectester.backend import WorldBackend
from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, SceneGT
from avsectester.scenarios.source import Dataset

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
        from avsectester.rendering.cameras import FThetaCamera as Lens

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


class NuRecDataset(Dataset):
    """Labeled NuRec scenes and their recorded initial-frame windows.

    ``keyframe`` chooses the candidate start as a fraction of the configured camera's frames.
    ``initial_sequence`` continues from that start and never advances a driving policy. Native
    annotations are available through ``raw_metadata`` and ``open_archive``. Rendering remains lazy.
    """

    def __init__(
        self,
        usdz_paths: list[str],
        sensor: str = "camera_front_wide_120fov",
        camera: str = "front",
        keyframe: float = 0.5,
        vehicle_labels: tuple[str, ...] = _VEHICLE_LABELS,
        max_scenes: int | None = None,
        endpoint: str = "127.0.0.1:50051",
    ) -> None:
        if not 0 <= keyframe <= 1:
            raise ValueError("keyframe must be in [0, 1]")
        if max_scenes is not None and max_scenes < 0:
            raise ValueError("max_scenes must be nonnegative")
        self.usdz_paths = [str(p) for p in usdz_paths]
        self.sensor, self.camera, self.keyframe = sensor, camera, keyframe
        self.vehicle_labels, self.max_scenes = tuple(vehicle_labels), max_scenes
        self.endpoint = endpoint

    @classmethod
    def from_glob(cls, pattern: str, **kwargs) -> NuRecDataset:
        import glob

        return cls(sorted(glob.glob(pattern, recursive=True)), **kwargs)

    @staticmethod
    def open_archive(path: str) -> zipfile.ZipFile:
        """Open the source archive read-only. The caller closes it, preferably with ``with``."""
        return zipfile.ZipFile(path, "r")

    @lru_cache(maxsize=8)
    def raw_metadata(self, path: str) -> dict[str, Any]:
        """Return uncropped annotations, including fields outside the unified scene schema.

        Cached dictionaries are shared between contexts. Treat them as source data rather than
        mutating them. ``open_archive`` exposes additional files without loading large assets.
        """
        with self.open_archive(path) as archive:
            return {
                name: json.loads(archive.read(name))
                for name in ("rig_trajectories.json", "sequence_tracks.json")
            }

    def _camera_data(self, path):
        raw = self.raw_metadata(str(path))
        rig = raw["rig_trajectories.json"]
        traj = rig["rig_trajectories"][0]
        key = next(k for k in rig["camera_calibrations"] if k.split("@")[0] == self.sensor)
        cam = FThetaCamera.from_calib(rig["camera_calibrations"][key])
        ts = np.asarray(traj["cameras_frame_timestamps_us"][key], dtype=np.int64)
        transforms = np.asarray(traj["cameras_frame_T_rig_worlds"][key], dtype=float)
        return raw, traj, key, cam, ts, transforms

    def scenes(self) -> Iterator[SceneGT]:
        if self.max_scenes == 0:
            return
        for i, path in enumerate(self.usdz_paths):
            if self.max_scenes is not None and i >= self.max_scenes:
                return
            yield self._scene_from_usdz(path)

    def _scene_from_usdz(self, path: str) -> SceneGT:
        *_, ts, _transforms = self._camera_data(path)
        return self.scene_at_frame(path, round(self.keyframe * (len(ts) - 1)))

    def initial_sequence(self, scene: SceneGT, frames: int) -> tuple[SceneGT, ...]:
        """Recorded frames beginning at ``scene``. A short tail is not padded or repeated."""
        if frames < 1:
            raise ValueError("frames must be positive")
        path, start = scene.source["usdz_path"], scene.source["frame_index"]
        *_, ts, _ = self._camera_data(path)
        return tuple(
            self.scene_at_frame(path, i) for i in range(start, min(start + frames, len(ts)))
        )

    def scene_at_frame(self, path: str, frame: int) -> SceneGT:
        raw, traj, key, cam, timestamps, transforms = self._camera_data(path)
        if not 0 <= frame < len(timestamps):
            raise IndexError("camera frame is outside the recorded sequence")
        # Render at the first exposure timestamp, matching this exact rig pose and actor time.
        timestamp_us = int(timestamps[frame, 0])
        world_from_rig = transforms[frame, 0]
        rig_from_world = np.linalg.inv(world_from_rig)
        objects = self._objects(raw["sequence_tracks.json"], timestamp_us, rig_from_world, cam)
        calib = CameraCalib(
            self.camera, cam.width, cam.height, model=cam, cam_to_ego=cam.t_sensor_rig
        )
        return SceneGT(
            frame,
            timestamp_us / 1e6,
            EgoState(
                self._ego_speed(transforms[:, 0, :3, 3], timestamps[:, 0], frame),
                world_from_rig.tolist(),
                center=tuple((traj.get("rig_bbox") or {}).get("centroid", (0, 0, 0))),
                extent=tuple((traj.get("rig_bbox") or {}).get("dim", (0, 0, 0))),
            ),
            {self.camera: calib},
            objects,
            source={
                "dataset": "nurec",
                "usdz_path": str(path),
                "scene_id": key.split("@")[1] if "@" in key else traj.get("sequence_id"),
                "sensor": self.sensor,
                "frame_index": frame,
                "timestamp_us": timestamp_us,
            },
        )

    def actor_poses(self, path: str, timestamp_us: int):
        """World poses and full cuboid dimensions for every actor present at this timestamp."""
        from avsectester.insertion import ActorPose

        tracks = self.raw_metadata(str(path))["sequence_tracks.json"]
        return {
            tid: ActorPose(pose, extent)
            for tid, _label, extent, pose in self._track_states(tracks, timestamp_us)
        }

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
                    box2d={self.camera: box} if box else {},
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

    def context(self, scene: SceneGT):
        """Expose source annotations, native archive access, and a lazy renderer to filters."""
        from avsectester.scenarios.context import FilterContext
        from avsectester.rendering.visibility import CuboidVisibilityEstimator

        path = scene.source["usdz_path"]
        context = FilterContext(
            scene=scene,
            dataset=self,
            metadata=self.raw_metadata(path),
            frame_context=self.context,
            native={
                "open_archive": lambda: self.open_archive(path),
                "actor_poses": lambda timestamp_us: self.actor_poses(path, timestamp_us),
            },
            visibility_estimator=CuboidVisibilityEstimator(),
        )
        context.renderer = context.own(self.make_renderer(scene))
        return context

    def make_renderer(self, scene):
        """Create a renderer without opening a connection until ``load_scene`` is called."""
        from avsectester.simulators.nurec import NuRecRenderer

        return NuRecRenderer(
            endpoint=self.endpoint,
            scene_id=scene.source.get("scene_id"),
            cameras=[self.sensor],
            start_timestamp_us=scene.source["timestamp_us"],
            start_transform=scene.ego.pose,
        )

    def make_backend(self, scene: SceneGT) -> WorldBackend:
        from avsectester.simulators.nurec import NuRecBackend

        pose = np.asarray(scene.ego.pose)
        ego0 = {
            "x": float(pose[0, 3]),
            "y": float(pose[1, 3]),
            "yaw": math.atan2(pose[1, 0], pose[0, 0]),
            "speed": scene.ego.speed,
        }
        return NuRecBackend(
            config={
                "scene": scene.source.get("scene_id"),
                "ego0": ego0,
                "camera_aliases": {self.sensor: self.camera},
            },
            renderer=self.make_renderer(scene),
        )

    def insertion_renderer(self, scene, backend, insertions, bindings=None, compositor=None):
        """Build a host-bound sensor insertion callable after the backend has been reset.

        ``bindings`` accepts a selected match's role-to-track-ID tuples. Actor poses are sampled at
        the effective render timestamp and camera pose. Model-visible time and localization do
        not move the geometry. An explicit render-viewpoint change updates camera geometry while
        the victim's physical pose still comes from backend ground truth.
        """
        from avsectester.insertion import ActorPose
        from avsectester.rendering.types import InsertionGeometry
        from avsectester.rendering.visibility import CuboidVisibilityEstimator
        from avsectester.simulators.nurec import NuRecInsertions

        renderer = backend.renderer
        renderer.service  # Validate that calibration has been loaded before constructing the adapter.
        objects = {obj.track_id: obj for obj in scene.objects}
        selected = {
            role: tuple(objects[track_id] for track_id in ids)
            for role, ids in (bindings or {}).items()
        }
        from avsectester.scenarios.context import FilterContext

        aliases = FilterContext(scene, bindings=selected).actor_aliases
        local_center = np.eye(4)
        local_center[:3, 3] = scene.ego.center
        path = scene.source["usdz_path"]

        def geometry(observation):
            pose = backend.ground_truth().vehicle_state
            request = backend.render_request(self.sensor)
            adapter.camera = renderer.camera_model(request.camera)
            actors = self.actor_poses(path, renderer.timestamp_us(request.pose))
            for alias, track_id in aliases.items():
                if track_id in actors:
                    if alias in actors and alias != track_id:
                        raise ValueError(f"Role {alias!r} conflicts with an actor track ID")
                    actors[alias] = actors[track_id]
            victim = ActorPose(renderer.rig_transform(pose) @ local_center, scene.ego.extent)
            return InsertionGeometry(
                actors, victim, renderer.cam_from_world(request.pose, request.camera),
            )

        adapter = NuRecInsertions(
            insertions,
            renderer.camera_model(self.sensor),
            geometry,
            compositor,
            visibility_estimator=CuboidVisibilityEstimator(),
            camera_name=self.camera,
        )
        return adapter
