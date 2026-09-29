"""nuRec (NVIDIA NRE) ``.usdz`` -> :class:`SceneGT`: filter real nuRec scenes by an attack requirement.

Each nuRec scene is a ``.usdz`` (a ZIP) reconstructed from a real driving clip (scene ids
``clipgt-<uuid>``). The render RPC returns pixels only, but the ``.usdz`` itself **ships ground truth**:

* ``sequence_tracks.json`` — actor cuboid tracks: ``tracks_id``, ``tracks_label_class``,
  ``tracks_poses`` (per timestamp, ``[x, y, z, qx, qy, qz, qw]`` in the **NRE** frame), ``tracks_flags``,
  ``tracks_timestamps_us``, and ``cuboidtracks_data.cuboids_dims`` (l, w, h);
* ``rig_trajectories.json`` — per rig_trajectory the **per-camera-frame** rig poses
  ``cameras_frame_T_rig_worlds`` (the rig's pose in the local frame, SE3, aligned 1:1 with the rendered
  video frames) and ``cameras_frame_timestamps_us``, plus each camera's ``camera_calibrations`` (an
  f-theta intrinsic + the ``T_sensor_rig`` extrinsic).

:class:`NuRecDataset` reads them and, at a chosen camera frame, expresses every actor present in the
**rig frame** — which is exactly our ego frame (x fwd, y left, z up; AlpaSim ``CONTRIBUTING.md``) — via
``inv(T_rig_world) @ actor_pose``. The track poses and the rig trajectory share the same frame, so this
inverse is the whole transform, matching AlpaSim (which uses the tracks directly with ``pose_local_to_rig``
and applies ``world_to_nre`` only on the renderer side, never to the tracks). 2-D boxes are projected with
the scene's real **f-theta** camera model, so a :class:`SceneGT` the requirement predicate runs on is
produced with no renderer or GPU. This puts nuRec on the same tier as
:class:`~avsectester.scenarios.datasets.nuscenes.NuScenesDataset` (a real *labeled* dataset).

Verified against a real ``.usdz`` (``PhysicalAI-Autonomous-Vehicles-NuRec`` 26.01) by overlaying the
projected boxes on the rendered ``.mp4`` — the composition (``inv(T_rig_world)``, no ``world_to_nre``),
the ``xyzw`` quaternion order (AlpaSim's ``Pose`` uses scipy order), the ``automobile / heavy_truck / bus
/ trailer`` label strings, and the f-theta poly all come from the data. ``visibility`` is set to 1.0 —
the tracks carry no occlusion fraction. ``cv2``/zip imported lazily; the module imports offline.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from avsectester.backend import WorldBackend
from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, SceneGT
from avsectester.scenarios.source import Dataset

# nuRec ``tracks_label_class`` substrings that count as a 4-wheeled "vehicle" (a rear-surface patch
# target). Real classes seen: automobile, heavy_truck, bus, trailer (+ protruding_object, excluded).
_VEHICLE_LABELS = ("automobile", "car", "truck", "bus", "trailer", "van", "vehicle")


def _quat_to_rot(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix from an ``xyzw`` (scipy / NRE-export order) unit quaternion."""
    x, y, z, w = (float(v) for v in quat)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


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
    from the optical axis to a radial pixel distance from ``principal_point``; ``linear_cde`` is the small
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
        return cls(width=int(w), height=int(h),
                   principal_point=np.asarray(p["principal_point"], dtype=np.float64),
                   angle_to_pixeldist=np.asarray(p["angle_to_pixeldist_poly"], dtype=np.float64),
                   max_angle=float(p["max_angle"]),
                   t_sensor_rig=np.asarray(calib["T_sensor_rig"], dtype=np.float64),
                   linear_cde=tuple(p.get("linear_cde", (1.0, 0.0, 0.0))))

    def project(self, pts_rig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project ego/rig-frame points ``(N, 3)`` -> pixels ``(N, 2)`` + a validity mask ``(N,)``."""
        pts = np.atleast_2d(np.asarray(pts_rig, dtype=np.float64))
        homo = np.c_[pts, np.ones(len(pts))]
        cam = (np.linalg.inv(self.t_sensor_rig) @ homo.T).T[:, :3]  # sensor frame (x right, y down, z fwd)
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
        rxy = np.hypot(x, y)
        theta = np.arctan2(rxy, z)
        valid = (z > 0) & (theta <= self.max_angle)
        r = np.polyval(self.angle_to_pixeldist[::-1], theta)  # polyval wants high->low
        ux = np.divide(x, rxy, out=np.zeros_like(x), where=rxy > 1e-9)
        uy = np.divide(y, rxy, out=np.zeros_like(y), where=rxy > 1e-9)
        du, dv = r * ux, r * uy
        c, d, e = self.linear_cde
        u = self.principal_point[0] + c * du + d * dv
        v = self.principal_point[1] + e * du + dv
        return np.c_[u, v], valid


def _box2d_from_corners(cam: FThetaCamera, corners_rig: np.ndarray) -> tuple | None:
    """Tight 2-D box (clipped to the frame) from a cuboid's 8 rig-frame corners, or None if it does not
    project (needs >= 2 valid corners in front of the camera)."""
    px, valid = cam.project(corners_rig)
    px = px[valid]
    if len(px) < 2:
        return None
    x1, y1 = float(np.clip(px[:, 0].min(), 0, cam.width)), float(np.clip(px[:, 1].min(), 0, cam.height))
    x2, y2 = float(np.clip(px[:, 0].max(), 0, cam.width)), float(np.clip(px[:, 1].max(), 0, cam.height))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class NuRecDataset(Dataset):
    """A real *labeled* :class:`Dataset` over nuRec ``.usdz`` scenes (peer of ``NuScenesDataset``).

    ``usdz_paths`` are the scene artifacts. For each, one :class:`SceneGT` is emitted at the camera frame
    ``keyframe`` (a fraction into the front-camera frames, 0.0 = start, 0.5 = middle), which aligns 1:1
    with the rendered ``.mp4``. ``sensor`` selects the nuRec camera (its f-theta model + extrinsic drive
    the 2-D boxes); the scene exposes it under the logical name ``camera`` (default ``front``) that
    requirements target. ``max_scenes`` caps iteration. Everything (zip read, JSON parse, projection) is
    CPU-only — no renderer needed to filter."""

    def __init__(self, usdz_paths: list[str], sensor: str = "camera_front_wide_120fov",
                 camera: str = "front", keyframe: float = 0.5,
                 vehicle_labels: tuple[str, ...] = _VEHICLE_LABELS, max_scenes: int | None = None) -> None:
        self.usdz_paths = list(usdz_paths)
        self.sensor = sensor
        self.camera = camera
        self.keyframe = keyframe
        self.vehicle_labels = tuple(vehicle_labels)
        self.max_scenes = max_scenes

    @classmethod
    def from_glob(cls, pattern: str, **kwargs) -> NuRecDataset:
        """Build over every ``.usdz`` matching a glob (recursive)."""
        import glob
        return cls(sorted(glob.glob(pattern, recursive=True)), **kwargs)

    def scenes(self) -> Iterator[SceneGT]:
        n = 0
        for path in self.usdz_paths:
            scene = self._scene_from_usdz(path)
            if scene is None:
                continue
            yield scene
            n += 1
            if self.max_scenes is not None and n >= self.max_scenes:
                return

    def _is_vehicle(self, label_class: str) -> bool:
        lc = label_class.lower()
        return any(v in lc for v in self.vehicle_labels)

    def _scene_from_usdz(self, path: str) -> SceneGT | None:
        with zipfile.ZipFile(path, "r") as zf:
            rig = json.loads(zf.read("rig_trajectories.json").decode("utf-8"))
            trk = json.loads(zf.read("sequence_tracks.json").decode("utf-8"))

        traj = rig["rig_trajectories"][0]
        fk = next(k for k in rig["camera_calibrations"] if k.split("@")[0] == self.sensor)
        scene_id = fk.split("@")[1] if "@" in fk else traj.get("sequence_id")
        cam = FThetaCamera.from_calib(rig["camera_calibrations"][fk])

        cam_ts = np.asarray(traj["cameras_frame_timestamps_us"][fk], dtype=np.int64)  # (F, 2)
        cam_T = np.asarray(traj["cameras_frame_T_rig_worlds"][fk], dtype=np.float64)  # (F, 2, 4, 4)
        fi = round(self.keyframe * (len(cam_ts) - 1))
        timestamp_us = int(cam_ts[fi].mean())
        t_rig_world = cam_T[fi, 0]                       # rig pose in the local/world frame at this frame
        # Track poses and the rig trajectory share the same frame, so an actor goes to the rig frame by
        # the inverse of the rig's own pose — exactly what AlpaSim does (it uses the tracks directly with
        # `pose_local_to_rig`; `world_to_nre` is only for the renderer and is NOT applied to the tracks).
        world_to_rig = np.linalg.inv(t_rig_world)
        ego_speed = self._ego_speed(cam_T[:, 0, :3, 3], cam_ts.mean(axis=1), fi)

        objects = self._objects(trk, timestamp_us, world_to_rig, cam)
        calib = CameraCalib(name=self.camera, width=cam.width, height=cam.height, model=cam)
        return SceneGT(frame=fi, t=timestamp_us / 1e6, ego=EgoState(speed=ego_speed),
                       cameras={self.camera: calib}, objects=objects,
                       source={"dataset": "nurec", "usdz_path": path, "scene_id": scene_id,
                               "sensor": self.sensor, "frame_index": fi, "timestamp_us": timestamp_us})

    def _objects(self, trk: dict, timestamp_us: int, world_to_rig: np.ndarray,
                 cam: FThetaCamera) -> list[ObjectGT]:
        (seq,) = trk.values()  # single-sequence reconstructions
        td = seq["tracks_data"]
        dims = seq["cuboidtracks_data"]["cuboids_dims"]
        objects: list[ObjectGT] = []
        for tid, label, aabb, ts_us, poses in zip(
                td["tracks_id"], td["tracks_label_class"], dims,
                td["tracks_timestamps_us"], td["tracks_poses"]):
            if not self._is_vehicle(str(label)):
                continue
            ts = np.asarray(ts_us, dtype=np.int64)
            if len(ts) == 0 or timestamp_us < ts.min() or timestamp_us > ts.max():
                continue                                 # actor not present at this frame
            pose = np.asarray(poses[int(np.argmin(np.abs(ts - timestamp_us)))], dtype=np.float64)
            pose_rig = world_to_rig @ _se3(_quat_to_rot(pose[3:]), pose[:3])
            center = pose_rig[:3, 3]                     # ego-frame centre (x fwd, y left, z up)
            fwd = pose_rig[:3, 0]                         # actor heading in the ego frame
            length, wdt, hgt = (float(v) for v in aabb)
            hl, hw, hh = length / 2, wdt / 2, hgt / 2
            local = np.array([[sx * hl, sy * hw, sz * hh] for sx in (-1, 1) for sy in (-1, 1)
                              for sz in (-1, 1)])
            corners = (pose_rig[:3, :3] @ local.T).T + center
            box = _box2d_from_corners(cam, corners)
            objects.append(ObjectGT(
                track_id=str(tid), category="vehicle",
                center=(float(center[0]), float(center[1]), float(center[2])),
                extent=(length, wdt, hgt), yaw=math.atan2(float(fwd[1]), float(fwd[0])),
                box2d={self.camera: box} if box else {}, visibility=1.0))
        return objects

    @staticmethod
    def _ego_speed(origins_world: np.ndarray, ts_us: np.ndarray, fi: int) -> float:
        """Ego speed (m/s) from consecutive rig world-origin positions (rig->world translation)."""
        if len(ts_us) < 2:
            return 0.0
        j = fi + 1 if fi + 1 < len(ts_us) else fi - 1
        dt = abs(float(ts_us[j]) - float(ts_us[fi])) / 1e6
        return float(np.linalg.norm(origins_world[j] - origins_world[fi]) / dt) if dt > 0 else 0.0

    def make_backend(self, scene: SceneGT) -> WorldBackend:
        """Build a live :class:`NuRecBackend` on this scene (needs the nre-ga renderer + server)."""
        from avsectester.simulators.nurec import NuRecBackend, NuRecRenderer

        scene_id = scene.source.get("scene_id")
        return NuRecBackend(config={"scene": scene_id},
                            renderer=NuRecRenderer(scene_id=scene_id, cameras=[self.sensor]))
