"""nuRec (NVIDIA NRE) ``.usdz`` -> :class:`SceneGT`: filter real nuRec scenes by an attack requirement.

Each nuRec scene is a ``.usdz`` (a ZIP) reconstructed from a real driving clip (scene ids
``clipgt-<uuid>``). The render RPC returns pixels only, but the ``.usdz`` itself **ships ground truth**:

* ``sequence_tracks.json`` — 7-DoF actor cuboid tracks (``track_id``, ``label_class``, per-timestamp
  pose = position + quaternion, and cuboid dims);
* ``rig_trajectories.json`` — the ego rig SE3 trajectory (``T_rig_worlds`` + timestamps), the
  ``world_to_nre`` transform, and the camera logical names / frame timestamps.

:class:`NuRecDataset` reads them and, at a chosen keyframe, expresses every actor in the **rig frame** —
which is exactly our ego frame (x forward, y left, z up; AlpaSim ``CONTRIBUTING.md`` "Coordinate
Systems") — to build a :class:`SceneGT` the requirement predicate runs on. This puts nuRec on the same
tier as :class:`~avsectester.scenarios.datasets.nuscenes.NuScenesDataset` (a real *labeled* dataset), with
no renderer or GPU needed for filtering.

Two documented approximations, both isolated and easy to replace once a real ``.usdz`` is on hand:

* ``box2d`` is filled by projecting the cuboid's 8 corners through a **pinhole built from the camera's
  horizontal FOV** (``hfov_deg``, default 120° for the wide front camera). This is exact near the
  boresight — where a lead vehicle sits, i.e. the physical-patch target — and avoids depending on the
  f-theta intrinsic, whose JSON layout is not available offline.
* ``visibility`` is set to 1.0 (the tracks carry no per-camera occlusion fraction).

The zip/JSON field names mirror ``alpasim_utils.scenario`` (``Rig.load_from_json`` /
``TrafficObjects.load_from_json``). ``cv2``/zip are imported lazily; the module imports offline.

.. warning::
   **UNVALIDATED against a real ``.usdz``.** No real nuRec scene is available on this machine (they are
   git-LFS / HF-license-gated, multi-GB, and the ``-NuRec`` HF dataset is not downloaded). The reader is
   written to AlpaSim's documented schema, but four points are *assumptions* only a real artifact can
   confirm: (1) the quaternion order in ``tracks_poses`` (assumed ``xyzw``, scipy); (2) the
   ``T_rig_world`` direction + ``world_to_nre`` composition; (3) the ``label_class`` strings; (4) the true
   camera intrinsic (a pinhole-from-FOV stands in for the wide/f-theta model). Do **not** treat filter
   results as trustworthy until a real ``.usdz`` has been read and these are checked.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Iterator

import numpy as np

from avsectester.backend import WorldBackend
from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, SceneGT
from avsectester.scenarios.source import Dataset

# nuRec ``label_class`` substrings that count as a 4-wheeled "vehicle" (a rear-surface patch target).
_VEHICLE_LABELS = ("car", "truck", "bus", "trailer", "van", "vehicle", "automobile")


def _quat_to_rot(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Rotation matrix from a unit quaternion. ``order`` is ``"xyzw"`` (scipy, AlpaSim's internal
    convention) or ``"wxyz"`` (gRPC order)."""
    q = np.asarray(quat, dtype=np.float64)
    x, y, z, w = (q[0], q[1], q[2], q[3]) if order == "xyzw" else (q[1], q[2], q[3], q[0])
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _se3(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = np.asarray(trans, dtype=np.float64)
    return m


def _pinhole_k(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Pinhole intrinsic from image size + horizontal FOV (``fx=fy``, principal point at centre)."""
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def _vehicle_category(label_class: str, vehicle_labels: tuple[str, ...]) -> str | None:
    lc = label_class.lower()
    return "vehicle" if any(v in lc for v in vehicle_labels) else None


def scene_from_tracks(tracks: list[dict], world_to_nre: np.ndarray, t_rig_world: np.ndarray,
                      k: np.ndarray, width: int, height: int, timestamp_us: int,
                      ego_speed: float = 0.0, camera: str = "front", quat_order: str = "xyzw",
                      vehicle_labels: tuple[str, ...] = _VEHICLE_LABELS) -> SceneGT:
    """Build a :class:`SceneGT` from decoded nuRec tracks at one keyframe. Pure — takes plain dicts, so
    it is testable without a real ``.usdz``.

    Each ``track`` is ``{track_id, label_class, dims (l,w,h), pos (x,y,z, NRE frame), quat (4,)}`` at this
    keyframe. ``t_rig_world`` maps a **world** point to the **rig** frame; ``world_to_nre`` maps world ->
    NRE (tracks are exported in NRE coords), so an actor goes NRE -> world -> rig via
    ``t_rig_world @ inv(world_to_nre)``. The rig frame is our ego frame (x fwd, y left, z up); the front
    camera frame used for projection is (x right, y down, z fwd) = ``(-y, -z, x)`` of the ego frame."""
    nre_to_rig = np.asarray(t_rig_world, dtype=np.float64) @ np.linalg.inv(np.asarray(world_to_nre, float))
    k = np.asarray(k, dtype=np.float64)
    objects: list[ObjectGT] = []
    for tr in tracks:
        category = _vehicle_category(tr["label_class"], vehicle_labels)
        if category is None:
            continue
        pose_nre = _se3(_quat_to_rot(tr["quat"], quat_order), tr["pos"])
        pose_rig = nre_to_rig @ pose_nre
        center = pose_rig[:3, 3]                          # ego-frame centre (x fwd, y left, z up)
        fwd = pose_rig[:3, 0]                             # actor +x (heading) in the ego frame
        yaw = math.atan2(float(fwd[1]), float(fwd[0]))    # 0 => same heading as ego (rear faces us)
        length, wdt, hgt = (float(v) for v in tr["dims"])

        # 8 cuboid corners in the actor frame -> ego frame -> camera frame -> pixels
        hl, hw, hh = length / 2, wdt / 2, hgt / 2
        local = np.array([[sx * hl, sy * hw, sz * hh] for sx in (-1, 1) for sy in (-1, 1)
                          for sz in (-1, 1)], dtype=np.float64)
        corners_ego = (pose_rig[:3, :3] @ local.T).T + center
        cam = np.stack([-corners_ego[:, 1], -corners_ego[:, 2], corners_ego[:, 0]], axis=1)
        box2d: dict[str, tuple[float, float, float, float]] = {}
        infront = cam[:, 2] > 0.1
        if infront.any():
            uv = (k @ cam[infront].T)
            px = uv[:2] / uv[2:3]
            x1, y1 = float(np.clip(px[0].min(), 0, width)), float(np.clip(px[1].min(), 0, height))
            x2, y2 = float(np.clip(px[0].max(), 0, width)), float(np.clip(px[1].max(), 0, height))
            if x2 > x1 and y2 > y1:
                box2d[camera] = (x1, y1, x2, y2)

        objects.append(ObjectGT(
            track_id=str(tr["track_id"]), category=category,
            center=(float(center[0]), float(center[1]), float(center[2])),
            extent=(length, wdt, hgt), yaw=yaw, box2d=box2d, visibility=1.0))

    calib = CameraCalib(name=camera, width=width, height=height, model=k)
    return SceneGT(frame=0, t=timestamp_us / 1e6, ego=EgoState(speed=ego_speed),
                   cameras={camera: calib}, objects=objects)


class NuRecDataset(Dataset):
    """A real *labeled* :class:`Dataset` over nuRec ``.usdz`` scenes (peer of ``NuScenesDataset``).

    ``usdz_paths`` are the scene artifacts. For each, one :class:`SceneGT` is emitted at ``keyframe`` (a
    fraction into the rig trajectory, 0.0 = start). ``sensor`` selects the nuRec camera + its ``hfov_deg``;
    the scene exposes it under the logical name ``camera`` (default ``front``) that requirements target.
    ``width``/``height`` are the render resolution used to build the pinhole + the 2-D boxes.
    ``max_scenes`` caps iteration. Everything (zip read, JSON parse, projection) is CPU-only."""

    def __init__(self, usdz_paths: list[str], sensor: str = "camera_front_wide_120fov",
                 camera: str = "front", hfov_deg: float = 120.0, width: int = 1920, height: int = 1080,
                 keyframe: float = 0.0, quat_order: str = "xyzw",
                 vehicle_labels: tuple[str, ...] = _VEHICLE_LABELS, max_scenes: int | None = None) -> None:
        self.usdz_paths = list(usdz_paths)
        self.sensor = sensor
        self.camera = camera
        self.hfov_deg = hfov_deg
        self.width = width
        self.height = height
        self.keyframe = keyframe
        self.quat_order = quat_order
        self.vehicle_labels = tuple(vehicle_labels)
        self.max_scenes = max_scenes

    @classmethod
    def from_glob(cls, pattern: str, **kwargs) -> NuRecDataset:
        """Build over every ``.usdz`` matching a glob (recursive)."""
        import glob
        return cls(sorted(glob.glob(pattern, recursive=True)), **kwargs)

    def scenes(self) -> Iterator[SceneGT]:
        k = _pinhole_k(self.width, self.height, self.hfov_deg)
        n = 0
        for path in self.usdz_paths:
            scene = self._scene_from_usdz(path, k)
            if scene is None:
                continue
            yield scene
            n += 1
            if self.max_scenes is not None and n >= self.max_scenes:
                return

    def _scene_from_usdz(self, path: str, k: np.ndarray) -> SceneGT | None:
        with zipfile.ZipFile(path, "r") as zf:
            rig = json.loads(zf.read("rig_trajectories.json").decode("utf-8"))
            trk = json.loads(zf.read("sequence_tracks.json").decode("utf-8"))

        world_to_nre = np.asarray(rig["world_to_nre"]["matrix"], dtype=np.float64)
        traj = rig["rig_trajectories"][0]
        rig_ts = np.asarray(traj["T_rig_world_timestamps_us"], dtype=np.int64)
        rig_mats = np.asarray(traj["T_rig_worlds"], dtype=np.float64)  # (T, 4, 4), world->rig
        ki = round(self.keyframe * (len(rig_ts) - 1))
        timestamp_us = int(rig_ts[ki])
        t_rig_world = rig_mats[ki]
        ego_speed = self._ego_speed(rig_mats, rig_ts, ki)

        tracks = self._decode_tracks(trk, timestamp_us)
        scene = scene_from_tracks(
            tracks, world_to_nre, t_rig_world, k, self.width, self.height, timestamp_us,
            ego_speed=ego_speed, camera=self.camera, quat_order=self.quat_order,
            vehicle_labels=self.vehicle_labels)
        scene.source = {"dataset": "nurec", "usdz_path": path, "sensor": self.sensor,
                        "timestamp_us": timestamp_us}
        return scene

    @staticmethod
    def _ego_speed(rig_mats: np.ndarray, rig_ts: np.ndarray, ki: int) -> float:
        """Ego speed (m/s) from consecutive rig world-origin positions (rig origin in world = -R^T t)."""
        if len(rig_ts) < 2:
            return 0.0
        j = ki + 1 if ki + 1 < len(rig_ts) else ki - 1
        def origin(m):
            return -m[:3, :3].T @ m[:3, 3]
        dt = abs(int(rig_ts[j]) - int(rig_ts[ki])) / 1e6
        return float(np.linalg.norm(origin(rig_mats[j]) - origin(rig_mats[ki])) / dt) if dt > 0 else 0.0

    def _decode_tracks(self, trk: dict, timestamp_us: int) -> list[dict]:
        """Decode ``sequence_tracks.json`` (mirrors ``alpasim_utils.scenario.TrafficObjects``), taking
        each track's pose at the timestamp nearest ``timestamp_us``."""
        (seq,) = trk.values()  # single-sequence reconstructions (as AlpaSim asserts)
        td = seq["tracks_data"]
        dims = seq["cuboidtracks_data"]["cuboids_dims"]
        out = []
        for tid, label, _flags, aabb, ts_us, poses in zip(
                td["tracks_id"], td["tracks_label_class"], td["tracks_flags"], dims,
                td["tracks_timestamps_us"], td["tracks_poses"]):
            ts = np.asarray(ts_us, dtype=np.int64)
            if len(ts) == 0:
                continue
            i = int(np.argmin(np.abs(ts - timestamp_us)))
            pose = np.asarray(poses[i], dtype=np.float64)  # [x, y, z, q0..q3]
            out.append({"track_id": tid, "label_class": str(label), "dims": aabb,
                        "pos": pose[:3], "quat": pose[3:]})
        return out

    def make_backend(self, scene: SceneGT) -> WorldBackend:
        """Build a live :class:`NuRecBackend` on this scene (needs the nre-ga renderer + server)."""
        from avsectester.simulators.nurec import NuRecBackend, NuRecRenderer

        scene_id = None  # resolved by the renderer from the loaded .usdz (scene id == metadata.scene_id)
        return NuRecBackend(config={"scene": scene_id},
                            renderer=NuRecRenderer(scene_id=scene_id, cameras=[self.sensor]))
