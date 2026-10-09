"""nuScenes -> :class:`SceneGT`: filter real nuScenes frames by an attack's scenario requirement.

``NuScenesDataset`` iterates keyframe samples, reads the front camera's ground-truth 3-D boxes (via the
devkit), and builds a :class:`~avsectester.scenarios.scene.SceneGT` the requirement predicate runs on;
``make_backend`` serves the recorded image (a :class:`RecordedFrameBackend`) so a sensor-level patch
attack can be composited onto the real frame. The box->SceneGT conversion (:func:`scene_from_cam_boxes`)
is a pure function tested independently of the devkit.

The devkit + nuScenes data are imported lazily, so this module imports without them.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from avsectester.backend import WorldBackend
from avsectester.plane import Control, Observation
from avsectester.rendering.types import Visibility
from avsectester.scenarios.context import FilterContext
from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, SceneGT
from avsectester.scenarios.source import Dataset

# nuScenes visibility token (fraction of the object visible across all cameras) -> a representative value
_VISIBILITY = {"1": 0.2, "2": 0.5, "3": 0.7, "4": 0.9}
# nuScenes categories that count as a (4-wheeled) "vehicle" for a rear-surface patch
_VEHICLE = {"vehicle.car", "vehicle.truck", "vehicle.bus.rigid", "vehicle.bus.bendy",
            "vehicle.trailer", "vehicle.construction", "vehicle.emergency.police",
            "vehicle.emergency.ambulance"}


def _project(pts_cam: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Pinhole-project camera-frame points ``(3, N)`` through intrinsic ``k`` -> ``(2, N)`` pixels."""
    uv = k @ pts_cam
    return uv[:2] / uv[2:3]


def scene_from_cam_boxes(boxes: list, k: np.ndarray, width: int, height: int, camera: str = "front",
                         visibility: dict[str, Visibility | float] | None = None,
                         cam_to_ego: np.ndarray | None = None) -> SceneGT:
    """Build a :class:`SceneGT` from nuScenes ``Box``es already in the camera frame + intrinsic ``k``.

    Each box carries ``.center`` (cam frame: x right, y down, z fwd), ``.corners()`` ``(3,8)``,
    ``.orientation`` (a quaternion with ``.rotate``), ``.name`` and ``.token``. Pure — takes duck-typed
    boxes, so it is testable without the devkit. ``cam_to_ego`` applies the calibrated camera pose.
    If omitted by a synthetic caller, the camera is assumed co-located and aligned with ego forward.
    The dataset adapter always passes its recorded extrinsic calibration."""
    visibility = visibility or {}
    if cam_to_ego is None:
        cam_to_ego = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    cam_to_ego = np.asarray(cam_to_ego)
    objects = []
    for b in boxes:
        if b.name not in _VEHICLE:
            continue
        corners = np.asarray(b.corners())  # (3, 8), camera frame
        if (corners[2] <= 0.1).all():      # entirely behind the camera
            continue
        px = _project(corners, k)
        x1, y1 = float(np.clip(px[0].min(), 0, width)), float(np.clip(px[1].min(), 0, height))
        x2, y2 = float(np.clip(px[0].max(), 0, width)), float(np.clip(px[1].max(), 0, height))
        if x2 <= x1 or y2 <= y1:
            continue
        center = cam_to_ego[:3, :3] @ np.asarray(b.center) + cam_to_ego[:3, 3]
        fwd = cam_to_ego[:3, :3] @ b.orientation.rotate(np.array([1.0, 0.0, 0.0]))
        pose = np.eye(4)
        pose[:3, :3] = cam_to_ego[:3, :3] @ np.column_stack(
            [b.orientation.rotate(axis) for axis in np.eye(3)])
        pose[:3, 3] = center
        w, length, h = (float(v) for v in b.wlh)  # nuScenes Box.wlh = (width, length, height)
        objects.append(ObjectGT(
            track_id=b.token, category="vehicle",
            center=tuple(float(v) for v in center),
            extent=(length, w, h),
            yaw=math.atan2(float(fwd[1]), float(fwd[0])),
            box2d={camera: (x1, y1, x2, y2)},
            visibility=visibility.get(b.token), pose=pose))
    calib = CameraCalib(name=camera, width=width, height=height, model=np.asarray(k),
                        cam_to_ego=cam_to_ego)
    return SceneGT(frame=0, t=0.0, ego=EgoState(speed=0.0), cameras={camera: calib}, objects=objects)


class RecordedFrameBackend(WorldBackend):
    """A ``WorldBackend`` that serves one recorded image (real data has no dynamics to step).

    ``reset``/``step`` return the same frame under ``sensor_id`` (default ``"front"``), so the standard
    run loop + a sensor-level composite attack work on a real nuScenes frame. Control is ignored."""

    def __init__(self, image_path: str, sensor_id: str = "front",
                 frame: int = 0, t: float = 0.0) -> None:
        self.image_path = image_path
        self.sensor_id = sensor_id
        self.frame, self.t = frame, t
        self._rgb = None

    def _observe(self) -> Observation:
        if self._rgb is None:
            import cv2
            bgr = cv2.imread(self.image_path)
            self._rgb = bgr[:, :, ::-1].copy() if bgr is not None else None
        return Observation(t=self.t, frame=self.frame, sensor_data={self.sensor_id: self._rgb})

    def reset(self) -> Observation:
        return self._observe()

    def step(self, control: Control) -> Observation:
        return self._observe()

    def close(self) -> None:
        self._rgb = None


class NuScenesDataset(Dataset):
    """A :class:`~avsectester.scenarios.source.Dataset` over nuScenes: keyframes -> ``SceneGT`` from GT.

    Pair with :class:`~avsectester.scenarios.source.DatasetFilter` to keep only the frames satisfying an
    attack's requirement. ``camera`` is the nuScenes sensor (exposed in ``SceneGT`` under the logical name
    ``front``); ``max_samples`` caps iteration for a quick pass over the (large) trainval split."""

    def __init__(self, dataroot: str, version: str = "v1.0-trainval", camera: str = "CAM_FRONT",
                 max_samples: int | None = None) -> None:
        self.dataroot = dataroot
        self.version = version
        self.camera = camera
        self.max_samples = max_samples
        self._nusc = None

    def _load(self):
        if self._nusc is None:
            from nuscenes import NuScenes
            self._nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)
        return self._nusc

    def scenes(self) -> Iterator[SceneGT]:
        from nuscenes.utils.geometry_utils import BoxVisibility

        nusc = self._load()
        samples = nusc.sample if self.max_samples is None else nusc.sample[:self.max_samples]
        for frame_index, sample in enumerate(samples):
            cam_token = sample["data"][self.camera]
            sd = nusc.get("sample_data", cam_token)
            path, boxes, k = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
            vis = {}
            for b in boxes:
                label = nusc.get("sample_annotation", b.token)["visibility_token"]
                if label in _VISIBILITY:
                    vis[b.token] = Visibility(_VISIBILITY[label], "nuscenes_label", label=label)
            from pyquaternion import Quaternion
            calibration = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            cam_to_ego = np.eye(4)
            cam_to_ego[:3, :3] = Quaternion(calibration["rotation"]).rotation_matrix
            cam_to_ego[:3, 3] = calibration["translation"]
            scene = scene_from_cam_boxes(boxes, np.asarray(k), sd["width"], sd["height"],
                                         visibility=vis, cam_to_ego=cam_to_ego)
            scene.frame = frame_index
            scene.t = float(sd["timestamp"]) / 1e6
            ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
            world_from_ego = np.eye(4)
            world_from_ego[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
            world_from_ego[:3, 3] = ego_pose["translation"]
            scene.ego.pose = world_from_ego
            scene.source = {"dataset": "nuscenes", "sample_token": sample["token"],
                            "camera": self.camera, "image_path": path}
            yield scene

    def context(self, scene: SceneGT) -> FilterContext:
        """Expose original records and the complete devkit API to custom initial filters.

        Object visibility labels apply to those annotations, not to newly inserted patches.
        Insertion visibility remains unknown unless the user supplies a measurement provider.
        """
        nusc = self._load()
        sample = nusc.get("sample", scene.source["sample_token"])
        sample_data = nusc.get("sample_data", sample["data"][self.camera])
        metadata = {
            "sample": sample,
            "sample_data": sample_data,
            "calibrated_sensor": nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"]),
            "ego_pose": nusc.get("ego_pose", sample_data["ego_pose_token"]),
        }
        return FilterContext(scene, dataset=self, metadata=metadata, native={"nusc": nusc})

    def make_backend(self, scene: SceneGT) -> WorldBackend:
        return RecordedFrameBackend(scene.source["image_path"], frame=scene.frame, t=scene.t)
