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
                         visibility: dict[str, float] | None = None) -> SceneGT:
    """Build a :class:`SceneGT` from nuScenes ``Box``es already in the camera frame + intrinsic ``k``.

    Each box carries ``.center`` (cam frame: x right, y down, z fwd), ``.corners()`` ``(3,8)``,
    ``.orientation`` (a quaternion with ``.rotate``), ``.name`` and ``.token``. Pure — takes duck-typed
    boxes, so it is testable without the devkit. Camera ~aligned with the ego, so ego frame
    (x fwd, y left, z up) is ``(z, -x, -y)`` of the camera frame; ``yaw`` is the vehicle heading relative
    to the camera forward (0 = its rear faces us)."""
    visibility = visibility or {}
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
        cx_r, cy_d, cz_f = (float(v) for v in b.center)
        fwd = b.orientation.rotate(np.array([1.0, 0.0, 0.0]))  # vehicle heading in the camera frame
        w, length, h = (float(v) for v in b.wlh)  # nuScenes Box.wlh = (width, length, height)
        objects.append(ObjectGT(
            track_id=b.token, category="vehicle",
            center=(cz_f, -cx_r, -cy_d),  # camera (right,down,fwd) -> ego (fwd,left,up)
            extent=(length, w, h),
            yaw=math.atan2(-float(fwd[0]), float(fwd[2])),  # ego yaw about +z (ego_y = -cam_x)
            box2d={camera: (x1, y1, x2, y2)},
            visibility=visibility.get(b.token, 1.0)))
    calib = CameraCalib(name=camera, width=width, height=height, model=np.asarray(k))
    return SceneGT(frame=0, t=0.0, ego=EgoState(speed=0.0), cameras={camera: calib}, objects=objects)


class RecordedFrameBackend(WorldBackend):
    """A ``WorldBackend`` that serves one recorded image (real data has no dynamics to step).

    ``reset``/``step`` return the same frame under ``sensor_id`` (default ``"front"``), so the standard
    run loop + a sensor-level composite attack work on a real nuScenes frame. Control is ignored."""

    def __init__(self, image_path: str, sensor_id: str = "front") -> None:
        self.image_path = image_path
        self.sensor_id = sensor_id
        self._rgb = None

    def _observe(self) -> Observation:
        if self._rgb is None:
            import cv2
            bgr = cv2.imread(self.image_path)
            self._rgb = bgr[:, :, ::-1].copy() if bgr is not None else None
        return Observation(t=0.0, frame=0, sensor_data={self.sensor_id: self._rgb})

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
        for sample in samples:
            cam_token = sample["data"][self.camera]
            sd = nusc.get("sample_data", cam_token)
            path, boxes, k = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
            vis = {b.token: _VISIBILITY.get(
                nusc.get("sample_annotation", b.token)["visibility_token"], 1.0) for b in boxes}
            scene = scene_from_cam_boxes(boxes, np.asarray(k), sd["width"], sd["height"],
                                         visibility=vis)
            scene.source = {"dataset": "nuscenes", "sample_token": sample["token"],
                            "camera": self.camera, "image_path": path}
            yield scene

    def make_backend(self, scene: SceneGT) -> WorldBackend:
        return RecordedFrameBackend(scene.source["image_path"])
