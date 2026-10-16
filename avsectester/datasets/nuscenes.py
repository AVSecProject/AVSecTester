"""Read nuScenes source records without simulator dependencies."""
from __future__ import annotations

import math

import numpy as np

from avsectester.rendering.types import Visibility
from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, SceneGT
from avsectester.datasets.base import Dataset

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
                         visibility: dict[str, Visibility] | None = None,
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


class NuScenesDataset(Dataset):
    """nuScenes scenes as source timelines of linked keyframes.

    Each scene is one scenario. ``camera`` is the nuScenes sensor (exposed under the logical name
    ``front``). Actor identities are stable instance tokens. The legacy facade in
    ``avsectester.scenarios.datasets.nuscenes`` keeps the original keyframe iteration.
    """

    def __init__(self, dataroot: str, version: str = "v1.0-trainval", camera: str = "CAM_FRONT") -> None:
        self.dataroot = dataroot
        self.version = version
        self.camera = camera
        self._nusc = None

    def _load(self):
        if self._nusc is None:
            from nuscenes import NuScenes
            self._nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)
        return self._nusc

    def _samples(self, scenario_id):
        nusc = self._load()
        scene = nusc.get("scene", scenario_id)
        samples, token = [], scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            samples.append(sample)
            token = sample["next"]
        return samples

    def scenarios(self):
        from .base import ScenarioDescriptor

        for scene in self._load().scene:
            yield ScenarioDescriptor(scene["token"], "nuscenes", {
                "dataroot": self.dataroot, "version": self.version, "name": scene["name"],
            })

    def frames(self, scenario_id):
        from .base import FrameRef

        nusc = self._load()
        return tuple(FrameRef(scenario_id, i, sample["token"],
                              nusc.get("sample_data", sample["data"][self.camera])["timestamp"] / 1e6)
                     for i, sample in enumerate(self._samples(scenario_id)))

    def reference(self, scenario_id, frame):
        self.checked_frame(scenario_id, frame)
        return self._reference(frame)

    def _reference(self, frame):
        from .base import ActorState, DataReference, ReferenceFrame

        nusc = self._load()
        scene = self._sample_scene(nusc.get("sample", frame.frame_id), frame.position)
        actors = {"ego": ActorState(scene.ego.pose)}
        for obj in scene.objects:
            annotation = nusc.get("sample_annotation", obj.track_id)
            velocity = np.asarray(nusc.box_velocity(obj.track_id), dtype=float)
            obj.track_id = annotation["instance_token"]
            actors[obj.track_id] = ActorState(
                np.asarray(scene.ego.pose) @ np.asarray(obj.pose),
                tuple(float(v) for v in velocity) if np.isfinite(velocity).all() else None,
            )
        sensor = DataReference(scene.source["image_path"], "front", frame.timestamp)
        return ReferenceFrame(frame, scene, actors, {"front": sensor}, {"kind": "recorded"})

    def native(self, scenario_id):
        from copy import deepcopy

        return deepcopy({"scene": self._load().get("scene", scenario_id),
                         "samples": self._samples(scenario_id)})

    def source_data(self, scenario_id, start):
        from .base import SourceData

        frames = self.frames(scenario_id)
        if (start.scenario_id != scenario_id or start.position >= len(frames)
                or start != frames[start.position]):
            raise ValueError("Frame is outside or does not match the source sequence")
        return SourceData("nuscenes", scenario_id, start, {
            "references": tuple(self._reference(frame) for frame in frames[start.position:]),
        })

    def _sample_scene(self, sample, frame_index):
        from nuscenes.utils.geometry_utils import BoxVisibility

        nusc = self._load()
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
        return scene
