"""Read nuScenes records without simulator dependencies."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from avsectester.data_types import ActorState
from avsectester.datasets.base import Dataset, FrameData, SensorDataReference
from avsectester.geometry.cameras import PinholeCamera
from avsectester.geometry.poses import make_pose, quaternion_to_rotation


class NuScenesDataset(Dataset):
    """nuScenes scenes as scenario timelines of linked keyframes.

    Each scene is one scenario. ``camera`` is the native nuScenes sensor name. Simulator aliases
    are configured separately. Actor identities are stable instance tokens, including annotations
    outside the selected camera and all annotated classes.
    """

    def __init__(self, dataroot: str | Path, version: str = "v1.0-trainval", camera: str = "CAM_FRONT") -> None:
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
            yield ScenarioDescriptor(scene["token"], "NuScenesDataset", {
                "dataroot": self.dataroot, "version": self.version, "name": scene["name"],
            })

    def frames(self, scenario_id):
        from .base import FrameReference

        nusc = self._load()
        return tuple(FrameReference(scenario_id, i, sample["token"],
                              nusc.get("sample_data", sample["data"][self.camera])["timestamp"] / 1e6)
                     for i, sample in enumerate(self._samples(scenario_id)))

    def frame_data(self, scenario_id, frame):
        self.checked_frame(scenario_id, frame)
        return self._frame_data(frame)

    def _ego_state(self, sample, sensor_record) -> ActorState:
        nusc = self._load()
        ego = nusc.get("ego_pose", sensor_record["ego_pose_token"])
        pose = make_pose(quaternion_to_rotation(*ego["rotation"]), ego["translation"])
        velocity = None
        adjacent_token = sample.get("next") or sample.get("prev")
        if adjacent_token:
            adjacent = nusc.get("sample", adjacent_token)
            other_sd = nusc.get("sample_data", adjacent["data"][self.camera])
            other_pose = nusc.get("ego_pose", other_sd["ego_pose_token"])
            dt = (other_sd["timestamp"] - sensor_record["timestamp"]) / 1e6
            if dt:
                velocity = tuple(float(v) for v in (
                    np.asarray(other_pose["translation"]) - np.asarray(ego["translation"])
                ) / dt)
        return ActorState(pose, velocity=velocity, category="vehicle")

    def _frame_data(self, frame):
        nusc = self._load()
        sample = nusc.get("sample", frame.frame_id)
        camera_token = sample["data"][self.camera]
        sensor_record = nusc.get("sample_data", camera_token)
        ego = self._ego_state(sample, sensor_record)
        actors = {}
        for annotation_id in sample.get("anns", ()):
            annotation = nusc.get("sample_annotation", annotation_id)
            width, length, height = annotation["size"]
            velocity = np.asarray(nusc.box_velocity(annotation_id), dtype=float)
            actors[annotation["instance_token"]] = ActorState(
                make_pose(quaternion_to_rotation(*annotation["rotation"]), annotation["translation"]),
                tuple(float(v) for v in velocity) if np.isfinite(velocity).all() else None,
                annotation["category_name"], (length, width, height),
            )
        calibration = nusc.get("calibrated_sensor", sensor_record["calibrated_sensor_token"])
        ego_from_sensor = make_pose(
            quaternion_to_rotation(*calibration["rotation"]), calibration["translation"],
        )
        camera = PinholeCamera(
            np.array(calibration["camera_intrinsic"], dtype=float, copy=True),
            sensor_record["width"], sensor_record["height"],
        )
        sensor = SensorDataReference(
            str(Path(self.dataroot) / sensor_record["filename"]), self.camera, frame.timestamp,
            {"camera": camera, "ego_from_sensor": ego_from_sensor,
             "sample_data_token": camera_token},
        )
        return FrameData(frame, actors=actors, sensor_references={self.camera: sensor},
                         provenance={"kind": "recorded"}, ego=ego)

    def native(self, scenario_id):
        from copy import deepcopy

        return deepcopy({"scene": self._load().get("scene", scenario_id),
                         "samples": self._samples(scenario_id)})

    def scenario_start(self, scenario_id, frame):
        from .base import ScenarioStart

        frames = self.frames(scenario_id)  # read the scene's sample chain once
        if (frame.scenario_id != scenario_id or frame.position >= len(frames)
                or frame != frames[frame.position]):
            raise ValueError("Frame is outside or does not match the scenario's frames")
        data = tuple(self._frame_data(item) for item in frames)
        return ScenarioStart("NuScenesDataset", scenario_id, frame, {
            "frame_data": data[frame.position:],
        }, recorded_ego=tuple(item.ego for item in data), scenario_map=self.scenario_map(scenario_id))
