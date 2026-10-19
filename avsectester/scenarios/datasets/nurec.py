"""Compatibility NuRec selection/backend helpers; source reads live in datasets.nurec."""
from __future__ import annotations

import math
from collections.abc import Iterator
import numpy as np
from avsectester.closed_loop.simulator import WorldBackend
from avsectester.datasets.nurec import NuRecDataset as SourceDataset
from avsectester.datasets.nurec import _VEHICLE_LABELS
from avsectester.datasets.nurec import FThetaCamera, _quat_to_rot, _se3, _interpolate_pose, _box2d_from_corners  # noqa: F401
from avsectester.scenarios.scene import SceneGT
from avsectester.scenarios.source import Dataset as LegacyDataset


class NuRecDataset(SourceDataset, LegacyDataset):
    """Legacy factories retained until CLI migration; use datasets.nurec for pure source access.

    Keeps the original parameters: ``sensor`` and ``camera`` are the new ``camera_id`` and
    ``camera_alias``; ``keyframe`` and ``max_scenes`` control only the legacy ``scenes()``.
    """

    def __init__(self, usdz_paths, sensor="camera_front_wide_120fov", camera="front",
                 keyframe=0.5, vehicle_labels=_VEHICLE_LABELS, max_scenes=None,
                 endpoint="127.0.0.1:50051", time_step=0.1):
        if not 0 <= keyframe <= 1:
            raise ValueError("keyframe must be in [0, 1]")
        if max_scenes is not None and max_scenes < 0:
            raise ValueError("max_scenes must be nonnegative")
        super().__init__(usdz_paths, camera_id=sensor, camera_alias=camera,
                         vehicle_labels=vehicle_labels, time_step=time_step)
        self.keyframe, self.max_scenes = keyframe, max_scenes
        self.endpoint = endpoint

    @property
    def sensor(self) -> str:
        return self.camera_id

    @property
    def camera(self) -> str:
        return self.camera_alias

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
        """The scene at a recorded camera frame (legacy indexing), not a decision position."""
        _raw, _traj, _key, _cam, timestamps, transforms = self._camera_data(path)
        if not 0 <= frame < len(timestamps):
            raise IndexError("camera frame is outside the recorded sequence")
        # Render at the first exposure timestamp, matching this exact rig pose and actor time.
        return self._scene(
            path, frame, int(timestamps[frame, 0]), transforms[frame, 0],
            self._ego_speed(transforms[:, 0, :3, 3], timestamps[:, 0], frame),
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
        from avsectester.closed_loop.simulators.nurec import NuRecRenderer

        return NuRecRenderer(
            endpoint=self.endpoint,
            scene_id=scene.source.get("scene_id"),
            cameras=[self.sensor],
            start_timestamp_us=scene.source["timestamp_us"],
            start_transform=scene.ego.pose,
        )

    def make_backend(self, scene: SceneGT) -> WorldBackend:
        from avsectester.closed_loop.simulators.nurec import NuRecSimulator

        pose = np.asarray(scene.ego.pose)
        ego0 = {
            "x": float(pose[0, 3]),
            "y": float(pose[1, 3]),
            "yaw": math.atan2(pose[1, 0], pose[0, 0]),
            "speed": scene.ego.speed,
        }
        return NuRecSimulator(
            config={
                "scene": scene.source.get("scene_id"),
                "dt": self.time_step,
                "ego0": ego0,
                "camera_aliases": {self.sensor: self.camera},
            },
            renderer=self.make_renderer(scene),
        )

    def insertion_renderer(self, scene, backend, insertions, bindings=None, compositor=None):
        """Legacy ``(observation, rgb) -> rgb`` insertion renderer for this recorded scene."""
        return backend._insertion_renderer(scene, self, insertions, bindings, compositor)
