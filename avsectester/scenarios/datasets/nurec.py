"""Compatibility NuRec selection/backend helpers; source reads live in datasets.nurec."""
from __future__ import annotations

import math
from collections.abc import Iterator
import numpy as np
from avsectester.backend import WorldBackend
from avsectester.datasets.nurec import NuRecDataset as SourceDataset
from avsectester.datasets.nurec import _VEHICLE_LABELS
from avsectester.datasets.nurec import FThetaCamera, _quat_to_rot, _se3, _interpolate_pose, _box2d_from_corners  # noqa: F401
from avsectester.scenarios.scene import SceneGT
from avsectester.scenarios.source import Dataset as LegacyDataset


class NuRecDataset(SourceDataset, LegacyDataset):
    """Legacy factories retained until CLI migration; use datasets.nurec for pure source access."""

    def __init__(self, usdz_paths, sensor="camera_front_wide_120fov", camera="front",
                 keyframe=0.5, vehicle_labels=_VEHICLE_LABELS, max_scenes=None,
                 endpoint="127.0.0.1:50051", time_step=0.1):
        super().__init__(usdz_paths, sensor, camera, keyframe, vehicle_labels, max_scenes, time_step)
        self.endpoint = endpoint

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
                "dt": self.time_step,
                "ego0": ego0,
                "camera_aliases": {self.sensor: self.camera},
            },
            renderer=self.make_renderer(scene),
        )

    def insertion_renderer(self, scene, backend, insertions, bindings=None, compositor=None):
        """Compatibility callable delegating every frame to the simulator renderer."""
        from avsectester.services import RenderRequest
        from avsectester.datasets.base import FrameRef
        from avsectester.perturbations.image import read_camera_rgb, write_camera_rgb

        frame = FrameRef(str(scene.source["usdz_path"]), scene.frame, scene.frame, scene.t)
        request = RenderRequest(frame.scenario_id, frame, self.camera, insertions,
                                bindings or {}, scene=scene, compositor=compositor)
        def render(observation, image):
            sample = read_camera_rgb(observation, self.camera)
            key = sample[0] if sample is not None else self.camera
            result = backend.render(write_camera_rgb(observation, key, image), request)
            render.evidence, render.resolved = result.evidence, result.resolved
            render.camera = backend.renderer.camera_model(backend.render_request(self.sensor).camera)
            return read_camera_rgb(result.observation, self.camera)[1]
        # Preserve geometry inspection on the legacy callable; rendering itself delegates.
        adapter = backend._insertion_renderer(scene, self, insertions, bindings, compositor)
        render.geometry, render.camera = adapter.geometry, adapter.camera
        render.evidence, render.resolved = {}, ()
        return render
