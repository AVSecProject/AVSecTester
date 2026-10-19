"""Compatibility context helpers over the pure nuScenes source provider."""
from __future__ import annotations

from collections.abc import Iterator

from avsectester.scenarios.scene import SceneGT
from avsectester.scenarios.context import FilterContext
from avsectester.datasets.nuscenes import NuScenesDataset as SourceDataset
from avsectester.datasets.nuscenes import scene_from_cam_boxes, _project  # noqa: F401


class NuScenesDataset(SourceDataset):
    """Legacy integration facade; new code uses datasets.nuscenes.NuScenesDataset.

    ``scenes`` keeps the original keyframe iteration: ``nusc.sample[:max_samples]`` in SDK
    order, the global sample index as ``SceneGT.frame`` and sample-annotation tokens as object
    IDs. ``max_samples`` caps that iteration only. The source interface (``frames``,
    ``reference``, ``source_data``) uses complete per-scene timelines and stable instance IDs.
    """

    def __init__(self, dataroot: str, version: str = "v1.0-trainval", camera: str = "CAM_FRONT",
                 max_samples: int | None = None) -> None:
        super().__init__(dataroot, version, camera)
        self.max_samples = max_samples

    def scenes(self) -> Iterator[SceneGT]:
        nusc = self._load()
        samples = nusc.sample if self.max_samples is None else nusc.sample[:self.max_samples]
        for frame_index, sample in enumerate(samples):
            yield self._sample_scene(sample, frame_index)

    def initial_sequence(self, scene: SceneGT, frames: int) -> tuple[SceneGT, ...]:
        """A legacy keyframe is one recorded state; recorded windows use the source interface."""
        return (scene,)

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
        from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator

        def make_backend():
            scenario = sample["scene_token"]
            frame = next(frame for frame in self.frames(scenario) if frame.frame_id == sample["token"])
            return RecordedReplaySimulator(self.scenario_start(scenario, frame)).create_runtime()
        return FilterContext(scene, dataset=self, metadata=metadata, native={
            "nusc": nusc, "make_backend": make_backend,
        })
