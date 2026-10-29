"""Replica-specific visibility measurements from saved decision captures."""
from __future__ import annotations

from collections.abc import Sequence

from avsectester.evaluation import (
    ComparisonGroup, Metric, MetricDescription, MetricField, MetricScalar,
    TreatmentMeasurements, UnavailableMetricValue,
)
from avsectester.validation import require_nonnegative_integer

from .online import REPLICA_OBJECT_ID


class ReplicaVisibility(Metric):
    """Count decisions where the inserted replica reaches the pixel threshold in a selected camera.

    camera_ids=None selects every camera that recorded replica visibility. A decision counts when
    at least one selected camera meets min_visible_pixels; a missing camera entry is invisible.
    Empty runs or runs that never recorded the replica object have unavailable evidence rather
    than a zero visible-step count.
    """

    def __init__(self, camera_ids: Sequence[str] | None = None,
                 min_visible_pixels: int = 100) -> None:
        if camera_ids is not None:
            if isinstance(camera_ids, str):
                raise ValueError("camera_ids must be a sequence of sensor names, not one string")
            camera_ids = tuple(camera_ids)
            if not camera_ids or any(not isinstance(camera, str) or not camera for camera in camera_ids):
                raise ValueError("camera_ids must be nonempty sensor names or None")
        require_nonnegative_integer(min_visible_pixels, "min_visible_pixels")
        self.camera_ids = camera_ids
        self.min_visible_pixels = min_visible_pixels

    def describe(self) -> MetricDescription:
        cameras = "every recorded camera" if self.camera_ids is None else ", ".join(self.camera_ids)
        return MetricDescription(
            "How long the inserted replica stop sign was visible to the driving system's cameras.",
            {"visible_steps": MetricField(
                "Decisions where the replica covers at least min_visible_pixels "
                f"({self.min_visible_pixels}) visible pixels in at least one of: {cameras}. "
                "Unavailable for treatments that insert no replica.",
                "decision steps",
            )},
        )

    def evaluate(self, group: ComparisonGroup) -> TreatmentMeasurements:
        measurements: dict[str, dict[str, MetricScalar]] = {}
        for name, log in group.runs.items():
            if not log.steps:
                value: MetricScalar = UnavailableMetricValue("The run has no decision steps")
            elif not any(REPLICA_OBJECT_ID in step.object_visibility for step in log.steps):
                value = UnavailableMetricValue("The run has no recorded replica object visibility")
            else:
                value = 0
                for step in log.steps:
                    cameras = step.object_visibility.get(REPLICA_OBJECT_ID, {})
                    selected = cameras if self.camera_ids is None else self.camera_ids
                    if any(cameras[camera].visible_pixels >= self.min_visible_pixels
                           for camera in selected if camera in cameras):
                        value += 1
            measurements[name] = {"visible_steps": value}
        return measurements


__all__ = [
    'ReplicaVisibility',
]
