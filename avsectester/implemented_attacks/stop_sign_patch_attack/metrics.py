"""Replica-specific visibility measurements from saved decision captures."""
from __future__ import annotations

from avsectester.evaluation import (
    ComparisonGroup, Metric, MetricDescription, MetricField, MetricScalar,
    TreatmentMeasurements, UnavailableMetricValue,
)
from avsectester.validation import require_nonnegative_integer

from .online import REPLICA_OBJECT_ID


class ReplicaVisibility(Metric):
    """Count decisions where the inserted replica reaches a camera's pixel threshold.

    camera_id=None selects the unique recorded replica camera across the comparison group.
    Multiple cameras require an explicit camera_id. A missing camera entry is invisible. Empty runs or runs that never recorded the
    replica object have unavailable evidence rather than a zero visible-step count.
    """

    def __init__(self, camera_id: str | None = None,
                 min_visible_pixels: int = 100) -> None:
        if camera_id is not None and (not isinstance(camera_id, str) or not camera_id):
            raise ValueError("camera_id must be a nonempty sensor name or None")
        require_nonnegative_integer(min_visible_pixels, "min_visible_pixels")
        self.camera_id = camera_id
        self.min_visible_pixels = min_visible_pixels

    def describe(self) -> MetricDescription:
        return MetricDescription(
            "How long the inserted replica stop sign was visible to the camera.",
            {"visible_steps": MetricField(
                "Decisions where the replica covers at least min_visible_pixels "
                f"({self.min_visible_pixels}) visible pixels in camera_id ({self.camera_id}). "
                "Unavailable for treatments that insert no replica.",
                "decision steps",
            )},
        )

    def evaluate(self, group: ComparisonGroup) -> TreatmentMeasurements:
        camera_id = self.camera_id
        if camera_id is None:
            cameras = {camera for log in group.runs.values() for step in log.steps
                       for camera in step.object_visibility.get(REPLICA_OBJECT_ID, {})}
            if len(cameras) > 1:
                raise ValueError("Replica visibility has multiple cameras; set camera_id explicitly")
            camera_id = next(iter(cameras), None)
        measurements: dict[str, dict[str, MetricScalar]] = {}
        for name, log in group.runs.items():
            if not log.steps:
                value: MetricScalar = UnavailableMetricValue("The run has no decision steps")
            elif not any(REPLICA_OBJECT_ID in step.object_visibility for step in log.steps):
                value = UnavailableMetricValue("The run has no recorded replica object visibility")
            elif camera_id is None:
                value = UnavailableMetricValue("The replica has no recorded camera visibility")
            else:
                visible_steps = 0
                for step in log.steps:
                    visibility = step.object_visibility.get(REPLICA_OBJECT_ID, {}).get(camera_id)
                    if visibility is not None and visibility.visible_pixels >= self.min_visible_pixels:
                        visible_steps += 1
                value = visible_steps
            measurements[name] = {"visible_steps": value}
        return measurements
