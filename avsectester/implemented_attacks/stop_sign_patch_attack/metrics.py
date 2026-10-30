"""Replica visibility and sign-detection measurements from saved decision captures."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from avsectester.evaluation import (
    ComparisonGroup, Metric, MetricDescription, MetricField, MetricScalar,
    TreatmentMeasurements, UnavailableMetricValue, signed_stop_line_distances,
)
from avsectester.runlog import RunLog
from avsectester.validation import require_nonnegative_integer, require_nonnegative_number

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


class SignDetection(Metric):
    """Count decisions where the driving system logged a sign detection.

    Detections are the ``detections`` entry of each decision's component log, as written by
    RuleBasedAVStack. A decision is visible when the inserted replica covers at least
    min_visible_pixels in camera_id, the camera the detector reads. The first-detection distance
    uses the segment's ``stop_line_points`` and the signed distance of StopLineMetrics.
    """

    def __init__(self, camera_id: str, min_visible_pixels: int = 100,
                 endpoint_tolerance_m: float = 3.) -> None:
        if not isinstance(camera_id, str) or not camera_id:
            raise ValueError("camera_id must be a nonempty sensor name")
        require_nonnegative_integer(min_visible_pixels, "min_visible_pixels")
        require_nonnegative_number(endpoint_tolerance_m, "endpoint_tolerance_m")
        self.camera_id = camera_id
        self.min_visible_pixels = min_visible_pixels
        self.endpoint_tolerance_m = endpoint_tolerance_m

    def describe(self) -> MetricDescription:
        visible = (f"A decision is visible when the replica covers at least min_visible_pixels "
                   f"({self.min_visible_pixels}) pixels in {self.camera_id}.")
        return MetricDescription(
            "How often the driving system detected a sign while the inserted replica was visible.",
            {
                "detected_steps": MetricField(
                    "Decisions with at least one logged sign detection. Unavailable if the driving "
                    "system logs no detections.", "decision steps"),
                "detection_rate": MetricField(
                    "Fraction of visible decisions with at least one logged sign detection. "
                    f"{visible} Unavailable without logged detections or visible decisions.",
                    "ratio (0–1)"),
                "first_detection_line_distance_m": MetricField(
                    "Signed horizontal distance from the ego to the selected stop line at the first "
                    "decision with a logged detection, positive before the line. Unavailable if no "
                    "decision has a detection or the stop line is unavailable.", "m"),
            },
        )

    def evaluate(self, group: ComparisonGroup) -> TreatmentMeasurements:
        points = group.segment.user_data.get("stop_line_points")
        return {name: self._measure(log, points) for name, log in group.runs.items()}

    def _measure(self, log: RunLog, points: object) -> dict[str, MetricScalar]:
        fields = ("detected_steps", "detection_rate", "first_detection_line_distance_m")
        if not log.steps:
            return {name: UnavailableMetricValue("The run has no decision steps") for name in fields}
        logged = [step.components or {} for step in log.steps]
        if any("detections" not in item for item in logged):
            return {name: UnavailableMetricValue("The driving system logs no detections")
                    for name in fields}
        detected = [bool(item["detections"]) for item in logged]
        visible = [
            (camera := step.object_visibility.get(REPLICA_OBJECT_ID, {}).get(self.camera_id)) is not None
            and camera.visible_pixels >= self.min_visible_pixels
            for step in log.steps
        ]
        rate: MetricScalar
        if not any(visible):
            rate = UnavailableMetricValue(f"The replica is never visible in {self.camera_id}")
        else:
            rate = sum(flag and seen for flag, seen in zip(detected, visible)) / sum(visible)
        first = next((index for index, flag in enumerate(detected) if flag), None)
        distance: MetricScalar
        if first is None:
            distance = UnavailableMetricValue("No decision has a logged detection")
        elif points is None:
            distance = UnavailableMetricValue("The attack segment has no stop_line_points in user_data")
        else:
            try:
                positions = np.stack([step.physical_pre.ego.pose[:3, 3] for step in log.steps])
                distances, _ = signed_stop_line_distances(
                    positions, np.asarray(points), endpoint_tolerance_m=self.endpoint_tolerance_m)
            except (AttributeError, TypeError, ValueError) as error:
                distance = UnavailableMetricValue(
                    f"Physical stop-line evidence is unavailable: {type(error).__name__}: {error}")
            else:
                distance = float(distances[first])
        return {"detected_steps": sum(detected), "detection_rate": rate,
                "first_detection_line_distance_m": distance}


__all__ = [
    'ReplicaVisibility',
    'SignDetection',
]
