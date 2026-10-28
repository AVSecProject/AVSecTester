"""Camera detection with the box-area forward-collision braking rule."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.data_types import Control, Observation, read_camera_rgb
from avsectester.closed_loop.hooks import CallInputs


class CameraForwardCollisionAVStack(AVStack):
    """Brake for a plausible detected vehicle whose image area exceeds ``area_brake``.

    Uses the CARLA-trained Faster R-CNN by default. ``detector_dataset="nuscenes"``
    selects the supported nuScenes model for recorded imagery.
    The image-area proxy is not a metric depth estimate. Three-dimensional detections are
    not provided, so DetectionMetrics is omitted by default. Models load on first use.
    """

    provides_detections = False
    command_modes = frozenset({"actuator"})
    supported_stages = frozenset({"perception.pre", "perception.post"})

    def __init__(self, gpu: int = 0, camera: str = "front", area_brake: float = .03,
                 throttle: float = .55, detector_dataset: str = "carla-vehicle",
                 threshold: float = .3, area_max: float = .3,
                 max_height_frac: float = .6) -> None:
        if not 0 < area_brake <= area_max <= 1:
            raise ValueError("Require 0 < area_brake <= area_max <= 1")
        if not 0 <= throttle <= 1 or not 0 <= threshold <= 1 or not 0 < max_height_frac <= 1:
            raise ValueError("throttle, threshold and maximum height must lie in [0,1]")
        self.gpu, self.camera, self.detector_dataset = gpu, camera, detector_dataset
        self.area_brake, self.throttle, self.threshold = area_brake, throttle, threshold
        self.area_max, self.max_height_frac = area_max, max_height_frac
        self._detector: Any = None
        self._last: Any = None

    def _load(self) -> None:
        import avstack.modules.perception.object2dfv  # noqa: F401 - native registry
        from avstack.config import MODELS

        self._detector = MODELS.build({
            "type": "MMDetObjectDetector2D", "model": "fasterrcnn", "dataset": self.detector_dataset,
            "gpu": self.gpu, "threshold": self.threshold,
        })

    @property
    def models(self) -> dict[str, Any]:
        if self._detector is None:
            self._load()
        return {"perception": self._detector.model}

    def detect(self, rgb: Any) -> list:
        """Return current image detections as ``(xyxy, score, vehicle)`` tuples.

        This also serves offline detector probes; it loads the configured model on demand.
        """
        from mmdet.apis import inference_detector

        model = self.models["perception"]
        instances = inference_detector(model, rgb[:, :, ::-1]).pred_instances
        boxes = instances.bboxes.detach().cpu().numpy()
        scores = instances.scores.detach().cpu().numpy()
        labels = instances.labels.detach().cpu().numpy()
        # All four CARLA-trained labels are vehicle categories. nuScenes has additional
        # categories, so keep only car/truck/trailer/bus/construction_vehicle indices.
        vehicle_labels = {0, 1, 2, 3, 4} if self.detector_dataset == "nuscenes" else None
        return [(box, float(score), "vehicle")
                for box, score, label in zip(boxes, scores, labels)
                if score >= self.threshold and (vehicle_labels is None or int(label) in vehicle_labels)]

    def reset(self, observation: Observation, past: Sequence[Observation] = ()) -> None:
        self._last = None

    def __call__(self, observation: Observation) -> Control:
        """Detect the selected RGB image and choose braking from the largest image-box area."""
        if self.camera not in observation.sensor_data:
            raise ValueError(f"Camera {self.camera!r} is absent from observation")
        captured = read_camera_rgb(observation, self.camera)
        if captured is None:
            raise ValueError(f"Camera {self.camera!r} has no RGB image")
        rgb = captured[1]
        inputs = self._emit("perception.pre", CallInputs(args=(rgb,)))
        if not isinstance(inputs, CallInputs):
            raise TypeError("perception.pre must return CallInputs")
        output = self._emit("perception.post", self.detect(*inputs.args, **inputs.kwargs))
        height, width = rgb.shape[:2]
        plausible = [(box, score, label) for box, score, label in output
                     if 0 < (box[2] - box[0]) * (box[3] - box[1]) <= self.area_max * width * height
                     and 0 < box[3] - box[1] <= self.max_height_frac * height]
        self._last = {"perception": plausible}
        close = any((box[2] - box[0]) * (box[3] - box[1]) >= self.area_brake * width * height
                    for box, _, _ in plausible)
        return Control(brake=1.) if close else Control(throttle=self.throttle)

    def component_log(self) -> dict[str, Any] | None:
        return self._last

    def close(self) -> None:
        self._detector = None
        self._last = None


__all__ = [
    'CameraForwardCollisionAVStack',
]
