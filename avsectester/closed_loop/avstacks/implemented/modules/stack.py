"""Detect signs, range by known size, track bearings and plan route-following stops."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter, ScorerDescriptor
from avsectester.data_types import Control, Observation, read_camera_rgb
from avsectester.geometry import CameraCalibration
from avsectester.geometry.poses import validated_transform
from avsectester.validation import require_positive_number

from .detector import YOLOXDetector
from .planner import StopSignPlanner
from .ranging import estimate_stop_sign_position
from .tracker import StopSignTracker


DEFAULT_CAMERA_ID = "camera_front_wide_120fov"
SIGN_LABEL = "stop sign"


class RuleBasedAVStack(AVStack):
    """Follow the route and stop for signs using calibrated camera observations.

    Detection, known-height monocular ranging and bearing-only EKF tracking feed
    a planner that cruises with IDM's free-flow term and brakes to sign-relative
    stop targets with constant required deceleration. No ground-truth sign
    positions are used. Only NuRec observations currently supply the required
    CameraCalibration. Detector, tracker and planner are supplied as components.
    """

    command_modes = frozenset({"trajectory"})
    provides_detections = False

    def __init__(
        self, detector: YOLOXDetector, tracker: StopSignTracker | None = None,
        planner: StopSignPlanner | None = None,
        camera_id: str = DEFAULT_CAMERA_ID, sign_height_m: float = .762,
    ) -> None:
        if not isinstance(detector, YOLOXDetector):
            raise TypeError("detector must be a YOLOXDetector")
        if tracker is not None and not isinstance(tracker, StopSignTracker):
            raise TypeError("tracker must be a StopSignTracker or None")
        if planner is not None and not isinstance(planner, StopSignPlanner):
            raise TypeError("planner must be a StopSignPlanner or None")
        if not isinstance(camera_id, str) or not camera_id:
            raise ValueError("camera_id must be a nonempty string")
        require_positive_number(sign_height_m, "sign_height_m")
        self.detector = detector
        self.tracker = StopSignTracker() if tracker is None else tracker
        self.planner = StopSignPlanner() if planner is None else planner
        self.camera_id = camera_id
        self.sign_height_m = sign_height_m
        self._closed = False
        self._last: dict[str, Any] | None = None

    def _calibration(self, observation: Observation) -> CameraCalibration:
        calibration = observation.calibration.get(self.camera_id)
        if not isinstance(calibration, CameraCalibration):
            raise ValueError(
                f"RuleBasedAVStack requires CameraCalibration for camera {self.camera_id!r}; "
                "currently only NuRec observations provide this calibration."
            )
        return calibration

    def reset(self, observation: Observation, past: Sequence[Observation] = ()) -> None:
        """Clear tracks and the stop state machine for a fresh run."""
        validated_transform(self._calibration(observation).rig_from_camera)
        self.tracker.reset()
        self.planner.reset(observation)
        self._last = None

    def __call__(self, observation: Observation) -> Control:
        """Detect, range and track stop signs, then return the planner's rig-frame trajectory."""
        if self._closed:
            raise RuntimeError("RuleBasedAVStack has been closed")
        calibration = self._calibration(observation)
        rig_from_camera = validated_transform(calibration.rig_from_camera)
        captured = read_camera_rgb(observation, self.camera_id)
        if captured is None:
            raise ValueError(f"Camera {self.camera_id!r} has no RGB image")
        detections = [(box, score) for box, score, label in self.detector.detect(captured[1])
                      if label == SIGN_LABEL]
        estimates = [
            estimate_stop_sign_position(box, calibration, self.sign_height_m)
            for box, _score in detections
        ]
        self.tracker.update(observation.vehicle_state, rig_from_camera[:3, 3], estimates)
        tracks = self.tracker.tracks
        command = self.planner.step(observation, tracks)
        self._last = {
            "detections": [{"box_xyxy": np.asarray(box).tolist(), "score": float(score)}
                           for box, score in detections],
            "sign_estimates": [
                {"rig_position_m": position.tolist(), "range_m": float(np.linalg.norm(position))}
                for position in estimates
            ],
            "tracks": [
                {"track_id": track.track_id, "world_position_m": track.position.tolist(),
                 "max_position_sigma_m": float(np.sqrt(max(0., np.linalg.eigvalsh(track.covariance)[-1])))}
                for track in tracks
            ],
            **self.planner.component_log,
        }
        return command

    def component_log(self) -> Mapping[str, Any] | None:
        """Return detections, ranges, world tracks and current planner outputs."""
        return self._last

    def close(self) -> None:
        """Release the detection model and clear recorded decision outputs."""
        self._closed = True
        self.detector.close()
        self._last = None


def rule_based_driving_adapter(factory: Callable[[int], AVStack]) -> DrivingAdapter:
    """Bind the configured rule driver's differentiable detection-confidence scorer.

    The factory must declare ``avstack_options`` with its constructed detector. Reading
    these options describes the scorer without constructing a driver or loading weights.
    """
    options = getattr(factory, "avstack_options", None)
    if not isinstance(options, Mapping):
        raise ValueError("rule_based_driving_adapter requires factory.avstack_options")
    detector = options.get("detector")
    if not isinstance(detector, YOLOXDetector):
        raise ValueError("factory.avstack_options must include a YOLOXDetector detector")
    descriptor = ScorerDescriptor(
        "rule_based.detection_confidence", {"image": "CHW float tensor [0,1]"},
        {"score": "scalar confidence"}, {"score": "tensor"}, ("image",),
        direction="minimize", sensor_id=options.get("camera_id", DEFAULT_CAMERA_ID),
        decision_threshold=detector.score_threshold,
    )

    def score(avstack: RuleBasedAVStack, inputs: Any, target: Any) -> Any:
        return avstack.detector.target_score(inputs, target)

    return DrivingAdapter(factory, scorers=((descriptor, score),))


__all__ = [
    'RuleBasedAVStack',
    'rule_based_driving_adapter',
]
