"""Detect signs, range by known size, track bearings and plan route-following stops."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.data_types import Control, Observation, read_camera_rgb
from avsectester.geometry import CameraCalibration
from avsectester.geometry.poses import validated_transform
from avsectester.validation import require_positive_number

from .detector import CocoStopSignDetector
from .planner import StopSignPlanner
from .ranging import estimate_stop_sign_position
from .tracker import StopSignTracker


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
        self, detector: CocoStopSignDetector, tracker: StopSignTracker | None = None,
        planner: StopSignPlanner | None = None,
        camera_id: str = "camera_front_wide_120fov", sign_height_m: float = .762,
    ) -> None:
        if not isinstance(detector, CocoStopSignDetector):
            raise TypeError("detector must be a CocoStopSignDetector")
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
        if self._closed:
            raise RuntimeError("RuleBasedAVStack has been closed")
        calibration = self._calibration(observation)
        rig_from_camera = validated_transform(calibration.rig_from_camera)
        captured = read_camera_rgb(observation, self.camera_id)
        if captured is None:
            raise ValueError(f"Camera {self.camera_id!r} has no RGB image")
        detections = self.detector.detect(captured[1])
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
