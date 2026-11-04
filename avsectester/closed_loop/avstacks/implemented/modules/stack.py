"""Detect and track stop signs and optional lead vehicles for route-following control."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter, ScorerDescriptor
from avsectester.data_types import Control, Observation, PointCloud, read_camera_rgb
from avsectester.geometry import CameraCalibration
from avsectester.geometry.lidars import LidarCalibration
from avsectester.geometry.poses import validated_transform
from avsectester.validation import require_positive_number

from .detector import YOLOXDetector
from .planner import RuleBasedPlanner
from .ranging import estimate_stop_sign_position, estimate_vehicle_face_points
from .tracker import StopSignTracker, VehicleTrack, VehicleTracker


DEFAULT_CAMERA_ID = "camera_front_wide_120fov"
DEFAULT_LIDAR_ID = "lidar_gt_top_p128"
SIGN_LABEL = "stop sign"


class RuleBasedAVStack(AVStack):
    """Follow the route, stop for signs and optionally follow camera-detected vehicles.

    Detection, known-height monocular ranging and bearing-only EKF tracking feed
    a planner that cruises with IDM's free-flow term and brakes to sign-relative
    stop targets with constant required deceleration. No ground-truth sign
    positions are used. Only NuRec observations currently supply the required
    CameraCalibration. Detector, tracker and planner are supplied as components.

    Supplying ``vehicle_tracker`` enables camera detection with LiDAR vehicle ranging.
    The selected LiDAR must supply a PointCloud and LidarCalibration. Visible-face measurements
    feed the tracker's two-stage score association and constant-velocity Kalman filter.
    Without a vehicle tracker, vehicle detections are ignored and LiDAR is not read.
    """

    command_modes = frozenset({"trajectory"})
    provides_detections = False

    def __init__(
        self, detector: YOLOXDetector, tracker: StopSignTracker | None = None,
        planner: RuleBasedPlanner | None = None,
        camera_id: str = DEFAULT_CAMERA_ID, sign_height_m: float = .762,
        *, vehicle_tracker: VehicleTracker | None = None,
        vehicle_labels: Sequence[str] = ("car", "truck", "bus"),
        lidar_id: str = DEFAULT_LIDAR_ID,
    ) -> None:
        if not isinstance(detector, YOLOXDetector):
            raise TypeError("detector must be a YOLOXDetector")
        if tracker is not None and not isinstance(tracker, StopSignTracker):
            raise TypeError("tracker must be a StopSignTracker or None")
        if planner is not None and not isinstance(planner, RuleBasedPlanner):
            raise TypeError("planner must be a RuleBasedPlanner or None")
        if vehicle_tracker is not None and not isinstance(vehicle_tracker, VehicleTracker):
            raise TypeError("vehicle_tracker must be a VehicleTracker or None")
        if (isinstance(vehicle_labels, (str, bytes)) or not isinstance(vehicle_labels, Sequence)
                or any(not isinstance(label, str) or not label.strip() for label in vehicle_labels)):
            raise ValueError("vehicle_labels must be a sequence of nonempty class names")
        if not isinstance(camera_id, str) or not camera_id:
            raise ValueError("camera_id must be a nonempty string")
        if not isinstance(lidar_id, str) or not lidar_id:
            raise ValueError("lidar_id must be a nonempty string")
        require_positive_number(sign_height_m, "sign_height_m")
        self.detector = detector
        self.tracker = StopSignTracker() if tracker is None else tracker
        self.planner = RuleBasedPlanner() if planner is None else planner
        self.camera_id = camera_id
        self.lidar_id = lidar_id
        self.sign_height_m = sign_height_m
        self.vehicle_tracker = vehicle_tracker
        self.vehicle_labels = tuple(vehicle_labels)
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

    def _lidar_calibration(self, observation: Observation) -> LidarCalibration:
        calibration = observation.calibration.get(self.lidar_id)
        if not isinstance(calibration, LidarCalibration):
            raise ValueError(f"LiDAR {self.lidar_id!r} requires LidarCalibration")
        return calibration

    def reset(self, observation: Observation, past: Sequence[Observation] = ()) -> None:
        """Clear sign and vehicle tracks and reset planning for a fresh run."""
        validated_transform(self._calibration(observation).rig_from_camera)
        if self.vehicle_tracker is not None:
            validated_transform(self._lidar_calibration(observation).rig_from_lidar)
            self.vehicle_tracker.reset()
        self.tracker.reset()
        self.planner.reset(observation)
        self._last = None

    def __call__(self, observation: Observation) -> Control:
        """Range and track selected detections, then return a rig-frame driving trajectory."""
        if self._closed:
            raise RuntimeError("RuleBasedAVStack has been closed")
        calibration = self._calibration(observation)
        rig_from_camera = validated_transform(calibration.rig_from_camera)
        captured = read_camera_rgb(observation, self.camera_id)
        if captured is None:
            raise ValueError(f"Camera {self.camera_id!r} has no RGB image")
        rig_points = None
        if self.vehicle_tracker is None:
            detected = self.detector.detect(captured[1])
        else:
            lidar = self._lidar_calibration(observation)
            transform = validated_transform(lidar.rig_from_lidar)
            cloud = observation.sensor_data.get(self.lidar_id)
            if not isinstance(cloud, PointCloud):
                raise ValueError(f"LiDAR {self.lidar_id!r} requires a PointCloud")
            points = np.asarray(cloud.points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError(f"LiDAR {self.lidar_id!r} points must be finite with shape (N, 3)")
            rig_points = points @ transform[:3, :3].T + transform[:3, 3]
            detected = self.detector.detect(
                captured[1], score_threshold=min(self.detector.score_threshold,
                                                 self.vehicle_tracker.low_score_threshold))
        detections = [(box, score) for box, score, label in detected
                      if label == SIGN_LABEL and score >= self.detector.score_threshold]
        estimates = [
            estimate_stop_sign_position(box, calibration, self.sign_height_m)
            for box, _score in detections
        ]
        self.tracker.update(observation.vehicle_state, rig_from_camera[:3, 3], estimates)
        tracks = self.tracker.tracks
        vehicle_detections: list[tuple[np.ndarray, float, str]] = []
        vehicle_estimates: list[dict[str, Any]] = []
        vehicle_tracks: tuple[VehicleTrack, ...] = ()
        skipped_vehicles = 0
        if self.vehicle_tracker is not None:
            vehicle_detections = [(box, score, label) for box, score, label in detected
                                  if label in self.vehicle_labels
                                  and score >= self.vehicle_tracker.low_score_threshold]
            face_points: list[np.ndarray] = []
            scores: list[float] = []
            assert rig_points is not None
            for index, (box, score, _label) in enumerate(vehicle_detections):
                try:
                    points = estimate_vehicle_face_points(box, calibration, rig_points)
                except ValueError:
                    skipped_vehicles += 1
                    continue
                face_points.append(points)
                scores.append(score)
                vehicle_estimates.append({"detection_index": index,
                                          "rig_face_points_m": points.tolist()})
            self.vehicle_tracker.update(observation.vehicle_state, observation.t, face_points, scores)
            vehicle_tracks = self.vehicle_tracker.tracks
        command = self.planner.step(observation, tracks, vehicle_tracks)
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
            "vehicle_detections": [
                {"box_xyxy": np.asarray(box).tolist(), "score": float(score), "label": label}
                for box, score, label in vehicle_detections
            ],
            "vehicle_estimates": vehicle_estimates,
            "vehicle_tracks": [
                {"track_id": track.track_id, "world_position_m": track.position.tolist(),
                 "world_velocity_mps": track.velocity.tolist(),
                 "half_width_m": track.half_width_m, "hits": track.hits}
                for track in vehicle_tracks
            ],
            "skipped_vehicle_detections": skipped_vehicles,
            **self.planner.component_log,
        }
        return command

    def component_log(self) -> Mapping[str, Any] | None:
        """Return sign and vehicle detections, range estimates, tracks and planner outputs."""
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
        "rule_based.detection_confidence", {"images": "BCHW float tensor [0,1]"},
        {"scores": "scalar confidence"}, {"scores": "tensor"}, ("images",),
        direction="minimize", sensor_id=options.get("camera_id", DEFAULT_CAMERA_ID),
        decision_threshold=detector.score_threshold,
    )

    def score(avstack: RuleBasedAVStack, images: Any, targets: Any) -> Any:
        return avstack.detector.target_scores(images, targets)

    return DrivingAdapter(factory, scorers=((descriptor, score),))


__all__ = [
    'RuleBasedAVStack',
    'rule_based_driving_adapter',
]
