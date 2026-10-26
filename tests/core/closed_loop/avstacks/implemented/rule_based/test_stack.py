"""A calibrated camera driver records detection, ranging and trajectory decisions."""

from dataclasses import replace
import weakref

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.implemented.rule_based import (
    CocoStopSignDetector, RuleBasedAVStack, StopSignPlanner, StopSignTracker,
)
from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState
from avsectester.geometry import CameraCalibration, PinholeCamera
from avsectester.geometry.cameras import BODY_FROM_OPTICAL
from avsectester.data_types import Observation
from tests.fakes import CocoDetectionModel


def test_rule_based_avstack():
    camera = PinholeCamera(np.array([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]]), 100, 80)
    observation = Observation(
        0., 0, {"front": np.full((80, 100, 3), 30, np.uint8)},
        {"front": CameraCalibration(camera, BODY_FROM_OPTICAL)},
        PlanarEgoState(speed=10.), 10.,
        np.column_stack((np.arange(101.), np.zeros(101), np.zeros(101))),
    )
    detector = CocoStopSignDetector(
        "unused-local-weights.pth", device="cpu", model=CocoDetectionModel(boxes=((48., 38., 52., 42.),)),
    )
    avstack = RuleBasedAVStack(detector, camera_id="front")
    assert avstack.detector is detector
    assert isinstance(avstack.planner, StopSignPlanner)
    assert isinstance(avstack.tracker, StopSignTracker)
    planner = StopSignPlanner(stop_before_sign_m=8.)
    tracker = StopSignTracker()
    composed = RuleBasedAVStack(detector, tracker, planner)
    assert composed.tracker is tracker
    assert composed.planner is planner
    with pytest.raises(TypeError, match="detector"):
        RuleBasedAVStack(object())
    with pytest.raises(TypeError, match="tracker"):
        RuleBasedAVStack(detector, object())
    with pytest.raises(TypeError, match="planner"):
        RuleBasedAVStack(detector, planner=object())
    model_reference = weakref.ref(detector.model)
    try:
        avstack.reset(observation)
        assert avstack.component_log() is None
        command = avstack(observation)
        assert avstack.command_modes == frozenset({"trajectory"})
        assert avstack.detections() is None
        assert command.trajectory[0] == ((0., 0., 0.), (1., 0., 0., 0.), 0)
        log = avstack.component_log()
        assert set(log) == {
            "detections", "sign_estimates", "state", "target_track_id", "distance_to_stop_m",
            "acceleration_mps2", "planned_speed_mps", "braking", "tracks",
        }
        assert log["detections"][0]["score"] == pytest.approx(.9)
        assert len(log["detections"][0]["box_xyxy"]) == 4
        assert len(log["sign_estimates"][0]["rig_position_m"]) == 3
        assert log["sign_estimates"][0]["range_m"] == pytest.approx(19.05, rel=.002)
        assert len(log["tracks"]) == 1
        track = log["tracks"][0]
        assert track["track_id"] == 0
        np.testing.assert_allclose(track["world_position_m"], [19.05, 0.], rtol=.002)
        assert track["max_position_sigma_m"] > 0.
        assert log["state"] == "STOPPING" and log["target_track_id"] == 0
        assert np.isfinite(log["distance_to_stop_m"])
        assert log["acceleration_mps2"] < 0.
        assert log["planned_speed_mps"] <= observation.ego_speed
        next_observation = replace(
            observation, t=.1, frame=1,
            vehicle_state=PlanarEgoState(x=1., speed=10., t=.1),
        )
        assert avstack(next_observation).trajectory[0][2] == 100_000
        assert len(avstack.tracker.tracks) == 1
        avstack.reset(observation)
        assert avstack.tracker.tracks == ()
        assert avstack.component_log() is None
        assert avstack(observation).trajectory == command.trajectory
        with pytest.raises(ValueError, match="CameraCalibration.*front.*NuRec"):
            avstack.reset(replace(observation, calibration={}))
        with pytest.raises(ValueError, match="front.*RGB image"):
            avstack(replace(observation, sensor_data={}))
    finally:
        avstack.close()
    assert avstack.component_log() is None
    assert detector.model is None
    assert model_reference() is None
