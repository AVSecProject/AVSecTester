"""A calibrated camera driver records detection, ranging and trajectory decisions."""

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.implemented.modules import (
    RuleBasedAVStack, StopSignPlanner, StopSignTracker, YOLOXDetector,
    rule_based_driving_adapter,
)
from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState


def test_rule_based_avstack(rule_based_observation, rule_based_detector):
    observation = rule_based_observation()
    detector = rule_based_detector
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
    try:
        avstack.reset(observation)
        assert avstack.component_log() is None
        command = avstack(observation)
        assert command.trajectory[0] == ((0., 0., 0.), (1., 0., 0., 0.), 0)
        log = avstack.component_log()
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


def test_rule_based_driving_adapter(tmp_path):
    # A missing checkpoint proves binding discovery does not construct the driver or load weights.
    detector = YOLOXDetector(tmp_path / "not-loaded.pth", device="cpu", score_threshold=.7)

    def factory(seed):
        raise RuntimeError("Binding discovery must not construct an AVStack")

    with pytest.raises(ValueError, match="factory.avstack_options"):
        rule_based_driving_adapter(factory)
    factory.avstack_options = MappingProxyType({})
    with pytest.raises(ValueError, match="detector"):
        rule_based_driving_adapter(factory)
    factory.avstack_options = MappingProxyType({"detector": detector})
    adapter = rule_based_driving_adapter(factory)
    descriptor, score = adapter.scorer_bindings["rule_based.detection_confidence"]
    assert descriptor.inputs == {"image": "CHW float tensor [0,1]"}
    assert descriptor.outputs == {"score": "scalar confidence"}
    assert descriptor.codecs == {"score": "tensor"}
    assert descriptor.differentiable_inputs == ("image",)
    assert descriptor.direction == "minimize" and descriptor.output_shape == ()
    assert descriptor.sensor_id == RuleBasedAVStack(detector).camera_id
    assert descriptor.decision_threshold == .7
    factory.avstack_options = MappingProxyType({"detector": detector, "camera_id": "front"})
    custom, _ = rule_based_driving_adapter(factory).scorer_bindings[descriptor.id]
    assert custom.sensor_id == "front" and custom.decision_threshold == .7
