"""A calibrated camera driver records detection, ranging and trajectory decisions."""

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.implemented.modules import (
    RuleBasedAVStack, RuleBasedPlanner, StopSignTracker, YOLOXDetector,
    VehicleTracker, rule_based_driving_adapter,
)
from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState
from avsectester.data_types import DeliveredAcquisition, PointCloud, TimedStateEstimate
from avsectester.geometry import CameraCalibration
from avsectester.geometry.lidars import LidarCalibration
from tests.fakes import yolox_detection_model


def test_rule_based_avstack_components(rule_based_detector):
    detector = rule_based_detector
    avstack = RuleBasedAVStack(detector, camera_id="front")
    assert avstack.detector is detector
    assert isinstance(avstack.planner, RuleBasedPlanner)
    assert isinstance(avstack.tracker, StopSignTracker)
    planner = RuleBasedPlanner(stop_before_sign_m=8.)
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
    with pytest.raises(TypeError, match="vehicle_tracker"):
        RuleBasedAVStack(detector, vehicle_tracker=object())
    with pytest.raises(ValueError, match="vehicle_labels"):
        RuleBasedAVStack(detector, vehicle_labels="car")
    with pytest.raises(ValueError, match="lidar_id"):
        RuleBasedAVStack(detector, lidar_id="")


def test_rule_based_stop_signs(rule_based_observation, rule_based_detector):
    observation = rule_based_observation()
    detector = rule_based_detector
    avstack = RuleBasedAVStack(detector, camera_id="front")
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
        assert log["vehicle_detections"] == log["vehicle_estimates"] == log["vehicle_tracks"] == []
        assert log["skipped_vehicle_detections"] == 0
        assert log["lead_track_id"] is None
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
        covariance = avstack.tracker.tracks[0].covariance.copy()
        empty = replace(observation, t=.05, frame=1, sensor_data={}, calibration={})
        assert avstack(empty).trajectory[0][2] == 50000
        np.testing.assert_array_equal(avstack.tracker.tracks[0].covariance, covariance)
        assert empty.sensor_data == {} and empty.deliveries == {}
        with pytest.raises(ValueError, match="front.*RGB image"):
            avstack(replace(empty, calibration=observation.calibration, deliveries={"front": ()}))
        avstack.reset(observation)
        with pytest.raises(ValueError, match="front.*RGB image"):
            avstack(replace(observation, sensor_data={}))
    finally:
        avstack.close()
    assert avstack.component_log() is None
    assert detector.model is None


def test_rule_based_vehicle_following(rule_based_observation):
    observation = rule_based_observation()
    car, truck, stop = [0.] * 80, [0.] * 80, [0.] * 80
    car[2], truck[7], stop[11] = 1., 1., 1.
    predictions = [[50., 40., 40., 40., .9, *car],
                   [10., 20., 4., 4., .95, *truck],
                   [80., 40., 4., 4., .3, *stop]]
    calibrated = observation.calibration["front"]
    mounting = calibrated.rig_from_camera.copy()
    mounting[2, 3] = 1.24
    lidar_id = "test_lidar"
    transform = np.array([[0., -1., 0., 2.], [1., 0., 0., -.5],
                          [0., 0., 1., 1.1], [0., 0., 0., 1.]])
    rig_points = np.column_stack((np.full(41, 10.), np.linspace(-1.4, 1.4, 41),
                                 np.full(41, 2.)))
    sensor_points = (rig_points - transform[:3, 3]) @ transform[:3, :3]
    lidar_observation = replace(observation, calibration={
        "front": CameraCalibration(calibrated.camera, mounting),
        lidar_id: LidarCalibration(transform, .1),
    }, sensor_data={**observation.sensor_data, lidar_id: PointCloud(sensor_points.astype(np.float32))})
    vehicle_tracker = VehicleTracker()
    following = RuleBasedAVStack(
        YOLOXDetector("synthetic-local-weights", device="cpu",
                      model=yolox_detection_model(predictions)),
        camera_id="front", vehicle_tracker=vehicle_tracker, lidar_id=lidar_id,
    )
    try:
        with pytest.raises(ValueError, match=lidar_id):
            following.reset(observation)
        following.reset(lidar_observation)
        with pytest.raises(ValueError, match=lidar_id):
            following(replace(lidar_observation, sensor_data=observation.sensor_data))
        with pytest.raises(ValueError, match=lidar_id):
            following(replace(lidar_observation, calibration=observation.calibration))
        following(lidar_observation)
        log = following.component_log()
        assert {row["label"] for row in log["vehicle_detections"]} == {"car", "truck"}
        assert log["skipped_vehicle_detections"] == 1
        assert len(log["vehicle_estimates"]) == 1
        assert log["vehicle_tracks"] == [] and log["lead_track_id"] is None
        assert log["vehicle_detections"][log["vehicle_estimates"][0]["detection_index"]]["label"] == "car"
        assert log["detections"] == [] and log["sign_estimates"] == []
        points = np.asarray(log["vehicle_estimates"][0]["rig_face_points_m"])
        assert points.shape == (3, 3)
        np.testing.assert_allclose(points[1], [10., 0., 0.], atol=1e-7)
        assert points[0, 1] > points[1, 1] > points[2, 1]
        predictions[0][4] = .3
        following(replace(lidar_observation, t=.1, frame=1,
                          vehicle_state=PlanarEgoState(speed=10., t=.1)))
        log = following.component_log()
        assert next(row for row in log["vehicle_detections"] if row["label"] == "car")["score"] == pytest.approx(.3)
        assert log["detections"] == [] and log["sign_estimates"] == []
        assert len(log["vehicle_tracks"]) == 1
        assert log["lead_track_id"] == log["vehicle_tracks"][0]["track_id"]
        assert log["lead_gap_m"] == pytest.approx(10. - 3.7)
        assert log["lead_speed_mps"] == 0.
        assert log["acceleration_mps2"] == -following.planner.max_deceleration_mps2
        width = np.linalg.norm(points[0, :2] - points[2, :2]) / 2
        assert log["vehicle_tracks"][0]["half_width_m"] == pytest.approx(width)
        assert log["vehicle_tracks"][0]["hits"] == 2
        np.testing.assert_allclose(log["vehicle_tracks"][0]["world_velocity_mps"], [0., 0.])
        np.testing.assert_allclose(log["lead_lateral_interval_m"], [-width, width])
        following.reset(lidar_observation)
        assert vehicle_tracker.tracks == ()
        assert following.component_log() is None
        predictions[0][4] = .9
        camera_state = PlanarEgoState(x=1., speed=10., t=.1)
        delayed = replace(lidar_observation, t=.2, vehicle_state=PlanarEgoState(x=2., speed=10., t=.2),
            deliveries={
                "front": (DeliveredAcquisition(100000,  .1, 1,
                    lidar_observation.sensor_data["front"], lidar_observation.calibration["front"], TimedStateEstimate(100000, .1, camera_state, 10.)),),
                lidar_id: (DeliveredAcquisition(0,  0., 0,
                    lidar_observation.sensor_data[lidar_id], lidar_observation.calibration[lidar_id],
                    TimedStateEstimate(0, 0., lidar_observation.vehicle_state, 10.)),),
            })
        following(delayed)
        no_lidar = replace(delayed, t=.3, frame=3,
            calibration={"front": delayed.calibration["front"]},
            sensor_data={"front": delayed.sensor_data["front"]},
            deliveries={"front": delayed.deliveries["front"]})
        following(no_lidar)
        assert no_lidar.sensor_data.keys() == {"front"}
        np.testing.assert_allclose(following.component_log()["vehicle_estimates"][0]["rig_face_points_m"][1],
                                   [9., 0., 0.], atol=1e-7)
    finally:
        following.close()
    custom = RuleBasedAVStack(
        YOLOXDetector("synthetic-local-weights", device="cpu",
                      model=yolox_detection_model(predictions)),
        camera_id="front", vehicle_tracker=VehicleTracker(), vehicle_labels=("truck",),
        lidar_id=lidar_id,
    )
    try:
        custom.reset(lidar_observation)
        custom(lidar_observation)
        log = custom.component_log()
        assert [row["label"] for row in log["vehicle_detections"]] == ["truck"]
        assert log["skipped_vehicle_detections"] == 1
        assert log["vehicle_tracks"] == [] and log["lead_track_id"] is None
    finally:
        custom.close()


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
    assert descriptor.inputs == {"images": "BCHW float tensor [0,1]"}
    assert descriptor.outputs == {"scores": "scalar confidence"}
    assert descriptor.codecs == {"scores": "tensor"}
    assert descriptor.differentiable_inputs == ("images",)
    assert descriptor.direction == "minimize" and descriptor.output_shape == ()
    assert descriptor.sensor_id == RuleBasedAVStack(detector).camera_id
    assert descriptor.decision_threshold == .7
    factory.avstack_options = MappingProxyType({"detector": detector, "camera_id": "front"})
    custom, _ = rule_based_driving_adapter(factory).scorer_bindings[descriptor.id]
    assert custom.sensor_id == "front" and custom.decision_threshold == .7


def test_delayed_sign_coordinates(rule_based_observation, rule_based_detector):
    observation = rule_based_observation()
    driver = RuleBasedAVStack(rule_based_detector, camera_id="front")
    driver.reset(observation)
    driver(observation)
    expected = driver.tracker.tracks[0].position.copy()
    captured = DeliveredAcquisition(0,  observation.t, observation.frame,
        observation.sensor_data["front"], observation.calibration["front"],
        TimedStateEstimate(0, observation.t, observation.vehicle_state, observation.ego_speed))
    delayed = replace(observation, t=.1, frame=1,
        vehicle_state=PlanarEgoState(x=5., speed=10., t=.1), deliveries={"front": (captured,)})
    driver(delayed)
    np.testing.assert_allclose(driver.tracker.tracks[0].position, expected, atol=1e-12)
    driver.close()
