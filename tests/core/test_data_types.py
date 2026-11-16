"""Canonical detections and camera transport use small, explicit numeric payloads."""
import numpy as np
import pytest
from dataclasses import replace
from types import SimpleNamespace
from avsectester.data_types import (
    ActorState, Control, Detection, Observation, read_camera_rgb, validate_command_modes,
    DeliveredAcquisition, TimedStateEstimate, interpolate_state_estimate, without_sensor_payloads,
    write_camera_rgb, write_sensor_data,
)
from avsectester.geometry.poses import make_pose, quaternion_to_rotation, rotation_matrix
from tests.fakes import CarlaImageData


def test_delivered():
    pixels = np.zeros((2, 3, 3), np.uint8)
    state = TimedStateEstimate(100000, .1, "captured", 1.)
    captured = DeliveredAcquisition(100000,  .1, 1, pixels, "lens", state, 4)
    observation = Observation(.1, 1, {"front": pixels}, {"front": "lens"},
        vehicle_state="captured", ego_speed=1., deliveries={"front": (captured,)},
        state_estimates=(state,))
    assert observation.delivered("front")[-1] is captured
    assert observation.delivered("front")[-1].state is observation.estimated_states()[-1]
    edited = write_camera_rgb(observation, "front", np.full_like(pixels, 50))
    edited.set_calibration("front", "edited lens")
    edited.vehicle_state, edited.ego_speed = "current", 2.
    latest = edited.delivered("front")[-1]
    assert latest.state.vehicle_state == "current" and latest.state.ego_speed == 2.
    assert latest.state.vehicle_state is edited.vehicle_state
    assert latest.calibration == "edited lens" and np.all(latest.data == 50)
    assert edited.deliveries["front"][-1] is captured
    later = TimedStateEstimate(200000, .2, "later", 3.)
    delayed = replace(edited, t=.2, vehicle_state="later", ego_speed=3.,
                      state_estimates=(later,))
    assert delayed.delivered("front")[-1].state is state
    stripped = replace(edited, sensor_data={})
    assert stripped.delivered("front")[-1].data is None
    assert stripped.delivered("front")[-1].state.vehicle_state is stripped.vehicle_state
    direct = Observation(.1, 1, {"front": pixels}, vehicle_state="direct")
    assert direct.delivered("front")[-1].capture_id is None
    assert direct.delivered("front")[-1].state.vehicle_state == "direct"
    with pytest.raises(KeyError):
        stripped.delivered("unknown")


def test_actor_state():
    pose = make_pose(rotation_matrix((0., 0., 90.)), [10., 5., 1.])
    actor = ActorState(pose, extent=(4., 2., 1.5), box_center=(2., 0., .75),
                       box_rotation=(0., 0., 90.))
    np.testing.assert_allclose(actor.box_transform()[:3, 3], [10., 7., 1.75])
    np.testing.assert_allclose(actor.box_transform()[:3, :3],
                               rotation_matrix((0., 0., 180.)), atol=1e-12)
    with pytest.raises(ValueError, match="extent"):
        ActorState(np.eye(4), extent=(-1., 2., 1.))


def test_detection():
    detection = Detection((1, 2, 3), (4, 2, 1), np.float32(.2), "vehicle", np.float64(.8))
    np.testing.assert_allclose(detection.position, [1., 2., 3.])
    assert detection.extent == (4., 2., 1.) and detection.yaw == pytest.approx(.2)
    assert detection.score == pytest.approx(.8)
    with pytest.raises(ValueError, match="extent"):
        Detection((0, 0, 0), (-1, 1, 1), 0)
    with pytest.raises(TypeError, match="category"):
        Detection((0, 0, 0), (1, 1, 1), 0, 42)


def test_read_camera_rgb():
    front = np.full((2, 3, 4), [20, 30, 40, 255], np.uint8)
    observation = Observation(0, 0, {"front": front, "rear": np.full((2, 3, 3), 70, np.uint8)})
    sensor, pixels = read_camera_rgb(observation)
    assert sensor == "front"
    np.testing.assert_array_equal(pixels, front[..., :3])
    assert read_camera_rgb(observation, "missing") is None
    assert read_camera_rgb(Observation(0, 0, {"lidar": np.zeros((3, 4))})) is None
    assert np.all(read_camera_rgb(observation, "rear")[1] == 70)
    native = CarlaImageData(data=np.full((2, 3, 3), 45, np.uint8), calibration="native lens")
    observation = Observation(0, 0, {"front": native})
    np.testing.assert_array_equal(read_camera_rgb(observation, "front")[1], 45)


def test_write_camera_rgb():
    original = np.full((2, 3, 3), 20, np.uint8)
    observation = Observation(0, 0, {"front": original, "lidar": np.zeros((2, 4))})
    changed = write_camera_rgb(observation, "front", np.full(original.shape, 90.5))
    assert np.all(read_camera_rgb(changed, "front")[1] == 90)
    assert changed.sensor_data["front"].dtype == np.uint8
    assert np.all(read_camera_rgb(observation, "front")[1] == 20)
    np.testing.assert_array_equal(changed.sensor_data["lidar"], observation.sensor_data["lidar"])
    native = CarlaImageData(data=np.full((2, 3, 3), 20, np.uint8), calibration="native lens")
    native_observation = Observation(0, 0, {"front": native})
    changed = write_camera_rgb(native_observation, "front", np.full((2, 3, 3), 80, np.uint8))
    assert changed.sensor_data["front"] is not native
    assert changed.sensor_data["front"].calibration == "native lens"
    np.testing.assert_array_equal(read_camera_rgb(changed, "front")[1], 80)
    np.testing.assert_array_equal(read_camera_rgb(native_observation, "front")[1], 20)


def test_write_sensor_data():
    original = np.zeros((2, 3, 4), np.float32)
    native = SimpleNamespace(data={"dv": np.zeros(3)}, timestamp=.3)
    observation = Observation(.3, 3, {"lidar": original, "imu": native})
    changed = write_sensor_data(observation, "lidar", np.full_like(original, 80.5))
    assert changed is not observation
    np.testing.assert_array_equal(changed.sensor_data["lidar"], 80.5)
    assert changed.sensor_data["lidar"].dtype == np.float32
    np.testing.assert_array_equal(observation.sensor_data["lidar"], 0.)
    changed = write_sensor_data(observation, "imu", {"dv": np.full(3, 2.5)})
    np.testing.assert_array_equal(changed.sensor_data["imu"].data["dv"], 2.5)
    np.testing.assert_array_equal(observation.sensor_data["imu"].data["dv"], 0.)
    assert changed.sensor_data["imu"].timestamp == native.timestamp
    with pytest.raises(KeyError):
        write_sensor_data(observation, "missing", np.zeros(3))


def test_set_calibration():
    native = CarlaImageData(data=np.zeros((2, 3, 3), np.uint8), calibration="original")
    observation = Observation(0., 0, {"front": native, "rear": np.zeros((2, 3, 3), np.uint8)})
    observation.set_calibration("front", "replacement")
    assert observation.calibration["front"] == native.calibration == "replacement"
    observation.set_calibration("rear", "array lens")
    assert observation.calibration["rear"] == "array lens"
    with pytest.raises(KeyError, match="Unknown observation sensor"):
        observation.set_calibration("missing", "lens")


def test_validate_command_modes():
    assert validate_command_modes(Control(throttle=.5), frozenset({"actuator"})) is None
    trajectory = [((0., 0., 0.), (1., 0., 0., 0.), 0)]
    assert validate_command_modes(Control(trajectory=trajectory), frozenset({"trajectory"})) is None
    assert validate_command_modes(Control(throttle=.5), None) is None
    with pytest.raises(ValueError, match="trajectory commands"):
        validate_command_modes(Control(trajectory=trajectory), frozenset({"actuator"}))
    with pytest.raises(ValueError, match="actuator command fields"):
        validate_command_modes(Control(brake=.5), frozenset({"trajectory"}))
    with pytest.raises(TypeError, match="Expected Control"):
        validate_command_modes(object(), None)


def test_without_sensor_payloads():
    pixels = np.zeros((2, 3, 3), np.uint8)
    state = TimedStateEstimate(0, 0., "state", 2.)
    sample = DeliveredAcquisition(0,  0., 0, pixels, "lens", state, 3, frame_start_us=-17000)
    observation = Observation(0., 0, {"front": pixels, "rear": pixels},
        {"front": "lens"}, deliveries={"front": (sample,)}, state_estimates=(state,))
    selected = without_sensor_payloads(observation, keep_sensor_ids=("front",))
    assert list(selected.sensor_data) == ["front"]
    assert selected.sensor_data["front"] is pixels
    assert selected.deliveries["front"][-1].data is None
    assert selected.deliveries["front"][-1].frame_start_us == -17000
    assert selected.deliveries["front"][-1].time_us == 0
    assert selected.calibration is observation.calibration
    assert selected.state_estimates is observation.state_estimates
    assert without_sensor_payloads(observation).sensor_data == {}
    assert observation.deliveries["front"][-1].data is pixels


def test_interpolate_state_estimate():
    from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState

    unknown = interpolate_state_estimate((TimedStateEstimate(0, 0., None, 1.),
        TimedStateEstimate(100000, .1, None, 3.)), 50000)
    assert unknown.vehicle_state is None and unknown.ego_speed == 2.
    first = TimedStateEstimate(0, 0., PlanarEgoState(x=0., yaw=np.deg2rad(170.)), 2.)
    last = TimedStateEstimate(100000, .1, PlanarEgoState(x=10., yaw=np.deg2rad(-170.)), 4.)
    assert interpolate_state_estimate((first, last), 0) is first
    middle = interpolate_state_estimate((first, last), 50000)
    assert middle.t == .05 and middle.vehicle_state.t == .05
    assert middle.vehicle_state.x == 5. and middle.ego_speed == 3.
    assert middle.vehicle_state.yaw == pytest.approx(np.pi)
    left = ActorState(np.eye(4), velocity=(0., 0., 0.))
    pose = make_pose(rotation_matrix((0., 0., 90.)), (10., 0., 0.))
    right = ActorState(pose, velocity=(2., 0., 0.))
    actor = interpolate_state_estimate((TimedStateEstimate(0, 0., left, 0.),
        TimedStateEstimate(100000, .1, right, 2.)), 50000)
    np.testing.assert_allclose(actor.vehicle_state.pose[:3, 3], [5., 0., 0.])
    np.testing.assert_allclose(actor.vehicle_state.pose[:3, :3], rotation_matrix((0., 0., 45.)))
    assert actor.vehicle_state.velocity == (1., 0., 0.)
    native_left = SimpleNamespace(position=SimpleNamespace(x=np.zeros(3)),
        attitude=SimpleNamespace(q=np.array([1., 0., 0., 0.])), t=0.)
    native_right = SimpleNamespace(position=SimpleNamespace(x=np.array([2., 0., 0.])),
        attitude=SimpleNamespace(q=np.array([0., 0., 0., 1.])), t=.1)
    native = interpolate_state_estimate((TimedStateEstimate(0, 0., native_left, 0.),
        TimedStateEstimate(100000, .1, native_right, 2.)), 50000)
    np.testing.assert_allclose(native.vehicle_state.position.x, [1., 0., 0.])
    np.testing.assert_allclose(quaternion_to_rotation(*native.vehicle_state.attitude.q),
                               rotation_matrix((0., 0., 90.)), atol=1e-15)
    assert native.vehicle_state.t == .05
    with pytest.raises(ValueError, match="outside"):
        interpolate_state_estimate((first, last), 100001)
