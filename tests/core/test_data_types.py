"""Canonical detections and camera transport use small, explicit numeric payloads."""
import numpy as np
import pytest
from types import SimpleNamespace
from avsectester.data_types import (
    ActorState, Control, Detection, Observation, read_camera_rgb, validate_command_modes,
    write_camera_rgb, write_sensor_data,
)
from avsectester.geometry.poses import make_pose, rotation_matrix
from tests.fakes import CarlaImageData


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
