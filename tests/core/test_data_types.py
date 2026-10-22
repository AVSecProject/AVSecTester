"""Canonical detections and camera transport use small, explicit numeric payloads."""
import numpy as np
import pytest
from operator import attrgetter
from types import SimpleNamespace
from avsectester.data_types import (
    ActorState, Detection, Observation, read_camera_rgb, write_camera_rgb, write_sensor_data,
)
from avsectester.geometry.poses import make_pose, rotation_matrix


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


def test_write_camera_rgb():
    original = np.full((2, 3, 3), 20, np.uint8)
    observation = Observation(0, 0, {"front": original, "lidar": np.zeros((2, 4))})
    changed = write_camera_rgb(observation, "front", np.full(original.shape, 90.5))
    assert np.all(read_camera_rgb(changed, "front")[1] == 90)
    assert changed.sensor_data["front"].dtype == np.uint8
    assert np.all(read_camera_rgb(observation, "front")[1] == 20)
    np.testing.assert_array_equal(changed.sensor_data["lidar"], observation.sensor_data["lidar"])


@pytest.mark.parametrize("sensor,payload,data,array,expected,metadata", [
    pytest.param("lidar", np.zeros((2, 3, 4), np.float32), np.full((2, 3, 4), 80.5, np.float32),
                 lambda value: value, np.full((2, 3, 4), 80.5, np.float32), (), id="range-image"),
    pytest.param("lidar", np.zeros((2, 4), np.float32), np.full((2, 4), 4.5, np.float32),
                 lambda value: value, np.full((2, 4), 4.5, np.float32), (), id="point-array"),
    pytest.param("gnss", SimpleNamespace(data=np.zeros(3), timestamp=.3), np.full(3, 1.25),
                 attrgetter("data"), np.full(3, 1.25), ("timestamp",), id="native-gnss"),
    pytest.param("imu", SimpleNamespace(data={"dv": np.zeros(3)}, timestamp=.3),
                 {"dv": np.full(3, 2.5)}, lambda value: value.data["dv"],
                 np.full(3, 2.5), ("timestamp",), id="native-imu"),
])
def test_write_sensor_data(sensor, payload, data, array, expected, metadata):
    original = array(payload).copy()
    observation = Observation(.3, 3, {sensor: payload})
    changed = write_sensor_data(observation, sensor, data)
    assert changed is not observation
    np.testing.assert_array_equal(array(changed.sensor_data[sensor]), expected)
    assert array(changed.sensor_data[sensor]).dtype == expected.dtype
    np.testing.assert_array_equal(array(observation.sensor_data[sensor]), original)
    for name in metadata:
        assert getattr(changed.sensor_data[sensor], name) == getattr(payload, name)
    with pytest.raises(KeyError):
        write_sensor_data(observation, "missing", data)
