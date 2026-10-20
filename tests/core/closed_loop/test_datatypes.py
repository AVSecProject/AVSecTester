"""Canonical detections and camera transport use small, explicit numeric payloads."""
import numpy as np
import pytest
from avsectester.closed_loop.datatypes import Detection, Observation, read_camera_rgb, write_camera_rgb


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
    changed = write_camera_rgb(observation, "front", np.full_like(original, 90))
    assert np.all(read_camera_rgb(changed, "front")[1] == 90)
    assert np.all(read_camera_rgb(observation, "front")[1] == 20)
    np.testing.assert_array_equal(changed.sensor_data["lidar"], observation.sensor_data["lidar"])
