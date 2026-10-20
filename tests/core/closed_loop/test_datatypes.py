"""Canonical detection metadata has normalized scalar values and explicit category types."""
import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Detection, Observation


def test_detection_normalizes_numpy_yaw_and_score_to_python_floats():
    detection = Detection((1, 2, 3), (4, 2, 1), np.float32(.2), 'vehicle', np.float64(.8))
    assert type(detection.yaw) is float and type(detection.score) is float
    assert detection.position == (1., 2., 3.) and detection.extent == (4., 2., 1.)


@pytest.mark.parametrize('category', [1, 2.5, [], {}])
def test_detection_requires_a_string_or_absent_category(category):
    with pytest.raises(TypeError, match='category'):
        Detection((1, 2, 3), (4, 2, 1), 0, category)


def test_read_camera_rgb_does_not_substitute_a_different_camera():
    from avsectester.closed_loop.datatypes import read_camera_rgb

    observation = Observation(0, 0, {"front": np.zeros((2, 2, 3), np.uint8)})
    assert read_camera_rgb(observation, "rear") is None
    assert read_camera_rgb(observation, "front")[0] == "front"
    assert read_camera_rgb(observation)[0] == "front"
