"""The camera policy's area filter, vehicle labels and braking rule."""
import numpy as np
import pytest

from avsectester.closed_loop.avstacks.implemented.camera_forward_collision import CameraForwardCollisionAVStack
from avsectester.data_types import Observation


def test_camera_braking(camera_detector_boundary):
    avstack = CameraForwardCollisionAVStack(detector_dataset="nuscenes")
    observation = Observation(0., 0, {"front": np.zeros((100, 100, 3), np.uint8)})
    avstack.reset(observation)
    for boxes, labels, brake in [([], [], 0.), ([[20., 20., 60., 50.]], [0], 1.),
                                  ([[0., 0., 100., 100.]], [0], 0.),
                                  ([[20., 20., 60., 50.]], [6], 0.)]:
        camera_detector_boundary.boxes, camera_detector_boundary.labels = boxes, labels
        command = avstack(observation)
        assert command.brake == brake
        assert command.throttle == (0. if brake else .55)
        assert bool(avstack.component_log()["perception"]) is bool(brake)
    with pytest.raises(ValueError, match="absent"):
        avstack(Observation(0., 0, {}))
    with pytest.raises(ValueError, match="area_brake"):
        CameraForwardCollisionAVStack(area_brake=2.)
