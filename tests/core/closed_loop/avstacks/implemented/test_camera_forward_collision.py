"""The camera policy's area filter, vehicle labels and braking rule."""
import numpy as np
import pytest

from avsectester.closed_loop.avstacks.implemented.camera_forward_collision import CameraForwardCollisionAVStack
from avsectester.data_types import Observation


def test_camera_braking(camera_detector_boundary):
    avstack = CameraForwardCollisionAVStack(detector_dataset="nuscenes")
    observation = Observation(0., 0, {"front": np.zeros((100, 100, 3), np.uint8)})
    avstack.reset(observation)
    vehicle = [[20., 20., 60., 50.]]
    for boxes, labels, expected in (
        ([], [], (0., .55, False)),
        (vehicle, [0], (1., 0., True)),
        ([[0., 0., 100., 100.]], [0], (0., .55, False)),
        (vehicle, [6], (0., .55, False)),
    ):
        camera_detector_boundary.boxes, camera_detector_boundary.labels = boxes, labels
        command = avstack(observation)
        assert (command.brake, command.throttle,
                bool(avstack.component_log()["perception"])) == expected
    camera_detector_boundary.boxes, camera_detector_boundary.labels = [[20., 20., 60., 50.]], [0]
    last = avstack(observation)
    camera_detector_boundary.boxes, camera_detector_boundary.labels = [], []
    empty = Observation(.05, 1, {})
    cached = avstack(empty)
    assert cached == last and cached is not last
    last.brake = .25
    cached.brake = .5
    assert avstack(empty).brake == 1.
    assert empty.sensor_data == {} and empty.deliveries == {}
    with pytest.raises(ValueError, match="absent"):
        avstack(Observation(.1, 2, {}, deliveries={"front": ()}))
    avstack.reset(observation)
    with pytest.raises(ValueError, match="absent"):
        avstack(empty)
    with pytest.raises(ValueError, match="area_brake"):
        CameraForwardCollisionAVStack(area_brake=2.)
