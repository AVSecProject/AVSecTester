"""Real avstack sensor/reference objects remain independent of simulator state."""

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("avstack.config")
pytest.importorskip("carla")

from avcarla.geometry import CarlaReferenceFrame
from avstack.geometry import GlobalOrigin3D

from avsectester.data_types import Observation
from avsectester.closed_loop.simulators.carla import CarlaSimulator


def test_copy_observation(native_sensor_case, make_ego):
    make_sensor, payload_array, attributes = native_sensor_case
    body = CarlaReferenceFrame(GlobalOrigin3D, location=(1, 2, 0))
    sensor_reference = CarlaReferenceFrame(body, location=(0, 0, 1))
    sensor = make_sensor(sensor_reference)
    calibration = sensor.calibration
    original = payload_array(sensor.data).copy()
    state = make_ego(xyz=(1, 2, 0), speed=4)
    observation = Observation(.5, 10, {"sensor": sensor}, {"sensor": calibration}, state, 4)

    with CarlaSimulator({}).create_runtime() as runtime:
        copied = runtime.copy_observation(observation)
    copied_sensor = copied.sensor_data["sensor"]
    assert copied_sensor.calibration is copied.calibration["sensor"]
    assert copied_sensor.reference.reference is not body
    copied_sensor.reference.reference.x[:] = 90
    copied.vehicle_state.position.x[:] = -90
    payload_array(copied_sensor.data)[:] = 99
    np.testing.assert_array_equal(payload_array(sensor.data), original)
    np.testing.assert_array_equal(body.x, [1, 2, 0])
    np.testing.assert_array_equal(state.position.x, [1, 2, 0])
    for name, expected in attributes.items():
        assert getattr(copied_sensor, name) == expected
    replacement = deepcopy(copied.calibration["sensor"])
    copied.set_calibration("sensor", replacement)
    assert copied_sensor.calibration is copied.calibration["sensor"] is replacement
    assert copied_sensor.reference is replacement.reference
    assert sensor.calibration is calibration


def test_interpolate_vehicle_state(make_ego):
    left = make_ego(xyz=(0., 0., 0.), yaw=0., speed=2., t=0.)
    right = make_ego(xyz=(10., 2., 0.), yaw=.4, speed=4., t=.1)
    with CarlaSimulator({}).create_runtime() as runtime:
        state = runtime.interpolate_vehicle_state(left, right, .3, 30000)
    np.testing.assert_allclose(state.position.x, [3., .6, 0.])
    expected_velocity = .7 * np.array([2., 0., 0.]) + .3 * 4. * np.array([
        np.cos(.4), np.sin(.4), 0.])
    np.testing.assert_allclose(state.velocity.x, expected_velocity)
    np.testing.assert_allclose(state.attitude.q.components,
                               make_ego(yaw=.12).attitude.q.components, atol=1e-12)
    assert state.t == .03
    np.testing.assert_array_equal(left.position.x, [0., 0., 0.])
