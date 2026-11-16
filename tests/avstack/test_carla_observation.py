"""Real avstack sensor/reference objects remain independent of simulator state."""

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("avstack.config")
pytest.importorskip("carla")

from avcarla.geometry import CarlaReferenceFrame
from avstack.geometry import GlobalOrigin3D

from avsectester.data_types import Observation, TimedStateEstimate, interpolate_state_estimate
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


def test_interpolate_state_estimate(make_ego):
    left = make_ego(xyz=(0, 0, 0), speed=2)
    right = make_ego(xyz=(10, 0, 0), speed=4)
    first = TimedStateEstimate(0, 0., left, 2.)
    last = TimedStateEstimate(100000, .1, right, 4.)
    assert interpolate_state_estimate((first, last), 0) is first
    midpoint = interpolate_state_estimate((first, last), 50000)
    np.testing.assert_allclose(midpoint.vehicle_state.position.x, [5., 0., 0.])
    np.testing.assert_allclose(midpoint.vehicle_state.velocity.x, [3., 0., 0.])
    assert midpoint.vehicle_state.attitude.q == left.attitude.q
    assert midpoint.vehicle_state.t == midpoint.t == .05
    np.testing.assert_array_equal(left.position.x, [0., 0., 0.])
