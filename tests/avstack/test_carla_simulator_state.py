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
from avsectester.closed_loop.simulators.carla.sensors import SensorCapture, register_pinhole_camera
from avsectester.serialization import DEFAULT_REGISTRY


def test_copy_observation(native_sensor_case, make_ego):
    make_sensor, payload_array, attributes = native_sensor_case
    body = CarlaReferenceFrame(GlobalOrigin3D, location=(1, 2, 0))
    sensor_reference = CarlaReferenceFrame(body, location=(0, 0, 1))
    sensor = make_sensor(sensor_reference)
    calibration = sensor.calibration
    original = payload_array(sensor.data).copy()
    state = make_ego(xyz=(1, 2, 0), speed=4)
    observation = Observation(.5, 10, {"sensor": sensor}, {"sensor": calibration}, state, 4)

    copied = CarlaSimulator({}).create_runtime().copy_observation(observation)
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


def test_register_pinhole_camera():
    from avcarla.config import CARLA
    from avcarla.sensors import CarlaRgbCamera

    ordinary = CARLA.get("CarlaRgbCamera")
    camera = register_pinhole_camera()
    assert issubclass(camera, CarlaRgbCamera)
    assert CARLA.get("PinholeRgbCamera") is camera
    assert CARLA.get("CarlaRgbCamera") is ordinary
    assert register_pinhole_camera() is camera


def test_sensor_capture():
    source = SensorCapture("carla.LidarMeasurement", np.array([[1., 2., 3., .5]], np.float32).tobytes(),
                           7, .7, {"channels": 16})
    captured = DEFAULT_REGISTRY.snapshot(source)
    source.metadata["channels"] = 32
    assert isinstance(captured, SensorCapture) and captured.metadata == {"channels": 16}
    np.testing.assert_array_equal(np.frombuffer(captured.raw_data, np.float32), [1., 2., 3., .5])
    decoded = DEFAULT_REGISTRY.decode(DEFAULT_REGISTRY.encode(captured))
    assert isinstance(decoded, dict)
    assert decoded["native_type"] == "carla.LidarMeasurement"
    assert decoded["frame"] == 7 and decoded["timestamp"] == .7
    assert decoded["metadata"] == {"channels": 16}
    assert decoded["raw_bytes"] == source.raw_bytes
