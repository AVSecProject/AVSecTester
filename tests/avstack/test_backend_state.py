"""Real avstack sensor/reference objects remain independent of simulator state."""

from types import SimpleNamespace

import carla
import numpy as np
import pytest

from avcarla.geometry import CarlaReferenceFrame, wrap_mobile_actor_to_object_state
from avstack.calibration import CameraCalibration, GpsCalibration, ImuCalibration, LidarCalibration
from avstack.geometry import GlobalOrigin3D
from avstack.sensors import GpsData, ImageData, ImuData, LidarData

from avsectester.plane import Observation
from avsectester.simulators.carla import CarlaBackend


@pytest.mark.parametrize("kind", ["image", "lidar", "gps", "imu"])
def test_carla_sensor_copy_detaches_real_avstack_reference_graph(kind, make_ego):
    body = CarlaReferenceFrame(GlobalOrigin3D, location=(1, 2, 0))
    sensor_reference = CarlaReferenceFrame(body, location=(0, 0, 1))
    if kind == "image":
        calibration = CameraCalibration(
            sensor_reference,
            np.array([[100, 0, 1, 0], [0, 100, 1, 0], [0, 0, 1, 0]], dtype=float),
            (2, 3, 3),
        )
        sensor = ImageData(0, 0, np.zeros((2, 3, 3), np.uint8), calibration, "0")
    elif kind == "lidar":
        calibration = LidarCalibration(sensor_reference)
        sensor = LidarData(0, 0, np.zeros((3, 4)), calibration, "0", flipy=True)
    elif kind == "gps":
        calibration = GpsCalibration(sensor_reference)
        sensor = GpsData(0, 0, np.zeros(3), calibration, "0", levar=sensor_reference.x)
    else:
        calibration = ImuCalibration(sensor_reference)
        sensor = ImuData(0, 0, {"accelerometer": [0, 0, 0]}, calibration, "0")
    state = make_ego(xyz=(1, 2, 0), speed=4)
    observation = Observation(0, 0, {kind: sensor}, {kind: calibration}, state, 4)

    copied = CarlaBackend({}).copy_observation(observation)
    copied_sensor = copied.sensor_data[kind]
    assert copied_sensor.calibration is copied.calibration[kind]
    assert copied_sensor.reference.reference is not body
    copied_sensor.reference.reference.x[:] = 90
    copied.vehicle_state.position.x[:] = -90
    if kind == "imu":
        copied_sensor.data["accelerometer"][0] = 99
        assert sensor.data["accelerometer"][0] == 0
    else:
        copied_sensor.data[:] = 99
        assert not sensor.data.any()
    np.testing.assert_array_equal(body.x, [1, 2, 0])
    np.testing.assert_array_equal(state.position.x, [1, 2, 0])
    if kind == "lidar":
        assert copied_sensor.flipy is True


@pytest.fixture
def native_truth_backend():
    transform = carla.Transform(carla.Location(11, 22, 3), carla.Rotation(yaw=30))
    native_actor = SimpleNamespace(
        id=1, type_id="vehicle.test", attributes={"number_of_wheels": "4"},
        bounding_box=SimpleNamespace(extent=carla.Vector3D(2, 1, 0.75)),
        get_transform=lambda: transform,
        get_velocity=lambda: carla.Vector3D(3, 4, 0),
        get_acceleration=lambda: carla.Vector3D(),
        get_angular_velocity=lambda: carla.Vector3D(),
    )
    backend = CarlaBackend({})
    clock = SimpleNamespace(frame=12, t=1.5, actor_present=True)
    backend.client = SimpleNamespace(world=SimpleNamespace(get_snapshot=lambda: SimpleNamespace(
        frame=clock.frame, timestamp=SimpleNamespace(elapsed_seconds=clock.t),
        find=lambda _: object() if clock.actor_present else None,
    )))
    backend.ego = SimpleNamespace(
        ID=1, actor=native_actor, timestamp=0.0,
        reference=CarlaReferenceFrame(GlobalOrigin3D, location=(-99, -99, -99)),
    )
    backend.ego.get_object_state = lambda: wrap_mobile_actor_to_object_state(
        backend.ego, backend.ego.timestamp,
    )
    return backend, transform, clock


def test_carla_truth_uses_native_actor_state_before_cached_body_reference_refresh(native_truth_backend):
    backend, transform, clock = native_truth_backend

    initial = backend.ground_truth()
    np.testing.assert_allclose(initial.vehicle_state.position.x, [11, -22, 3])
    assert initial.vehicle_state.attitude.yaw == pytest.approx(-np.pi / 6)
    assert initial.ego_speed == 5

    # A new native snapshot is available before _observe refreshes the sensor body frame.
    transform.location.x = 15
    transform.rotation.yaw = 60
    clock.frame, clock.t = 13, 1.6
    changed = backend.ground_truth()
    np.testing.assert_allclose(changed.vehicle_state.position.x, [15, -22, 3])
    assert changed.vehicle_state.attitude.yaw == pytest.approx(-np.pi / 3)
    assert changed.frame == 13 and changed.t == 1.6
    np.testing.assert_allclose(initial.vehicle_state.position.x, [11, -22, 3])
    np.testing.assert_allclose(backend.ego.reference.x, [-99, -99, -99])


def test_carla_setup_truth_uses_known_spawn_until_actor_is_in_snapshot(native_truth_backend):
    backend, transform, clock = native_truth_backend
    backend.ego.spawn_transform = carla.Transform(
        carla.Location(10, 20, 1), carla.Rotation(yaw=45),
    )
    clock.actor_present = False
    transform.location = carla.Location()
    transform.rotation = carla.Rotation()

    initial = backend.ground_truth()
    np.testing.assert_allclose(initial.vehicle_state.position.x, [10, -20, 1])
    np.testing.assert_allclose(initial.vehicle_state.box.position.x, [10, -20, 1])
    assert initial.vehicle_state.attitude.yaw == pytest.approx(-np.pi / 4)
    assert initial.vehicle_state.box.attitude.yaw == pytest.approx(-np.pi / 4)

    clock.actor_present = True
    transform.location.x = 15
    later = backend.ground_truth()
    np.testing.assert_allclose(later.vehicle_state.position.x, [15, 0, 0])
    assert later.vehicle_state.attitude.yaw == 0
    assert initial.vehicle_state.position.x[0] == 10
