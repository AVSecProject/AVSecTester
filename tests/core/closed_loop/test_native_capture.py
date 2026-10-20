"""Native capture stages and isolation between observations and physical state."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Control, Observation
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.nurec import EgoPose, NuRecSimulator, StubRenderer


class RecordingRenderer(StubRenderer):
    def __init__(self):
        super().__init__(cameras=["front"], height=2, width=3)
        self.calibration = {"front": {"intrinsic": np.eye(3)}}
        self.image = np.zeros((2, 3, 3), dtype=np.uint8)
        self.requests = []

    def render(self, pose, camera):
        self.requests.append((deepcopy(pose), camera))
        return self.image


def test_nurec_observations_detach_pose_images_and_calibration():
    renderer = RecordingRenderer()
    simulator_runtime = NuRecSimulator({"ego0": {"x": 2.0, "speed": 4.0}}, renderer=renderer).create_runtime()
    simulator_runtime.reset()
    observation = simulator_runtime.observe()
    truth = simulator_runtime.state()

    observation.vehicle_state.x = 90
    observation.ego_speed = 100
    observation.sensor_data["front"][:] = 255
    observation.calibration["front"]["intrinsic"][:] = 9
    truth.vehicle_state.x = -90

    assert simulator_runtime.pose.x == 2
    assert simulator_runtime.pose.speed == 4
    assert not renderer.image.any()
    np.testing.assert_array_equal(renderer.calibration["front"]["intrinsic"], np.eye(3))
    simulator_runtime.step(Control())
    later = simulator_runtime.observe()
    assert later.vehicle_state.x == pytest.approx(2.2)
    assert observation.vehicle_state.x == 90
    assert simulator_runtime.state().vehicle_state is not simulator_runtime.pose


def test_nurec_world_interventions_change_physics_before_sensing():
    simulator_runtime = NuRecSimulator({"dt": 0.1, "ego0": {"speed": 2.0}}, renderer=RecordingRenderer()).create_runtime()
    emitted = []

    class Dispatcher:
        def emit(self, stage, value, *, sensor=None):
            emitted.append(stage)
            if stage == "simulator.setup":
                simulator_runtime.pose.x = 10
            elif stage == "simulator.step.pre":
                simulator_runtime.pose.speed = 4
            elif stage == "simulator.step.post":
                simulator_runtime.pose.y = 3
            return value

    simulator_runtime._dispatcher = Dispatcher()
    simulator_runtime.reset()
    initial = simulator_runtime.observe()
    simulator_runtime.step(Control())
    moved = simulator_runtime.observe()
    assert initial.vehicle_state.x == 10
    assert moved.vehicle_state.x == pytest.approx(10.4)
    assert moved.vehicle_state.y == 3
    assert moved.ego_speed == 4
    assert emitted == [
        "simulator.setup", "render.pre", "render.post",
        "simulator.step.pre", "simulator.step.post", "render.pre", "render.post",
    ]


def test_nurec_render_replacement_preserves_physical_pose_and_output_sensor_identity():
    from avsectester.closed_loop.hooks import CameraRenderRequest

    renderer = RecordingRenderer()
    renderer.calibration["alternate"] = {"intrinsic": 2 * np.eye(3)}
    simulator_runtime = NuRecSimulator(
        {"ego0": {"x": 2.0}, "camera_aliases": {"front": "camera"}}, renderer=renderer,
    ).create_runtime()
    sensors = []

    class Dispatcher:
        def emit(self, stage, value, *, sensor=None):
            if stage == "render.pre":
                sensors.append(sensor)
                return CameraRenderRequest(EgoPose(x=100), "alternate")
            if stage == "render.post":
                sensors.append(sensor)
                return np.full((2, 3, 3), 123, dtype=np.uint8)
            return value

    simulator_runtime._dispatcher = Dispatcher()
    simulator_runtime.reset()
    observation = simulator_runtime.observe()
    assert renderer.requests[0][0].x == 100
    assert renderer.requests[0][1] == "alternate"
    assert simulator_runtime.pose.x == 2
    assert observation.vehicle_state.x == 2
    assert list(observation.sensor_data) == ["camera"]
    assert (observation.sensor_data["camera"] == 123).all()
    np.testing.assert_array_equal(observation.calibration["camera"]["intrinsic"], 2 * np.eye(3))
    assert sensors == ["front", "front"]


class Velocity:
    def __init__(self, speed):
        self.speed = speed

    def norm(self):
        return self.speed


def test_carla_observation_detaches_shared_sensor_and_state_reference_graph():
    simulator_runtime = CarlaSimulator({}, settle_iters=1).create_runtime()
    reference = SimpleNamespace(x=np.array([1.0, 2.0, 3.0]), q=np.array([1.0, 0, 0, 0]))
    calibration = SimpleNamespace(reference=reference)
    payload = SimpleNamespace(data=np.zeros((2, 3)), calibration=calibration, frame=12)
    state = SimpleNamespace(reference=reference, velocity=Velocity(5.0))
    snapshot = SimpleNamespace(frame=12, timestamp=SimpleNamespace(elapsed_seconds=1.5))
    simulator_runtime.client = SimpleNamespace(world=SimpleNamespace(get_snapshot=lambda: snapshot))
    native_ego = SimpleNamespace(
        type_id="vehicle.test", get_transform=lambda: SimpleNamespace(get_matrix=lambda: np.eye(4)),
        get_velocity=lambda: SimpleNamespace(x=5., y=0., z=0.),
        bounding_box=SimpleNamespace(extent=SimpleNamespace(x=2., y=1., z=.7)))
    simulator_runtime.ego = SimpleNamespace(
        actor=native_ego, reference=reference,
        timestamp=0.0,
        get_pose=lambda: SimpleNamespace(
            position=SimpleNamespace(x=reference.x), attitude=SimpleNamespace(q=reference.q),
        ),
        get_object_state=lambda: state,
        sensors={"lidar": SimpleNamespace(source_identifier="lidar", frame0=0)},
        sensor_data_manager=SimpleNamespace(empty=lambda: False, has_data=lambda name: True,
                                           pop=lambda name: payload),
    )
    observation = simulator_runtime._observe()
    truth = simulator_runtime._state()

    assert observation.calibration["lidar"] is observation.sensor_data["lidar"].calibration
    copied_reference = observation.sensor_data["lidar"].calibration.reference
    assert copied_reference is observation.vehicle_state.reference
    copied_reference.x[:] = 90
    observation.sensor_data["lidar"].data[:] = 9
    truth.vehicle_state.reference.x[:] = -90

    np.testing.assert_array_equal(reference.x, [1, 2, 3])
    assert not payload.data.any()
    assert simulator_runtime._state().ego_speed == 5
    assert truth.frame == 12 and truth.t == 1.5
    replacement = SimpleNamespace(reference=SimpleNamespace(x=np.zeros(3)))
    observation.set_calibration("lidar", replacement)
    assert observation.calibration["lidar"] is replacement
    assert observation.sensor_data["lidar"].calibration is replacement
    assert payload.calibration is calibration
    with pytest.raises(KeyError):
        observation.set_calibration("missing", replacement)


def test_carla_copy_borrows_native_capture_and_detaches_its_wrapper():
    class NativeCapture:
        __module__ = "carla.libcarla"

        @property
        def raw_data(self):
            return b"immutable capture"

        def __deepcopy__(self, _):
            raise RuntimeError("Native capture cannot be pickled")

    capture = NativeCapture()
    payload = SimpleNamespace(data=capture, calibration={"matrix": np.eye(3)})
    observation = Observation(0, 0, {"lidar": payload}, vehicle_state=EgoPose())
    copied = CarlaSimulator({}).create_runtime().copy_observation(observation)

    assert copied.sensor_data["lidar"] is not payload
    assert copied.sensor_data["lidar"].data is capture
    assert copied.vehicle_state is not observation.vehicle_state
    copied.sensor_data["lidar"].calibration["matrix"][:] = 0
    np.testing.assert_array_equal(payload.calibration["matrix"], np.eye(3))
    copied.sensor_data["lidar"].data = np.zeros((1, 4))
    assert payload.data is capture


def test_nurec_unknown_camera_geometry_fails_before_render_request_lookup():
    simulator_runtime = NuRecSimulator({}, renderer=RecordingRenderer()).create_runtime()
    simulator_runtime.reset()
    with pytest.raises(ValueError, match="Unknown NuRec camera 'missing'"):
        simulator_runtime.sensor_geometry("missing")
    with pytest.raises(ValueError, match="Unknown NuRec camera 'missing'"):
        simulator_runtime.render_request("missing")
    assert simulator_runtime.camera_sensors == frozenset()
