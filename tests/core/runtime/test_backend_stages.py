"""Backend intervention stages and the boundary between observations and physical state."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.plane import Control, Observation
from avsectester.simulators.carla import CarlaBackend
from avsectester.simulators.nurec import EgoPose, NuRecBackend, StubRenderer


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
    backend = NuRecBackend({"ego0": {"x": 2.0, "speed": 4.0}}, renderer=renderer)
    observation = backend.reset()
    truth = backend.ground_truth()

    observation.vehicle_state.x = 90
    observation.ego_speed = 100
    observation.sensor_data["front"][:] = 255
    observation.calibration["front"]["intrinsic"][:] = 9
    truth.vehicle_state.x = -90

    assert backend.pose.x == 2
    assert backend.pose.speed == 4
    assert not renderer.image.any()
    np.testing.assert_array_equal(renderer.calibration["front"]["intrinsic"], np.eye(3))
    later = backend.step(Control())
    assert later.vehicle_state.x == pytest.approx(2.2)
    assert observation.vehicle_state.x == 90
    assert backend.ground_truth().vehicle_state is not backend.pose


def test_nurec_world_interventions_change_physics_before_sensing():
    backend = NuRecBackend({"dt": 0.1, "ego0": {"speed": 2.0}}, renderer=RecordingRenderer())
    emitted = []

    class Runtime:
        def emit(self, stage, value, *, sensor=None):
            emitted.append(stage)
            if stage == "world.setup":
                backend.pose.x = 10
            elif stage == "world.step.pre":
                backend.pose.speed = 4
            elif stage == "world.step.post":
                backend.pose.y = 3
            return value

    backend._runtime = Runtime()
    initial = backend.reset()
    moved = backend.step(Control())

    assert initial.vehicle_state.x == 10
    assert moved.vehicle_state.x == pytest.approx(10.4)
    assert moved.vehicle_state.y == 3
    assert moved.ego_speed == 4
    assert emitted == [
        "world.setup", "render.pre", "render.post",
        "world.step.pre", "world.step.post", "render.pre", "render.post",
    ]


def test_nurec_render_replacement_preserves_physical_pose_and_output_sensor_identity():
    from avsectester.runtime import RenderRequest

    renderer = RecordingRenderer()
    renderer.calibration["alternate"] = {"intrinsic": 2 * np.eye(3)}
    backend = NuRecBackend(
        {"ego0": {"x": 2.0}, "camera_aliases": {"front": "camera"}}, renderer=renderer,
    )
    sensors = []

    class Runtime:
        def emit(self, stage, value, *, sensor=None):
            if stage == "render.pre":
                sensors.append(sensor)
                return RenderRequest(EgoPose(x=100), "alternate")
            if stage == "render.post":
                sensors.append(sensor)
                return np.full((2, 3, 3), 123, dtype=np.uint8)
            return value

    backend._runtime = Runtime()
    observation = backend.reset()

    assert renderer.requests[0][0].x == 100
    assert renderer.requests[0][1] == "alternate"
    assert backend.pose.x == 2
    assert observation.vehicle_state.x == 2
    assert list(observation.sensor_data) == ["camera"]
    assert (observation.sensor_data["camera"] == 123).all()
    np.testing.assert_array_equal(observation.calibration["camera"]["intrinsic"], 2 * np.eye(3))
    assert sensors == ["front", "front"]


def test_nurec_visibility_uses_the_updated_render_camera():
    from avsectester.insertion import ActorPose, Insertion, Orientation, PlaneAsset, WorldPlacement
    from avsectester.rendering.cameras import PinholeCamera
    from avsectester.rendering.types import InsertionGeometry
    from avsectester.rendering.visibility import CuboidVisibilityEstimator
    from avsectester.simulators.nurec import NuRecInsertions

    original_camera = PinholeCamera(np.array([[2, 0, 2], [0, 2, 1], [0, 0, 1]]), 4, 2)
    changed_camera = PinholeCamera(np.array([[4, 0, 4], [0, 4, 2], [0, 0, 1]]), 8, 4)
    optical_from_world = np.array([
        [0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1],
    ])
    insertion = Insertion(
        "sign", PlaneAsset(np.full((8, 8, 4), [255, 0, 0, 255], np.uint8), 4, 4),
        WorldPlacement((5, 0, 0)), Orientation("fixed_world", (0, 0, 180)),
    )

    def geometry(_):
        renderer.camera = changed_camera
        return InsertionGeometry({}, ActorPose(np.eye(4)), optical_from_world)

    renderer = NuRecInsertions(
        [insertion], original_camera, geometry, visibility_estimator=CuboidVisibilityEstimator(),
    )
    image = renderer(Observation(0, 0), np.zeros((4, 8, 3), np.uint8))

    assert image.any()
    assert renderer.evidence["sign"].visible_mask.shape == (4, 8)
    assert renderer.evidence["sign"].visibility.fraction == 1


class Velocity:
    def __init__(self, speed):
        self.speed = speed

    def norm(self):
        return self.speed


def test_carla_observation_detaches_shared_sensor_and_state_reference_graph():
    backend = CarlaBackend({}, settle_iters=1)
    reference = SimpleNamespace(x=np.array([1.0, 2.0, 3.0]), q=np.array([1.0, 0, 0, 0]))
    calibration = SimpleNamespace(reference=reference)
    payload = SimpleNamespace(data=np.zeros((2, 3)), calibration=calibration)
    state = SimpleNamespace(reference=reference, velocity=Velocity(5.0))
    snapshot = SimpleNamespace(frame=12, timestamp=SimpleNamespace(elapsed_seconds=1.5))
    backend.client = SimpleNamespace(world=SimpleNamespace(get_snapshot=lambda: snapshot))
    backend.ego = SimpleNamespace(
        reference=reference,
        timestamp=0.0,
        get_pose=lambda: SimpleNamespace(
            position=SimpleNamespace(x=reference.x), attitude=SimpleNamespace(q=reference.q),
        ),
        get_object_state=lambda: state,
        sensor_data_manager=SimpleNamespace(empty=lambda: False, pop=lambda: {"lidar": payload}),
    )
    observation = backend._observe()
    truth = backend.ground_truth()

    copied_reference = observation.sensor_data["lidar"].calibration.reference
    assert copied_reference is observation.vehicle_state.reference
    copied_reference.x[:] = 90
    observation.sensor_data["lidar"].data[:] = 9
    truth.vehicle_state.reference.x[:] = -90

    np.testing.assert_array_equal(reference.x, [1, 2, 3])
    assert not payload.data.any()
    assert backend.ground_truth().ego_speed == 5
    assert truth.frame == 12 and truth.t == 1.5


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
    copied = CarlaBackend({}).copy_observation(observation)

    assert copied.sensor_data["lidar"] is not payload
    assert copied.sensor_data["lidar"].data is capture
    assert copied.vehicle_state is not observation.vehicle_state
    copied.sensor_data["lidar"].calibration["matrix"][:] = 0
    np.testing.assert_array_equal(payload.calibration["matrix"], np.eye(3))
    copied.sensor_data["lidar"].data = np.zeros((1, 4))
    assert payload.data is capture
