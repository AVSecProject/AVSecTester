"""CARLA selection resource lifetime, native frame access and depth alignment."""

from copy import deepcopy
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.insertion import (
    AttachedPlacement,
    Insertion,
    Orientation,
    PlaneAsset,
    WorldPlacement,
    rotation_matrix,
)
from avsectester.plane import Observation
from avsectester.scenarios.carla_gt import carla_actor_pose
from avsectester.scenarios.carla_provider import (
    CarlaCandidateProvider,
    CarlaSelectionBackend,
    decode_carla_depth,
    depth_visibility,
    register_pinhole_camera,
    _require_pinhole_sensor,
)
from avsectester.scenarios.context import FilterContext
from avsectester.scenarios.filters import Constraint, FilterResult
from avsectester.scenarios.requirement import InitialWindow, ScenarioRequirement
from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, SceneGT
from avsectester.simulators.carla import CarlaBackend, insertion_perturbation


def depth_image(value, frame=1):
    code = int(round(value / 1000 * (2**24 - 1)))
    pixels = np.array([[[code >> 16, (code >> 8) & 255, code & 255, 255]]], np.uint8)
    return SimpleNamespace(raw_data=pixels.tobytes(), height=1, width=1, frame=frame)


def test_depth_decode_channel_order_and_range():
    for value in (0, 1, 17.25, 999, 1000):
        assert decode_carla_depth(depth_image(value))[0, 0] == pytest.approx(value, abs=0.00003)


def test_selection_camera_configuration_is_explicit_and_does_not_modify_input():
    config = {"ego": {"sensors": [{"type": "CarlaRgbCamera"}, {"type": "CustomCamera"}]}}
    backend = CarlaSelectionBackend(config)
    assert config["ego"]["sensors"][0]["type"] == "CarlaRgbCamera"
    assert backend.scenario["ego"]["sensors"][0]["type"] == "PinholeRgbCamera"
    assert backend.scenario["ego"]["sensors"][1]["type"] == "CustomCamera"
    _require_pinhole_sensor(SimpleNamespace(attributes={"lens_k": "0"}))
    with pytest.raises(ValueError, match="undistorted"):
        _require_pinhole_sensor(SimpleNamespace(attributes={"lens_k": "-1"}))
    with pytest.raises(ValueError, match="undistorted"):
        _require_pinhole_sensor(
            SimpleNamespace(attributes={"lens_k": "0", "lens_circle_multiplier": "1"})
        )


def test_pinhole_registration_sets_lens_before_spawn_and_remains_lazy(monkeypatch):
    import sys

    class Registry:
        module_dict = {}

        def register_module(self):
            def register(cls):
                self.module_dict[cls.__name__] = cls
                return cls

            return register

    class RgbSensor:
        def __init__(self, *, do_spawn):
            assert not do_spawn
            self.attributes = {"image_size_x": 100}
            self.lens = {}
            self.bp = SimpleNamespace(
                set_attribute=lambda key, value: self.lens.update({key: value})
            )
            self.spawned = False

        def spawn(self):
            assert self.lens == {"lens_k": "0", "lens_kcube": "0", "lens_circle_multiplier": "0"}
            self.spawned = True

    registry = Registry()
    monkeypatch.setitem(sys.modules, "avcarla.config", SimpleNamespace(CARLA=registry))
    monkeypatch.setitem(sys.modules, "avcarla.sensors", SimpleNamespace(CarlaRgbCamera=RgbSensor))
    sensor_type = register_pinhole_camera()
    assert register_pinhole_camera() is sensor_type
    assert sensor_type().spawned
    assert not sensor_type(do_spawn=False).spawned


def test_selection_observation_consumes_matching_depth_frame(monkeypatch):
    monkeypatch.setattr(CarlaBackend, "_observe", lambda _: Observation(1.0, 12))
    backend = object.__new__(CarlaSelectionBackend)
    backend._depth_queue = Queue()
    backend._depth_queue.put(depth_image(3, frame=11))
    backend._depth_queue.put(depth_image(8, frame=12))
    observation = backend._observe()
    assert observation.frame == backend.depth_frame == 12
    assert backend.depth[0, 0] == pytest.approx(8, abs=0.00003)
    backend._depth_queue.put(depth_image(10, frame=13))
    with pytest.raises(RuntimeError, match="frames differ"):
        backend._observe()


def test_actor_pose_includes_rotated_offset_bounding_box(monkeypatch):
    import sys

    # A CARLA transform is left-handed. Reflection is applied to both frame bases.
    world = np.eye(4)
    world[:3, :3] = rotation_matrix((10, -20, 35))
    world[:3, 3] = [30, 40, 2]
    local = np.eye(4)
    local[:3, :3] = rotation_matrix((5, 0, 12))
    local[:3, 3] = [0.3, 0.2, 0.8]
    monkeypatch.setitem(
        sys.modules,
        "carla",
        SimpleNamespace(
            Transform=lambda *_: SimpleNamespace(get_matrix=lambda: local),
        ),
    )
    actor = SimpleNamespace(
        get_transform=lambda: SimpleNamespace(get_inverse_matrix=lambda: np.linalg.inv(world)),
        bounding_box=SimpleNamespace(
            rotation=object(),
            location=SimpleNamespace(x=0.3, y=0.2, z=0.8),
            extent=SimpleNamespace(x=2, y=1, z=0.7),
        ),
    )
    result = carla_actor_pose(actor)
    reflection = np.diag([1, -1, 1, 1])
    np.testing.assert_allclose(result.transform, reflection @ world @ local @ reflection)
    assert result.extent == (4, 2, 1.4)
    assert np.linalg.det(result.transform[:3, :3]) == pytest.approx(1)


class PreviewBackend:
    def __init__(self, scenario):
        self.scenario = scenario
        self.replay_scenario = None
        self.frame = 0
        self.closed = False
        self.prepared = False

    def prepare_clean_attack_pair(self):
        self.prepared = True

    def reset(self):
        assert self.prepared
        self.frame = 10
        self.replay_scenario = {**deepcopy(self.scenario), "resolved_spawn": 7}

    def selection_context(self):
        scene = SceneGT(self.frame, self.frame / 10, EgoState(), {}, [])
        return FilterContext(scene, backend=self, native={"world": self})

    def close(self):
        self.closed = True


def test_provider_evaluates_native_data_at_each_initial_frame_and_replays_first_case():
    instances, visited = [], []

    def factory(scenario):
        instance = PreviewBackend(scenario)
        instances.append(instance)
        return instance

    def advance(backend, index):
        backend.frame += 1

    class NativeFrame(Constraint):
        def evaluate(self, context):
            visited.append((context.scene.frame, context.native["world"].frame))
            return FilterResult.from_bool(
                context.scene.frame == context.native["world"].frame, "Native frame matches scene"
            )

    provider = CarlaCandidateProvider(
        initial_frames=5, advance_initial=advance, backend_factory=factory
    )
    original = {"weather": "ClearNoon"}
    req = ScenarioRequirement("native", constraints=[NativeFrame()], window=InitialWindow(3))
    with provider(original) as context:
        assert instances[0].frame == 10  # Opening the candidate does not consume the sequence.
        result = req.evaluate(context)
        assert result.status == "pass"
        assert visited == [(10, 10), (11, 11), (12, 12)]
        assert instances[0].frame == 12  # Never consume unused prepared frames.
        replay = context.native["make_backend"]
        with pytest.raises(RuntimeError, match="new CARLA candidate"):
            req.evaluate(context)
    assert instances[0].closed
    original["weather"] = "WetCloudyNoon"
    fresh = replay()
    assert fresh is not instances[0]
    assert fresh.scenario == {"weather": "ClearNoon", "resolved_spawn": 7}


def test_provider_always_cleans_up_failed_preparation_and_filter():
    instances = []

    class Broken(PreviewBackend):
        def reset(self):
            raise RuntimeError("reset failed")

    def factory(config):
        instance = Broken(config)
        instances.append(instance)
        return instance

    with pytest.raises(RuntimeError, match="reset failed"):
        with CarlaCandidateProvider(backend_factory=factory)({}):
            pass
    assert instances[0].closed

    instance = PreviewBackend({})
    with pytest.raises(RuntimeError, match="filter failed"):
        with CarlaCandidateProvider(backend_factory=lambda _: instance)({}):
            raise RuntimeError("filter failed")
    assert instance.closed


def test_provider_requires_an_explicit_prescribed_sequence():
    with pytest.raises(ValueError, match="prescribed"):
        CarlaCandidateProvider(initial_frames=2)
    with pytest.raises(ValueError, match="positive integer"):
        CarlaCandidateProvider(initial_frames=True)


def test_provider_visibility_uses_this_frame_depth_and_other_insertions():
    optical_to_ego = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    calibration = CameraCalib("front", 100, 80, (50, 50, 40), optical_to_ego)
    texture = np.full((8, 8, 4), 255, np.uint8)
    surface = PlaneAsset(texture, 4, 2)
    items = (
        Insertion(
            "rear", surface, WorldPlacement((10, 0, 0)), Orientation("fixed_world", (0, 0, 180))
        ),
        Insertion(
            "front", surface, WorldPlacement((5, 0, 0)), Orientation("fixed_world", (0, 0, 180))
        ),
    )
    context = FilterContext(
        SceneGT(10, 1, EgoState(pose=np.eye(4)), {"front": calibration}, []),
        insertions=items,
        native={"depth": {"front": np.full((80, 100), 100)}},
    )
    assert depth_visibility(context, "rear", "front").visibility.fraction == 0
    assert depth_visibility(context, "front", "front").visibility.fraction == 1
    assert depth_visibility(context, "rear", "missing") is None


def test_insertion_updates_model_input_with_attached_and_world_positions():
    optical_to_ego = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    calibration = CameraCalib("front", 100, 80, (50, 50, 40), optical_to_ego)
    red = np.full((8, 8, 4), [255, 0, 0, 255], np.uint8)
    blue = np.full((8, 8, 4), [0, 0, 255, 255], np.uint8)
    items = (
        Insertion(
            "patch",
            PlaneAsset(red, 1, 1),
            AttachedPlacement("attacker", "rear_center", (-0.1, 0, 0)),
            Orientation("follow_host", (0, 0, 180)),
        ),
        Insertion(
            "sign",
            PlaneAsset(blue, 1, 1),
            WorldPlacement((8, -3, 0)),
            Orientation("fixed_world", (0, 0, 180)),
        ),
    )
    states = {}
    for frame, host_y in ((10, 0), (11, 2)):
        scene = SceneGT(
            frame,
            frame / 10,
            EgoState(pose=np.eye(4)),
            {"front": calibration},
            [
                ObjectGT("host", "vehicle", (10, host_y, 0), (4, 2, 2)),
            ],
        )
        states[frame] = FilterContext(scene, native={"depth": {"front": np.full((80, 100), 100)}})
    backend = SimpleNamespace(frame=10)
    backend.selection_context = lambda: states[backend.frame]
    backend.ground_truth = lambda: SimpleNamespace(frame=backend.frame)
    perturb = insertion_perturbation(backend, items, bindings={"attacker": ("host",)})
    images = []
    for frame in (10, 11):
        backend.frame = frame
        original = Observation(frame / 10, frame, {"front": np.zeros((80, 100, 3), np.uint8)})
        changed = perturb(original)
        assert changed is not original
        assert not original.sensor_data["front"].any()
        images.append(changed.sensor_data["front"])
    red_centres, blue_centres = [], []
    for image in images:
        red_centres.append(np.nonzero(image[..., 0] > 127)[1].mean())
        blue_centres.append(np.nonzero(image[..., 2] > 127)[1].mean())
    assert red_centres[1] < red_centres[0] - 10
    assert blue_centres[0] == blue_centres[1]
    # A recreated camera must use its current calibration, not the previous reset's lens.
    context = states[11]
    context.scene.cameras["front"] = CameraCalib("front", 200, 160, (100, 100, 80), optical_to_ego)
    context.native["depth"]["front"] = np.full((160, 200), 100)
    larger = perturb(Observation(1.1, 11, {"front": np.zeros((160, 200, 3), np.uint8)}))
    assert larger.sensor_data["front"].shape == (160, 200, 3)
    assert larger.sensor_data["front"].any()
    # Reject stale physical geometry, independent of the model-visible frame number.
    backend.ground_truth = lambda: SimpleNamespace(frame=10)
    with pytest.raises(ValueError, match="same frame"):
        perturb(Observation(1, 10, {"front": np.zeros((80, 100, 3), np.uint8)}))
    backend.ground_truth = lambda: SimpleNamespace(frame=11)
    states[11].scene.objects.clear()
    with pytest.raises(KeyError, match="Selected insertion actors unavailable"):
        perturb(Observation(1.1, 11, {"front": np.zeros((80, 100, 3), np.uint8)}))


def test_insertion_uses_selected_sensor_source_and_its_absolute_frame():
    class ImagePayload:
        def __init__(self, frame):
            self.frame = frame
            self.data = np.zeros((80, 100, 3), np.uint8)

        @property
        def rgb_image(self):
            return self.data

    optical_to_ego = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    scene = SceneGT(
        10,
        1,
        EgoState(pose=np.eye(4)),
        {
            "front": CameraCalib("front", 100, 80, (50, 50, 40), optical_to_ego),
        },
        [],
    )
    context = FilterContext(
        scene,
        native={
            "depth": {"front": np.full((80, 100), 100)},
            "camera_sources": {"front": "camera-0-0"},
            "camera_frame_offsets": {"front": 7},
        },
    )
    texture = np.full((8, 8, 4), [255, 0, 0, 255], np.uint8)
    item = Insertion("sign", PlaneAsset(texture, 1, 1), WorldPlacement((10, 0, 0)))
    perturb = insertion_perturbation(SimpleNamespace(
        selection_context=lambda: context,
        ground_truth=lambda: SimpleNamespace(frame=10),
    ), (item,))
    other = np.full((80, 100, 3), 77, np.uint8)
    original = Observation(999, 999, {"other-camera": other, "camera-0-0": ImagePayload(frame=3)})
    changed = perturb(original)
    assert changed.sensor_data["other-camera"] is other
    assert not original.sensor_data["camera-0-0"].rgb_image.any()
    assert changed.sensor_data["camera-0-0"].rgb_image.any()
    original.sensor_data["camera-0-0"].frame = 2
    with pytest.raises(ValueError, match="payload and geometry"):
        perturb(original)
    del original.sensor_data["camera-0-0"]
    with pytest.raises(ValueError, match="Missing insertion camera"):
        perturb(original)
